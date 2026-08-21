// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// GenerationSession : one live KV slot. Prefix reuse:
// n_past = lcp(cached, new); if n_past == new.count { n_past -= 1 };
// truncate (or rebuild if a layer can't), prefill only the suffix.

import Foundation

/// Persistent generation slot: layer caches + the token ids they hold.
///
/// `Sendable` is `@unchecked` because the session is shared across HTTP
/// Tasks; every mutable field is under `lock`.
public final class GenerationSession: @unchecked Sendable {
    private let lock = NSLock()
    private let admit = AsyncLock()
    private let model: Model?
    private let makeCaches: @Sendable () -> [any LayerCacheProtocol]
    private var caches: [any LayerCacheProtocol]
    private var cachedTokens: [Int]
    private var lastNPast: Int = 0
    private var producerTask: Task<Void, Never>?

    public init(model: Model) {
        self.model = model
        self.makeCaches = { model.engine.makeLayerCaches() }
        self.caches = model.engine.makeLayerCaches()
        self.cachedTokens = []
    }

    /// Test hook: inject fake caches (no weights). `complete` is unavailable.
    internal init(makeCaches: @escaping @Sendable () -> [any LayerCacheProtocol]) {
        self.model = nil
        self.makeCaches = makeCaches
        self.caches = makeCaches()
        self.cachedTokens = []
    }

    public var cachedTokenCount: Int {
        lock.withLock { cachedTokens.count }
    }

    /// Prefix tokens reused on the last `prepare` / `complete` (usage.cached_tokens).
    public var lastCachedTokens: Int {
        lock.withLock { lastNPast }
    }

    /// Cancel the in-flight producer so prefill checks cancellation
    /// between GPU chunks without waiting for the first yield.
    public func abort() {
        lock.withLock { producerTask }?.cancel()
    }

    /// Longest common prefix length of two token sequences.
    public static func commonPrefix(_ a: [Int], _ b: [Int]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n, a[i] == b[i] { i += 1 }
        return i
    }

    /// LCP, then `-1` when the new prompt is fully cached so we still
    /// have a position to compute logits from.
    public static func nPast(cached: [Int], new: [Int]) -> Int {
        guard !new.isEmpty else { return 0 }
        var n = commonPrefix(cached, new)
        if n == new.count { n -= 1 }
        return n
    }

    /// Stream a completion, reusing KV for the shared prefix.
    /// Cancel the consuming `Task` (or the returned stream) to abort
    /// prefill and decode.
    public func complete(
        promptTokens: [Int],
        parameters: GenerationParameters? = nil,
        profile: Profile = .shared
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task(priority: .high) {
                await self.admit.acquire()
                defer { self.admit.release() }
                do {
                    try Task.checkCancellation()
                    let (model, caches, nPast) = try self.begin(promptTokens: promptTokens)
                    let params = parameters ?? model.defaultGenerationParameters
                    let generated = try await model.driveGeneration(
                        promptTokens: promptTokens,
                        parameters: params,
                        profile: profile,
                        continuation: continuation,
                        sessionCaches: caches,
                        prefillStart: nPast
                    )
                    self.commit(promptTokens: promptTokens, generated: generated)
                    continuation.finish()
                } catch is CancellationError {
                    self.invalidate()
                    continuation.finish()
                } catch {
                    self.invalidate()
                    continuation.finish(throwing: error)
                }
                self.lock.withLock { self.producerTask = nil }
            }
            self.lock.withLock { self.producerTask = producer }
            continuation.onTermination = { @Sendable _ in
                producer.cancel()
            }
        }
    }

    // MARK: - Internals (tested)

    /// Truncate-or-rebuild to `nPast`. Returns the n_past actually used
    /// (0 if caches had to be rebuilt).
    @discardableResult
    internal func prepare(for newIds: [Int]) -> Int {
        lock.withLock {
            prepareLocked(for: newIds)
        }
    }

    internal func snapshotCachedTokens() -> [Int] {
        lock.withLock { cachedTokens }
    }

    internal func testingReplaceCaches(
        _ caches: [any LayerCacheProtocol], tokens: [Int]
    ) {
        lock.withLock {
            self.caches = caches
            self.cachedTokens = tokens
        }
    }

    private func begin(promptTokens: [Int]) throws -> (
        Model, [any LayerCacheProtocol], Int
    ) {
        try lock.withLock {
            guard let model else {
                throw ServeError.sessionHasNoModel
            }
            let nPast = prepareLocked(for: promptTokens)
            return (model, caches, nPast)
        }
    }

    private func prepareLocked(for newIds: [Int]) -> Int {
        var nPast = Self.nPast(cached: cachedTokens, new: newIds)
        if !tryTruncateCachesLocked(to: nPast) {
            Device.shared.releaseGenerationCaches()
            caches = makeCaches()
            cachedTokens = []
            nPast = 0
        } else {
            cachedTokens = Array(newIds.prefix(nPast))
        }
        lastNPast = nPast
        return nPast
    }

    private func tryTruncateCachesLocked(to nPast: Int) -> Bool {
        for cache in caches {
            if let kv = cache as? KVCacheProtocol {
                if nPast > kv.length { return false }
                switch kv.eviction {
                case .unbounded:
                    break
                case .window(let maxSize, _):
                    if kv.absolutePosition > maxSize { return false }
                }
            } else if nPast != cache.length {
                // Recurrent (GDN/SSM/conv) cannot rewind; keep only on append.
                return false
            }
        }
        for cache in caches {
            if let kv = cache as? KVCacheProtocol {
                kv.truncate(toLength: nPast)
            }
        }
        return true
    }

    private func commit(promptTokens: [Int], generated: [Int]) {
        lock.withLock {
            cachedTokens = promptTokens + generated
        }
    }

    private func invalidate() {
        lock.withLock {
            Device.shared.releaseGenerationCaches()
            caches = makeCaches()
            cachedTokens = []
            lastNPast = 0
        }
    }
}

/// FIFO admit so two HTTP requests cannot interleave on one KV slot.
/// Waiters park on continuations (no thread-pool `wait()`).
final class AsyncLock: @unchecked Sendable {
    private let lock = NSLock()
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            self.lock.lock()
            if !self.busy {
                self.busy = true
                self.lock.unlock()
                c.resume()
                return
            }
            self.waiters.append(c)
            self.lock.unlock()
        }
    }

    func release() {
        lock.lock()
        if waiters.isEmpty {
            busy = false
            lock.unlock()
            return
        }
        let next = waiters.removeFirst()
        lock.unlock()
        next.resume()
    }
}

public enum ServeError: Error, CustomStringConvertible, Sendable {
    case sessionHasNoModel
    case badRequest(String)
    case notFound
    case methodNotAllowed

    public var description: String {
        switch self {
        case .sessionHasNoModel:
            return "GenerationSession was created without a Model"
        case .badRequest(let m):
            return m
        case .notFound:
            return "not found"
        case .methodNotAllowed:
            return "method not allowed"
        }
    }
}
