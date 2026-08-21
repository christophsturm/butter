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
// ToolCallParser: recover structured tool_calls from leaked markup
// (JSON fences and family XML). Lives next to chat templating.

import Foundation

public struct ParsedToolCall: Sendable, Equatable {
    public var id: String
    public var name: String
    /// JSON object as a string (OpenAI contract).
    public var arguments: String

    public init(id: String, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

public enum ToolCallParser {
    /// Matched markup must occupy at least this fraction of `content`.
    public static let recoveryMinRatio: Double = 0.5

    public static func recover(from content: String) -> (calls: [ParsedToolCall], residual: String)? {
        let visible = stripThinkBlocks(content)
        if let r = recoverJSONFence(visible) { return r }
        if let r = recoverQwen3Coder(visible) { return r }
        if let r = recoverPhi4Pipe(visible) { return r }
        if let r = recoverMistralBracket(visible) { return r }
        return nil
    }

    /// Drop `<think>...</think>` (and an unclosed `<think>` tail) so
    /// reasoning text cannot look like a tool leak.
    public static func stripThinkBlocks(_ content: String) -> String {
        var s = content
        while let start = s.range(of: "<think>") {
            if let end = s.range(of: "</think>", range: start.lowerBound..<s.endIndex) {
                s.removeSubrange(start.lowerBound..<end.upperBound)
            } else {
                s.removeSubrange(start.lowerBound..<s.endIndex)
                break
            }
        }
        return s
    }

    public static func looksLikeToolMarkup(_ text: String) -> Bool {
        text.contains("<tool_call") || text.contains("<|tool_call")
            || text.contains("[TOOL_CALLS]")
    }

    public static func makeId(_ seed: Int) -> String {
        String(format: "chatcmpl-tool-recovered-%012d", seed)
    }

    // MARK: - Shapes

    static func recoverJSONFence(_ content: String) -> (calls: [ParsedToolCall], residual: String)? {
        let matches = matchAll(
            content, pattern: #"<tool_call>\s*(\{.*?\})\s*</tool_call>"#)
        guard !matches.isEmpty else { return nil }
        var calls: [ParsedToolCall] = []
        for (i, m) in matches.enumerated() {
            guard let obj = parseJSONObject(m.groups[0]),
                let fn = validateFunction(name: obj["name"], arguments: obj["arguments"])
            else { continue }
            calls.append(
                ParsedToolCall(id: makeId(i), name: fn.name, arguments: fn.arguments))
        }
        return finish(content: content, matches: matches, calls: calls)
    }

    static func recoverQwen3Coder(_ content: String) -> (
        calls: [ParsedToolCall], residual: String
    )? {
        let matches = matchAll(
            content,
            pattern: #"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>"#)
        guard !matches.isEmpty else { return nil }
        let paramRe = try! NSRegularExpression(
            pattern: #"<parameter=([^>]+)>(.*?)</parameter>"#,
            options: [.dotMatchesLineSeparators])
        var calls: [ParsedToolCall] = []
        for (i, m) in matches.enumerated() {
            let name = m.groups[0].trimmingCharacters(in: .whitespacesAndNewlines)
            let body = m.groups[1]
            var params: [String: String] = [:]
            let ns = body as NSString
            for pm in paramRe.matches(in: body, range: NSRange(location: 0, length: ns.length))
            {
                guard pm.numberOfRanges >= 3,
                    let kr = Range(pm.range(at: 1), in: body),
                    let vr = Range(pm.range(at: 2), in: body)
                else { continue }
                params[String(body[kr]).trimmingCharacters(in: .whitespacesAndNewlines)] =
                    String(body[vr]).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            guard let fn = validateFunction(name: name, arguments: params) else { continue }
            calls.append(
                ParsedToolCall(id: makeId(i), name: fn.name, arguments: fn.arguments))
        }
        return finish(content: content, matches: matches, calls: calls)
    }

    static func recoverPhi4Pipe(_ content: String) -> (calls: [ParsedToolCall], residual: String)? {
        let matches = matchAll(
            content,
            pattern: #"<\|tool_calls?\|>\s*(\[.*?\]|\{.*?\})\s*<\|/tool_calls?\|>"#)
        guard !matches.isEmpty else { return nil }
        var calls: [ParsedToolCall] = []
        var seq = 0
        for m in matches {
            guard let value = parseJSON(m.groups[0]) else { continue }
            let items: [[String: Any]]
            if let arr = value as? [Any] {
                items = arr.compactMap { $0 as? [String: Any] }
            } else if let obj = value as? [String: Any] {
                items = [obj]
            } else {
                continue
            }
            for obj in items {
                guard let fn = validateFunction(name: obj["name"], arguments: obj["arguments"])
                else { continue }
                calls.append(
                    ParsedToolCall(id: makeId(seq), name: fn.name, arguments: fn.arguments))
                seq += 1
            }
        }
        return finish(content: content, matches: matches, calls: calls)
    }

    static func recoverMistralBracket(_ content: String) -> (
        calls: [ParsedToolCall], residual: String
    )? {
        let matches = matchAll(content, pattern: #"\[TOOL_CALLS\]\s*(\[.*?\])"#)
        guard !matches.isEmpty else { return nil }
        var calls: [ParsedToolCall] = []
        var seq = 0
        for m in matches {
            guard let arr = parseJSON(m.groups[0]) as? [Any] else { continue }
            for obj in arr {
                guard let dict = obj as? [String: Any],
                    let fn = validateFunction(name: dict["name"], arguments: dict["arguments"])
                else { continue }
                calls.append(
                    ParsedToolCall(id: makeId(seq), name: fn.name, arguments: fn.arguments))
                seq += 1
            }
        }
        return finish(content: content, matches: matches, calls: calls)
    }

    // MARK: - Helpers

    private struct RegexMatch {
        let range: Range<String.Index>
        let groups: [String]
    }

    private static func matchAll(_ content: String, pattern: String) -> [RegexMatch] {
        guard let re = try? NSRegularExpression(
            pattern: pattern, options: [.dotMatchesLineSeparators])
        else { return [] }
        let ns = content as NSString
        let full = NSRange(location: 0, length: ns.length)
        return re.matches(in: content, range: full).compactMap { m in
            guard let r = Range(m.range, in: content) else { return nil }
            var groups: [String] = []
            for i in 1 ..< m.numberOfRanges {
                if m.range(at: i).location == NSNotFound {
                    groups.append("")
                } else if let gr = Range(m.range(at: i), in: content) {
                    groups.append(String(content[gr]))
                }
            }
            return RegexMatch(range: r, groups: groups)
        }
    }

    private static func finish(
        content: String, matches: [RegexMatch], calls: [ParsedToolCall]
    ) -> (calls: [ParsedToolCall], residual: String)? {
        guard !calls.isEmpty else { return nil }
        let matchedLen = matches.reduce(0) { $0 + content.distance(from: $1.range.lowerBound, to: $1.range.upperBound) }
        if Double(matchedLen) < Double(content.count) * recoveryMinRatio { return nil }
        var residual = ""
        var cursor = content.startIndex
        for m in matches {
            residual += content[cursor ..< m.range.lowerBound]
            cursor = m.range.upperBound
        }
        residual += content[cursor...]
        return (calls, residual.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    private static func validateFunction(name: Any?, arguments: Any?) -> (
        name: String, arguments: String
    )? {
        guard let name = name as? String, !name.isEmpty else { return nil }
        if let s = arguments as? String {
            guard let data = s.data(using: .utf8),
                (try? JSONSerialization.jsonObject(with: data)) != nil
            else { return nil }
            return (name, s)
        }
        if let dict = arguments as? [String: Any],
            let data = try? JSONSerialization.data(withJSONObject: dict),
            let s = String(data: data, encoding: .utf8)
        {
            return (name, s)
        }
        if let dict = arguments as? [String: String],
            let data = try? JSONSerialization.data(withJSONObject: dict),
            let s = String(data: data, encoding: .utf8)
        {
            return (name, s)
        }
        return nil
    }

    private static func parseJSONObject(_ s: String) -> [String: Any]? {
        parseJSON(s) as? [String: Any]
    }

    private static func parseJSON(_ s: String) -> Any? {
        guard let data = s.data(using: .utf8) else { return nil }
        return try? JSONSerialization.jsonObject(with: data)
    }
}
