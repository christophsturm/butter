// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// ButterEnv — cheap, live environment-variable reads for hot paths.
//
// `ProcessInfo.processInfo.environment` materialises the ENTIRE
// environment dictionary (one bridged String per var) on every access.
// Measured on the Laguna decode loop that is ~1.3% of decode throughput
// PER ENV VAR CARRIED BY THE PROCESS (see the Laguna Parity compendium
// §6.9): an A/B arm that merely exports more vars than its control is
// taxed for free, which fabricated a clean −4.3% "regression" that three
// dummy vars reproduced exactly.
//
// `getenv(3)` reads one entry from `environ` directly: no dictionary,
// no bridging, cost independent of how many vars the process carries.
// Semantics stay LIVE — `setenv`/`unsetenv` from tests or the CLI are
// observed on the next read, exactly like the `Debug.envIsSet` helper
// this generalises.
//
// Usage: `ButterEnv["BUTTER_FLAG"]` wherever hot code previously
// subscripted the full `ProcessInfo` environment dictionary. Flags that are
// fixed for the lifetime of the process (launch-time A/B switches on
// per-token/per-layer paths) should additionally be hoisted to a
// `static let` at their use site so the read happens exactly once —
// see e.g. `LagunaNvfp4Fuse` / `MoELayer.NvDecodeFlags`.
//
// TODO: audit remaining `ProcessInfo.processInfo.environment` reads in
// ButterCLI (cold paths, left as-is deliberately).

import Foundation

public enum ButterEnv {
    /// Live single-var read via `getenv(3)`. Returns `nil` when unset.
    @inline(__always)
    public static subscript(_ name: String) -> String? {
        guard let c = getenv(name) else { return nil }
        return String(cString: c)
    }
}
