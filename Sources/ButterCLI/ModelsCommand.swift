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
// `butter models` — list every supported model family with example
// HuggingFace repo IDs, so a user can copy-paste an ID straight into
// `butter generate --model <id>` / `butter bench --model <id>`.
//
// The catalog is rendered DIRECTLY from `ModelRegistry.catalog` (the
// single source of truth in `Sources/Butter/Loader/ModelRegistryTable.swift`)
// — the same table the loader dispatches on — so the list can never drift
// from what Butter actually supports. Any mlx-format 2/3/4/5/6/8-bit
// conversion of a supported architecture also loads; the IDs are the
// canonical published examples.

import ArgumentParser
import ButterSwift
import Foundation

struct ModelsCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "models",
        abstract: "List the model families Butter supports, with example repo IDs."
    )

    func run() throws {
        print("Butter — supported model families\n")
        for (kind, entries) in ModelRegistry.catalog {
            let title = kind.title
            print("── \(title) " + String(repeating: "─", count: max(0, 50 - title.count)))
            for e in entries {
                let types = e.modelTypes.sorted().joined(separator: " / ")
                print("  \(e.family)" + (types.isEmpty ? "" : "  [\(types)]"))
                if !e.summary.isEmpty { print("    \(e.summary)") }
                for repo in e.repos { print("    • \(repo)") }
                print("")
            }
        }
        print(
            """
            Pass any example repo ID to a command:

              butter generate --model mlx-community/Qwen3.5-0.8B-MLX-4bit --prompt "Hello"
              butter bench    --model mlx-community/Qwen3-1.7B-8bit --prompt "Hello" --stats

            Any mlx-format 2/3/4/5/6/8-bit conversion of a supported
            architecture also loads — the IDs above are the published
            examples we verified. Repos are resolved + cached through the
            HuggingFace Hub.
            """)
    }
}
