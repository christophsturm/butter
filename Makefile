# Butter — Makefile
#
# Common dev-loop targets. See planning/plan.md for the phased
# build-out and scripts/ for the longer-form scripts.

.DEFAULT_GOAL := help

# ─── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT := $(shell pwd)
# Sibling kernels DSL (renamed to Iron as part of the org rebrand).
IRON_DIR := $(PROJECT_ROOT)/../iron
KERNEL_OUT := $(PROJECT_ROOT)/Sources/IronSwift

# ─── Help ─────────────────────────────────────────────────────────────
.PHONY: help
help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ─── Setup ────────────────────────────────────────────────────────────
.PHONY: setup-dev
setup-dev: ## one-time dev environment setup (toolchains, deps, first build)
	./scripts/setup-dev.sh

# Alias kept for muscle-memory parity with the older `make setup` name.
.PHONY: setup
setup: setup-dev

# ─── Git hooks ────────────────────────────────────────────────────────
# `core.hooksPath = scripts/hooks` is per-clone — every contributor runs
# `make install-hooks` once after cloning. The hooks themselves are
# version-controlled under scripts/hooks/ so updates propagate via pull.
.PHONY: install-hooks
install-hooks: ## set core.hooksPath -> scripts/hooks (pre-commit / commit-msg / pre-push)
	./scripts/install-hooks.sh

.PHONY: uninstall-hooks
uninstall-hooks: ## clear core.hooksPath (disables the in-tree hooks)
	git config --unset core.hooksPath && echo "✓ Uninstalled hooks"

# ─── Build ────────────────────────────────────────────────────────────
.PHONY: build
build: regenerate-kernels ## swift build (debug)
	swift build

.PHONY: build-release
build-release: regenerate-kernels ## swift build (release)
	swift build -c release

.PHONY: regenerate-kernels
regenerate-kernels: ## run `iron build --emit` to regenerate metallib + Swift wrappers
	@if [ ! -d "$(IRON_DIR)" ]; then \
	  echo "Error: iron not found at $(IRON_DIR)"; \
	  echo "Clone the sibling repo at ../iron."; \
	  exit 1; \
	fi
	@# Run from iron so its rust-toolchain.toml is honored.
	@# `iron build` writes:
	@#   $(KERNEL_OUT)/Resources/kernels/<name>.metal
	@#   $(KERNEL_OUT)/Resources/kernels.metallib
	@#   $(KERNEL_OUT)/Resources/manifest.json
	@#   $(KERNEL_OUT)/Generated/IronKernels.swift
	@# Build `__iron_runner` too — iron spawns it by bare name via PATH.
	@if [ -x "$(IRON_DIR)/target/release/iron" ] && \
	   [ -x "$(IRON_DIR)/target/release/__iron_runner" ]; then \
	  (cd $(IRON_DIR) && PATH="$$PWD/target/release:$$PATH" \
	    iron build --emit all --out $(KERNEL_OUT)); \
	else \
	  (cd $(IRON_DIR) && \
	    cargo build --release --bin iron --bin __iron_runner && \
	    PATH="$$PWD/target/release:$$PATH" \
	      iron build --emit all --out $(KERNEL_OUT)); \
	fi
	@# Prepend `// swift-format-ignore-file` so swift-format skips wrappers.
	@gen=$(KERNEL_OUT)/Generated/IronKernels.swift; \
	  if [ -f "$$gen" ] && ! head -1 "$$gen" | grep -q swift-format-ignore-file; then \
	    tmp="$$gen.tmp"; \
	    { printf '// swift-format-ignore-file\n//\n'; cat "$$gen"; } > "$$tmp" && mv "$$tmp" "$$gen"; \
	  fi

# ─── Bench ────────────────────────────────────────────────────────────
# Quality benches (wikitext2 perplexity / KLD, NIAH retrieval) forced-decode
# an entire corpus. In debug that's far too slow for the numbers to mean
# anything (design §7), and the CLI hard-refuses a debug quality bench. Always
# go through release. Pass-through args via ARGS=:
#
#   make bench ARGS="--method wikitext2 --model <repo> --wikitext2-corpus wiki.test.raw"
#   make bench ARGS="--method niah --model <long-ctx-repo>"
.PHONY: bench
bench: regenerate-kernels ## run `butter bench` in release (quality benches require it)
	swift run -c release butter bench $(ARGS)

# ─── Test ─────────────────────────────────────────────────────────────
#
# Production-parity defaults. The 2026-05-19 GPU-pin root cause —
# wrong-dispatch-shape Ops wrappers — is fixed at the source (see the
# post-mortem in papers/ and OpsValidation in Sources/ButterSwift/). Test
# runs now use the same BUTTER_MAX_COMMAND_BUFFERS=16 cap as production,
# so anything that passes in CI is proven safe under production load.
#
# Defense in depth still in place:
#
#  1. **OpsValidation** preconditions on every reduction-mode wrapper.
#     Catches degenerate dispatch shapes (wrong head_dim, wrong n,
#     etc.) before the kernel ever launches.
#
#  2. **Thread-safe shared state.** PSOCache uses single-flight
#     compilation (compileLock) so parallel suites can't both compile
#     the same PSO. BufferPool uses NSLock. See PSOCache.swift +
#     BufferPool.swift.
#
#  3. **ModelLoadLock** (Tests/ModelIntegrationTests/ModelLoadLock.swift) — global
#     async mutex around `Model.load(...)`. Different concern from GPU
#     access: model load is heavy on RAM + disk-IO + GPU memory
#     allocation BEFORE any cmdbuf exists. The lock makes Model.load()
#     a global critical section so only one multi-GB checkpoint is
#     loading at a time.
#
# Targets:
# - `make test-unit`         — ButterTests + IronSwiftTests at the
#                              production cap (BUTTER_MAX_COMMAND_BUFFERS=16).
# - `make test-integration`  — ModelIntegrationTests at production cap + ModelLoadLock
#                              + `--parallel --num-workers 1` (memory
#                              pressure, not GPU). Matches release.yml.
# - `make test`              — both in sequence.
# - `make test-stress`       — canary; both suites at production cap with
#                              integration parallelism uncapped. Run after
#                              touching anything dispatch-related to
#                              confirm production safety holds under
#                              maximal parallel load.

.PHONY: test
test: regenerate-kernels test-unit test-integration ## run unit then integration test suites

.PHONY: test-unit
test-unit: regenerate-kernels ## unit + Metal tests at production cap (BUTTER_MAX_COMMAND_BUFFERS=16), serialized
	@# --no-parallel: this is the SERIALIZED correctness gate. Several Metal
	@# tests do exact-equality comparisons against a reference computed on the
	@# same GPU, and the Ops layer shares Device.shared scratch slabs — running
	@# suites concurrently lets one test's dispatch stomp another's scratch mid
	@# read (e.g. AURAQuantizedKVCache rollback saw maxErr 2e38 under parallel
	@# execution, 0 serialized). `test-stress` is the uncapped-parallelism canary.
	BUTTER_MAX_COMMAND_BUFFERS=16 swift test --no-parallel --filter "ButterTests|IronSwiftTests"

.PHONY: test-integration
test-integration: regenerate-kernels ## end-to-end model tests; production cap + ModelLoadLock; matches release.yml
	@# ModelLoadLock (Tests/ModelIntegrationTests/ModelLoadLock.swift) serializes
	@# Model.load() across suites so multi-GB checkpoints load one at a
	@# time. --num-workers 1 caps Swift Testing's cross-suite parallelism
	@# to one model resident at a time (memory pressure, not GPU).
	BUTTER_MAX_COMMAND_BUFFERS=16 swift test --parallel --num-workers 1 --filter "ModelIntegrationTests"

.PHONY: test-stress
test-stress: regenerate-kernels ## canary; production cap with uncapped parallelism — run after touching dispatch code
	@echo "Stress mode. Running unit + integration at BUTTER_MAX_COMMAND_BUFFERS=16"
	@echo "with no --num-workers cap on integration. If anything regresses our"
	@echo "wrapper-precondition / PSOCache / ModelLoadLock defenses, this is"
	@echo "where it surfaces."
	@echo ""
	BUTTER_MAX_COMMAND_BUFFERS=16 swift test --filter "ButterTests|IronSwiftTests"
	BUTTER_MAX_COMMAND_BUFFERS=16 swift test --filter "ModelIntegrationTests"

.PHONY: coverage
coverage: ## swift test with coverage report (unit suite only, matches ci.yml)
	BUTTER_MAX_COMMAND_BUFFERS=16 ./scripts/coverage.sh

.PHONY: integration-bisect
integration-bisect: regenerate-kernels ## run each ModelIntegrationTests/*IntegrationTests suite alone; tag GPU-pinned exits
	@# Runs every integration suite in its OWN swift-test process, captures
	@# pass/fail/timeout + GPU active-residency 3 s after exit. Any suite
	@# that leaves the GPU at ≥ 50% after the test ends is flagged "PINNED"
	@# and earmarked for xctrace profiling. GPU sampling requires sudo
	@# (powermetrics) — without it, the table still shows pass/fail but
	@# the GPU column degrades to "?".
	@#
	@# Pass suite names to run only a subset:
	@#   make integration-bisect SUITES="Whisper Llama"
	@#
	@# Or use the script directly for per-suite timeouts:
	@#   PER_TEST_TIMEOUT=600 ./scripts/integration-bisect.sh Whisper
	./scripts/integration-bisect.sh $(SUITES)

# ─── Lint / format ────────────────────────────────────────────────────
# Invoke via `xcrun swift-format` so the call works both when
# swift-format is on $PATH (e.g. brew install swift-format) AND when
# it's only shipped inside the Xcode toolchain (xcrun resolves it via
# DEVELOPER_DIR). `xcrun` falls back to a PATH lookup when neither
# path resolves, so the behavior is identical for Homebrew installs.
#
# Files that should be skipped opt out at the source-file level via
# `// swift-format-ignore-file` on line 1. See
# Sources/IronSwift/Generated/IronKernels.swift for the
# canonical example (the file is auto-regenerated by Iron's
# `iron build --emit swift`; the directive needs to be re-emitted at
# the top by the codegen — see planning/session-plan.md).
.PHONY: format
format: ## run swift-format on all .swift files
	xcrun swift-format format --in-place --configuration .swift-format --recursive .

.PHONY: format-check
format-check: ## swift-format lint (no writes)
	xcrun swift-format lint --configuration .swift-format --recursive . && echo "format OK"

# ─── Docs ─────────────────────────────────────────────────────────────
# User-facing documentation lives at the butter-website repo (../butter-website)
# (source: github.com/thewafflehaus/butter-website).
#
# The website fetches markdown from this repo at build time, so committing
# changes to documentation/, README.md, planning/architecture.md, or
# planning/roadmap.md on `main` triggers a rebuild via a GitHub Action.

WEBSITE_DIR := $(PROJECT_ROOT)/../butter-website

.PHONY: docs
docs: ## verify markdown + preview the docs site locally (if ../butter-website is checked out)
	./scripts/verify-docs.sh
	@if [ -d "$(WEBSITE_DIR)" ]; then \
	  echo ""; \
	  echo "Preview the docs site (Ctrl+C to stop):"; \
	  echo "  cd $(WEBSITE_DIR) && pnpm dev"; \
	  echo ""; \
	  echo "Or to build a one-shot static preview:"; \
	  echo "  cd $(WEBSITE_DIR) && pnpm build && pnpm dlx serve dist"; \
	else \
	  echo ""; \
	  echo "Tip: clone the docs site to preview locally:"; \
	  echo "  git clone https://github.com/thewafflehaus/butter-website $(WEBSITE_DIR)"; \
	fi

.PHONY: docs-verify
docs-verify: ## swift-docc target-by-target verification only (no website preview)
	./scripts/verify-docs.sh

# ─── Clean ────────────────────────────────────────────────────────────
.PHONY: clean
clean: ## remove .build and generated kernel artifacts
	rm -rf .build
	rm -f Sources/IronSwift/Resources/kernels.metallib
	rm -f Sources/IronSwift/Resources/manifest.json
	rm -rf Sources/IronSwift/Resources/kernels
	rm -rf Sources/IronSwift/Generated/IronKernels.swift
