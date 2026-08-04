# Qwen3x (Qwen 3.5 / 3.6) — ledger

## w42 — dense-checkpoint batched prefill produced token-0 garbage (fixed)

**Symptom.** Every *dense-FFN* Qwen3x hybrid produced degenerate greedy
output in `butter generate`: the first sampled token was always token 0
("!"), then loops ("!systemd" on Qwen3.6-27B). Qwen3.6-35B-A3B (MoE FFN)
was unaffected. Reported against `dev` @ `a57bbbbd` and a Jul-31 binary
— pre-existing, not a recent regression of the Laguna campaign.

**Reproduction matrix** (greedy, `-p "The capital of France is"
--max-tokens 20 --temperature 0`, dev @ a57bbbbd):

| checkpoint | FFN | result before fix | after fix |
|---|---|---|---|
| Qwen3.6-27B 4bit (local) | dense | `!systemd` loops | "Paris." + coherent |
| Qwen3.5-4B 4bit (local) | dense | `!user # 1. 1.1...` | "Paris." + coherent |
| mlx-community/Qwen3.5-0.8B-4bit | dense | `!根号内是负数，则…` (token-0 first, verified at base — matches the reported integration-test signature) | coherent start; batched output byte-identical to `BUTTER_LEGACY_FORWARDMANY=1` |
| Qwen3.6-35B-A3B 4bit | MoE | known good | unchanged |
| any dense + `BUTTER_LEGACY_FORWARDMANY=1` | dense | coherent | coherent |

**Bisect trail.**
1. Env-flag matrix on the 4B: `BUTTER_LEGACY_FORWARDMANY=1` fixed it;
   `BUTTER_GDN_NO_FUSED_PREP=1` fixed it (falls back to the per-token
   blit path for GDN layers); every kernel-level toggle
   (`BUTTER_SDPA_LEGACY_D256`, `BUTTER_GDN_NO_PREP_CHUNK[_FAST]`,
   `BUTTER_NO_CONV1D_MANY`, `BUTTER_QMM_COOP/MPP`, `BUTTER_NO_FUSED_QKV`,
   `BUTTER_GDN_WY`) changed nothing → model-driver-level bug, not a
   kernel bug.
2. Replacing all five GDN batched projections with per-row gemv loops
   changed nothing → projections exonerated.
3. Stage dumps (blit copy + completed-handler) showed **every** tensor
   on the prefill `workCmd` — including the embedding gather output —
   reading back zero, while an identical gather re-dispatched on a
   fresh, synchronously-committed cmd produced correct values.
4. Completed handlers on the prefill `workCmd` fired with
   `status == .notEnqueued`: the command buffer was **never committed**
   — all its encoded work (embedding + first GDN layer) was dropped.

**Root cause.** `Qwen35GDNLayer.decodeMany` declares
`commitsCommandBuffer == true`, and `_forwardManyBatched` therefore
unconditionally replaces `workCmd` after every fused-GDN layer. That
contract only held for MoE FFN layers (`Qwen35MoEFFN`/`MoELayer`
`forwardMany` commit internally). The dense branch of
`qwen35ApplyFFNMany` leaves the whole layer queued on the caller's cmd,
so on dense checkpoints every GDN layer's `workCmd` — embedding gather
included on layer 0 — was discarded uncommitted. The residual stream
stayed zero through the entire stack; `lm_head(rms_norm(0)) = 0` gives
a uniform logits row and `argmax = 0` → token "!" every step. The
single-token decode path (`qwen35ApplyFFN(commitCmd: true)`) always
committed, which is why decode and `BUTTER_LEGACY_FORWARDMANY=1` were
correct, and why the MoE 35B-A3B never showed it.

**Fix.** `Qwen35GDNLayer.decodeMany` now commits `cmd` (no wait — same
pipelining contract as single-token) when its FFN is dense, honouring
its declared `commitsCommandBuffer` contract in the batched path
(`Sources/ButterSwift/Models/Text/Qwen3xText.swift`).

**Regression tests.**
- `Tests/ButterTests/Models/Qwen35BatchedPrefillTests.swift` — tiny
  synthetic all-GDN dense hybrid (bf16, 2 layers); asserts batched
  `forwardMany` logits are non-zero (the bug's signature) and match the
  per-token `forward(tokenId:)` reference (max-abs + argmax).
- `Tests/ButterTests/Ops/DequantGatherBatchedTests.swift` — bf16
  `Ops.dequantGather` vs CPU reference at the production embedding shape
  (hidden 2560, gs 64) for nTokens 1/5/32 (coverage gap found while
  bisecting; the op itself was never at fault).
- `Tests/ModelIntegrationTests/Text/Qwen35TextIntegrationTests.swift`
  (0.8B dense) exercises the fixed path end-to-end (see open item
  below for its current status).

**Validation (post-fix, M5 Max).** Qwen3.6-27B coherent greedy output
("capital of France" / haiku / hash-map prompts); Qwen3.5-4B
unchanged-coherent; 0.8B batched output byte-identical to the legacy
per-token path over 60 greedy tokens; new unit suites green;
`make test-unit` green.

**Open items (out of w42 scope, observed while validating).**
1. `Qwen35TextIntegrationTests` (0.8B) now clears the token-0 failure
   but still trips `expectCoherentOutput`'s diversity floor: greedy
   200-token decode falls into a "first press was founded in 1476"
   repetition loop (15/200 unique < 0.2). The batched and legacy
   per-token paths produce byte-identical text, so this is model-level
   greedy repetition on the 4-bit 0.8B checkpoint (or a separate
   quality regression), NOT the dropped-cmd bug. Needs its own
   investigation (compare vs mlx-lm reference / an older butter build).
2. Greedy near-tie instability on Qwen3.6-27B: the haiku prompt's first
   generated token varied across identical solo runs ("Thinking" /
   "-" / "<think>"), and one run under heavy concurrent GPU load from
   an unrelated test suite produced a token-0 "!systemd" row. Deterministic
   solo repro of the fixed build stayed coherent. Suspect run-to-run
   float jitter at a near-tie plus a load-coupled robustness gap
   (committed command buffers are never checked for `.error`); worth a
   separate hardening pass.
