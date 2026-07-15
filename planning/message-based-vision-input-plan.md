# Message-Based Vision Input Plan

| Field | Value |
|---|---|
| Status | Core validated (P1+P3) — per-family sweep (P2/P4) remaining |
| Goal | `Model.generate(.vision(messages:media:))` uniform with `.text`; no caller-built `promptTokens`/placeholders |
| Decisions | Content-parts message shape (HF/OpenAI-style); all ~15 VL families in one sweep; deprecate token-based `VisionRequest.promptTokens` |
| Branch | `ek/fix-all-the-things` (current) |

## Mechanism (template-driven primary + FFAI-controlled fallback)

Verified against cached VL models (2026-07-15): Qwen2.5/3/3.5/3.6-VL, Gemma4, SmolVLM2, FastVLM chat templates all iterate content-parts (`content is string` / `is sequence` / `content.type == 'image'`) and emit the family's vision markers with **ONE placeholder token per image** (e.g. Qwen `<|vision_start|><|image_pad|><|vision_end|>`). This is the HF processor model: template inserts one marker, the processor expands it to the grid count.

**Primary (template-driven):**
1. `renderChatTemplate(messages w/ content-parts)` → tokens with 1 placeholder/image (+ family markers, straight from the shipped template).
2. Encode each media once → vision rows + count N (`encodeImage().shape[0]` etc.).
3. Expand each placeholder token (`imageTokenId`/`videoTokenId`/`audioTokenId`, already on `VisionModel`) in order → N copies.
4. `VisionModel.splice` + `generate` with pre-encoded rows. Encode ONCE.
- Needs ~no new per-family config: reuses existing `imageTokenId` + count strategy.

**Fallback (FFAI-controlled framing):** if template render throws (swift-jinja can't handle the content-part constructs) OR emits no placeholder, splice the family's known markers (recap table) around `placeholderId × N` at the media slot (divergence-located). Per-family marker config lives here, exercised only for templates that don't support content-parts.

## Progress
- ✅ **P1a** (`f5bbcb4`): `ContentPart` + `ChatMessage.content: [ContentPart]`, back-compat String init, `asTemplateMessage` text/multimodal split.
- ✅ **P1b + P3** (`b96c476`): message-based `VisionRequest` + `VisionMedia`; `VisionPromptBuilder` (template render → placeholder expand); `.vision` dispatch. Token path = deprecated escape hatch.
- ✅ **Validated** (`46797a6`): text-fill in `.vision` dispatch; Qwen3-VL unified image test passes end-to-end against cached model (coherent dog caption, no hand-built tokens). Unit gate 1737/221 green throughout.
- ⏳ **P4** (remaining): route the other ~13 VL + omni integration suites through `Model.generate(.vision(messages:media:))`; validate each family's template framing (GPU, ~5 min/family). Add video + audio unified tests.
- ⏳ **P2** (remaining): FFAI-controlled fallback framing for any family whose template doesn't handle content-parts (currently throws `templateFramingUnavailable`). Determine which families need it during the P4 sweep.
- ⏳ **Follow-ups**: multi-image / interleaved (dispatch caps at 1 media/request today); encode-once (builder + generateStream currently encode twice); CLI migration (drop `MultimodalPrompt`).

## Phases

### P1 — Core types (content-parts + reshaped vision input)
- `ChatTemplate.swift`: add `enum ContentPart { case text(String), image, video, audio }`. `ChatMessage.content: String` → `[ContentPart]`. Keep `init(role:content:String)` (wraps `[.text]`) — the 27 existing sites compile unchanged (0 read `.content` back). Add `init(role:content:[ContentPart])`, `var textContent: String`, `var mediaParts: [ContentPart]`.
- `asTemplateMessage`: text-only turn → `"content": String` (text path byte-identical); multimodal turn → `"content": [[String:Sendable]]` HF list (`{"type":"text","text":…}` / `{"type":"image"}`).
- `ModelIO.swift`: reshape `VisionRequest` to `{ messages:[ChatMessage], media:[VisionMedia], maxTokens, parameters, eos… }` where `VisionMedia = .image(RGBImage) | .video([RGBImage]) | .audio([Float])`. Media parts in messages bind positionally to `media`. Keep a deprecated token-based escape hatch (`VisionRequest.tokens(...)` factory) for power users / determinism tests.
- Gate: `make test-unit` green (text path unaffected).

### P2 — Per-family framing config on VisionModel
- Add to `VisionModel` a `VisionFraming` value: pre/post marker token seqs (or strings) around the placeholder run, per-frame video wrap, count strategy (`dynamic` = `encodeImage().shape[0]` | `fixed(imageTokenCount)` | `formula`).
- Populate for all families at their construction sites (recap table): Qwen2/2.5/3/3.5/3.6-VL (`<|vision_start|>`/`<|image_pad|>`/`<|vision_end|>`), Gemma3/4 (boi/`image_soft_token`/eoi), MiniCPM-V 4.5/4.6 (`<image>`/`<|image_pad|>`/`</image>`), Mistral3/Pixtral (`[IMG]`), Idefics3 (formula), SmolVLM2 (`<image>` formula), LFM2-VL, GlmOcr, FastVLM (id −200), Paligemma, NemotronH-VL.

### P3 — Core prompt-builder + dispatch
- New `VisionModel.buildPrompt(messages:media:tokenizer:) -> (promptTokens:[Int], encoded:[EncodedMedia])`: render template w/ slots → encode media → splice per-family framing + `placeholderId×N`. Reuse encoded rows in `generate` (no re-encode).
- `ModelInference.swift` `.vision` path → call builder → `generate`. Keep token-based path via the escape hatch.

### P4 — Migrate tests + CLI
- All `Tests/ModelIntegrationTests/Vision/**` + Qwen/MiniCPM VLM suites → message-based `Model.generate(.vision(messages:media:))`. Keep 1 token-based test (escape hatch coverage).
- `FFAICLI/GenerateMediaCommands.swift` → message-based; delete `MultimodalPrompt` divergence hack.
- Gate: `make test-unit`; vision integration suites (user-run) validate framing per family.

## Resolved scope (2026-07-15)
- **Full multimodal in one sweep**: image (single + multi + interleaved) + video (per-frame framing) + audio slots + omni. All ~15 VL families + all vision/omni tests + CLI.
- Content-parts message shape; token-based `VisionRequest.tokens(...)` retained as a deprecated escape hatch (1 test keeps coverage).

## Open (decide during impl)
- Per-frame video wrap vs single run for families that differ (Gemma4 per-frame vs Qwen merged) — follow each family's existing test framing.
- Combined image+video+audio in ONE turn (omni) — `VisionModel` has 4 single-modality generate overloads today; may need a combined path or sequential encode+splice.
