# Tool-round tokenizer fixture

`chat_template.jinja` is the Qwen3.8 family template verbatim, apart from its
trailing newline. `tokenizer.json` is the dependency-closed BPE slice needed by
the two exact tool-round prompts in `HybridPrefixTests`. Original merge order is
preserved and token ids are remapped below the tiny test model's 512-token
vocabulary.
