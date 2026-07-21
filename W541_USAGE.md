# Running the 1.5B on the ThinkPad W541 (CPU) — do this, or it'll feel dumb

The model itself is fine at low temperature *with the chat template applied*. The
usual reason a small local model "sucks" is the launch command, not the weights:

- `llama-cli -m model.gguf -p "make a calculator"` runs in **raw completion mode**
  with **no chat template** and **temperature ~0.8**. An instruct model fed a raw,
  unformatted prompt at high temp will ramble and produce garbage. That is almost
  certainly what happened.

## The right way

**Option A — chat server (recommended).** `llama-server` applies the model's chat
template automatically:

```bash
llama-server -m qwen-coder-1.5b-py-Q4_K_M.gguf \
  -c 4096 --temp 0.2 --top-p 0.9 --repeat-penalty 1.1 --host 127.0.0.1 --port 8080
```
Then hit `http://127.0.0.1:8080` (built-in web UI) or the OpenAI-compatible
`/v1/chat/completions` endpoint. Point any client (Open-WebUI, opencode, etc.) here.

**Option B — interactive CLI in conversation mode.** The `-cnv` flag is what turns on
the chat template + turn-taking:

```bash
llama-cli -m qwen-coder-1.5b-py-Q4_K_M.gguf -cnv \
  --temp 0.2 --top-p 0.9 --repeat-penalty 1.1 \
  -p "You are a helpful Python coding assistant."
```

## Settings that matter for a 1.5B

| flag | value | why |
|---|---|---|
| chat template | **on** (`-cnv` or `llama-server`) | without it, instruct models break |
| `--temp` | **0.2–0.3** | low = focused, correct code; high = rambling |
| `--top-p` | 0.9 | mild nucleus sampling |
| `--repeat-penalty` | 1.1 | stops small models looping |
| `-c` | 4096 | context window |

## Speed
~10 tok/s on the W541 (Haswell CPU, no GPU) is expected and fine for Python help.
The Quadro K2100M is too old/small to help — CPU is the right call.

## Ask for complete things
Small models do better with a clear ask: *"Write a complete Python program that ..."*
rather than a two-word prompt.
