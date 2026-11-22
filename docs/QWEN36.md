# Qwen3.6-35B-A3B: pictures, several requests at once and speed

This page is for people who run [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (35B mixture of
experts, 3B active, 256 experts, top-8) with Strata and want to know how pictures, parallel requests and the prompt
chunk behave. For sizes and memory, see [MODELS.md](MODELS.md).

Strata runs [Unsloth's GGUFs with the MTP layer](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)
(UD-Q4_K_XL, UD-Q6_K_XL, Q8_0) on its own engine path (`engine/src/qwen35`; `strata` switches to it from the file's
architecture). It runs on NVIDIA and AMD cards. The Intel Arc and Mac engines do not run it.

## Quick start

Pick it in the app's model list, in `strata launch` or in setup:

```sh
strata launch --family qwen36 --model ud-q4_k_xl
strata launch --family qwen36 --model ud-q4_k_xl --vision on --parallel 2
```

| Option | Effect |
| --- | --- |
| `--vision on\|off\|gpu\|cpu` | Terminal: pictures and which reader. Source setup uses `--vision yes\|no\|gpu\|cpu` and `--vision-tokens N`. |
| `--parallel N` | How many requests are answered at once, 1 to 8. Source setup has the same option. |
| `--prefill 512` | Engine flag in the config's `args`, not a `strata launch` option: the prompt chunk size ([Prompt chunk](#prompt-chunk)). |

## Pictures

The vision encoder is Unsloth's `mmproj-F16.gguf` (0.9 GB, pinned revision, SHA-256 checked by setup): a 27-layer ViT
with a 2x2 merger, the same projector family as the Flash-Next models' (`qwen3vl_merger`; only its output width
differs: 2048 here, 2560 there). Strata runs it with the same helper (`strata-vision`), so the
encoder choices are the Flash-Next ones ([Images](DETAILS.md#images-vision)).

Pictures are on by default, by the same rule as for every model on NVIDIA and AMD cards:

- The CPU reader runs under 24 GB of graphics memory. It takes a few seconds per picture and no VRAM.
- The card's reader runs from 24 GB (GPU encoder, about 1.2 GB of VRAM at up to 1,024 image tokens). On AMD it is
  `strata-vision-hip`, with the CPU reader as its fallback.

Your explicit choice is kept, with a note. A picture becomes up to 1,024 tokens of the context (the CPU default is 300).
The engine puts the encoder's rows where the prompt has `<|image_pad|>` tokens and gives every row its 2-D position,
interleaved M-RoPE as llama.cpp does for this model (rotary sections 11 / 11 / 10).

Pictures work with several requests at once, with the MTP drafts and with the conversation cache. The same conversation
sent again continues from the held state when its pictures are the same (matched by the server's key for each
picture). A different picture starts that conversation again.

> **Measured on:** RTX 3060 12 GB (CUDA) and an R9700 box (32 threads) and a 3060 box (10 threads), UD-Q4_K_XL, Strata
> 0.1.45.

| Encoder | Time per picture (640x400 = 260 tokens / 800x450 = 350 tokens) | VRAM |
| --- | --- | --- |
| GPU, RTX 3060 | 0.10 s / 0.15 s | 1,179 MiB at up to 1,024 tokens (1,337 MiB at 2,048), CUDA context included |
| CPU, 32 threads (R9700 box) | 0.9 s / 1.1 s | none |
| CPU, 10 threads (3060 box) | 1.6 s / 2.1 s | none |

## Several requests at once

`"parallel": N` in the model's config (`strata-qwen36-<size>.json`) or `--parallel N` makes the engine keep N
conversations open (`--batch N`, up to 8) and decode them together. Every window carries one token of each
conversation, so the dense weights, the head and every routed expert that conversations share are read once for all of
them. The server speaks the same engine protocol as for Flash-Next ([BATCHING.md](BATCHING.md)).

A new request waits only for the read of its own prompt, which runs between the decode windows of the others. A
conversation continues in the slot that holds its start, or the engine copies the longer held start from another slot.
A request that runs alone (the solo path, with its MTP drafts) moves into a batch slot when another arrives, and back
when it is alone again, and gives the same text either way.
`STRATA_Q36_READ_WINDOWS=4` lets the decoding slots produce more tokens during a long prompt read, at the cost of a
slower read. The default is one decode window per chunk.

**MTP drafts.** A request that has the engine to itself keeps its MTP drafts. With several requests the batch window has
no drafts unless `--batch-mtp` is in the config's `args` (opt-in). Each slot's window then also carries the proposal its
MTP block made, one per slot, up to the rows to spare in the 8-row window.

**What a slot costs.** Each slot keeps its own cache in VRAM: 1.5 GB at a context of 32K (fp32 keys and values of the 10
attention layers and the MTP block, 45 KB per token) and 0.07 GB of recurrent state. Setup writes the context for one
conversation, so N slots take N times that from the room the experts would have. On a card that does not hold every
expert (12 and 16 GB, the hybrid mode) the expert slots lose that room, and a batch reads more distinct experts from RAM.

Setup offers `--parallel` where the sessions do not eat the experts' room, says the price in the same breath and asks
"How many requests at once?" Enter keeps one at a time, and any number you give is kept.

| Card type | Slots setup offers |
| --- | --- |
| Every expert still fits | Up to 3 to 4 (3 on a 32 GB card at 32K) |
| Hybrid mode | The count whose sessions take at most 35% of the room the experts would have. 12 GB: 4 slots at 8K, none at 32K. 16 GB: 2 slots at 32K. |

The engine says what the sessions take before it allocates them and refuses, with the numbers, a count that cannot fit.

In the hybrid mode the text of a request can differ a little between running alone and running beside others. The
experts that sit on the GPU or on the CPU change with the routing of all the requests, and the two compute with
different rounding. With the slot tables fixed (`STRATA_Q36_FIXED_SLOTS=1`, a test setting) every slot equals its solo
run there too.

## Prompt chunk

The MoE prompt kernels read every expert once per chunk, so a bigger chunk feeds them more rows per expert. With every
expert in VRAM the engine takes the largest of 4096 / 2048 / 1024 that fits beside the experts, the 0.77 GiB reserve
and the sessions' caches.

| Situation | Chunk |
| --- | --- |
| Every expert in VRAM, 8K to 64K context | 4096 |
| Every expert in VRAM, 100K context | 2048 |
| Every expert in VRAM, 131K context | 512 |
| `--batch 2..4` | 1024 at most, so the other slots are not held for a whole chunk |
| `--batch 8` | 512 |
| Hybrid mode | 2048; 1024 or 512 on a card with 7.5 or 5.5 GiB free |

In the hybrid mode a bigger chunk takes expert slots (an RTX 3060 at chunk 4096 read prompts 5% faster and decoded 13%
slower). `--prefill 512` (or `STRATA_Q36_CHUNK=512`) gives the 512 size, and then the text is the base engine's, bit
for bit. The chunk size changes the GEMM tiles the BLAS picks and with them the last bits of the prompt state. This is
the size of the llama.cpp CPU-against-HIP difference on the same file, not a quality loss. A prompt shorter than the
chunk (every chat turn after the first) is one chunk either way.

## Speed and exactness

Every change on the decode path gives the old kernels' bits. `STRATA_Q36_VERIFY=1 strata ... --no-graph` runs the old
kernel beside each new one on the real activations and compares them bit for bit. `STRATA_Q36_OLD_KERNELS=1` switches
the new kernels off. Gate/up for Q4_K experts uses the V2K kernels (`STRATA_EXPERT_V2K`, the default on gfx12).

> **Measured on:** AMD Radeon AI PRO R9700 32 GB (HIP gfx1201), UD-Q4_K_XL, Strata 0.1.45 against the engine before it,
> 5 interleaved pairs, medians; llama.cpp 972b5fc HIP on the same file (`-fa 1`).

| | Before | 0.1.45 | llama.cpp |
| --- | ---: | ---: | ---: |
| Decode tg128 | 83.2 tok/s | 117.8 | 81.9 to 82.1 |
| Prompt pp4096 | 3,453 | 6,220 (chunk 4096) | 3,794 (`-ub 2048`) / 3,042 (`-ub 512`) |

More numbers are in [History](#history--older-measurements).

## Limits

- Eight slots at most (the window holds 8 rows). `--batch-mtp` shares those rows between slots and proposals.
- A sampling request in a batch decodes one token per window without proposals.
- `logit_bias` is not available on this engine path. The server refuses a request that has it (as it does with
  `--batch` for Flash-Next).
- A slot that stopped being alone goes back to the solo path only through the server's own hand-over. The MTP block's
  cache has holes for the tokens a slot decoded in batch windows, so the drafts of a later solo stretch are a little
  worse. The text is not affected.
- Intel Arc (SYCL) and Mac do not run this family.

## Testing

The gates are in `tools/qwen35/` and talk to the engine through its serve protocol (no server needed). Run them under
the box's GPU lock.

| Script | Checks |
| --- | --- |
| `q36_solo_gate.py` + `q36_cmp_gate.py` | The 10 parity prompts x 128 greedy tokens through `GEN`, compared with another build's. |
| `q36_batch_gate.py --slots N --waves --long 2500` | Every slot's tokens equal its solo tokens, slots reused, a long prompt read beside decoding slots. |
| `q36_proto_edge.py` | `max_new` 1, `BSTOP`, a busy slot, `STOP` during a prompt read, held-state reuse, sampled requests, the context's end. |
| `q36_interleave_probe.py` | What the decoding slots see while a long prompt is read. |
| `q36_vision_cmp.py` (with `q36_mtmd_dump.cpp` built against llama.cpp) | The picture comparison against llama.cpp. |
| `q36_vision_proto.py` | Pictures through `GENI` / `BGENI` (solo, in a batch, the conversation cache). |
| `q36_vision_serve.py`, `q36_serve_par.py`, `q36_serve_handoff.py` | The same through the server. |
| `q36_par_bench.py` | The aggregate-throughput A/B against `llama-server -np N`. |

## History / older measurements

### Pictures checked against llama.cpp

> **Measured on:** R9700 32 GB (HIP), UD-Q4_K_XL, Strata 0.1.45. llama.cpp's `mtmd` and its own graph, the encoder on the
> CPU on both sides, the same prompt ids, llama.cpp's answer fed to both, logits compared step by step over 24 steps.
> The floor is llama.cpp's CPU backend against its own HIP run.

| Picture and question | Prompt tokens | First token | Mean KL, llama.cpp HIP vs Strata | Floor |
| --- | ---: | --- | ---: | ---: |
| 640x400, text in the image | 288 | same | 0.029 | 0.056 |
| 640x400, list the shapes | 294 | same | 0.035 | 0.039 |
| 800x450, text in the image | 378 | same | 0.021 | 0.031 |
| 800x450, describe the shapes | 381 | same | 0.0005 | 0.0019 |
| 1024x1024, the title (1,024 image tokens, 3 chunks) | 1,042 | same | 0.0022 | 0.0026 |
| two pictures in one request | 655 | differs (a near tie; the floor's KL at this step is 0.16) | 0.014 | 0.022 |

The answers are the same text as llama.cpp's. With the picture options on, a request without pictures gives the same
tokens as without them (10 prompts x 128 greedy tokens).

### Aggregate decode speed, several requests at once

> **Measured on:** same prompts, greedy, 128 new tokens each, wall time from first request to last answer, 5
> interleaved pairs, medians, Strata 0.1.45 with `--batch 8 --mtp 0` against `llama-server -np N`.

| Card | Requests | Strata tok/s | llama.cpp | Ratio |
| --- | ---: | ---: | ---: | ---: |
| R9700 32 GB, UD-Q4_K_XL | 1 | 83.9 | 73.8 | 1.14 |
| | 2 | 173.5 | 107.4 | 1.61 |
| | 4 | 252.0 | 165.0 | 1.53 |
| | 8 | 325.4 | 208.1 | 1.56 |
| RTX 3060 12 GB, hybrid (llama.cpp `-ncmoe 28 -t 8`, the best split of a sweep) | 1 | 64.0 | 41.1 | 1.56 |
| | 2 | 85.1 | 52.5 | 1.62 |
| | 4 | 99.3 | 63.0 | 1.58 |

All pairs were faster for Strata. Every slot's tokens equal the tokens of the same request alone.

### Several requests with the base engine, and with `--batch-mtp`

> **Measured on:** R9700 32 GB, 5 interleaved pairs, Strata 0.1.45 (before the speed work below).

| Requests | `--mtp 0` | `--mtp 2 --batch-mtp` | Ratio |
| ---: | ---: | ---: | ---: |
| 1 | 83.7 tok/s | 134.0 | 1.60 |
| 2 | 173.1 | 207.0 | 1.20 |
| 4 | 251.0 | 275.0 | 1.10 |
| 8 | 324.0 | 320.5 | 0.99 (no row to spare: no proposals) |

> **Measured on:** hybrid cards, UD-Q4_K_XL, medians of 5 interleaved pairs, `--mtp 2`; "together" is the total tokens
> per second of N simultaneous requests against one request at a time with the draft layer.

| Card, context | Slots | One request alone, against 1 slot | N together, against one at a time |
| --- | ---: | ---: | ---: |
| RTX 3060 12 GB, 8K | 2 / 4 / 8 | -3.5% / -4.3% / -12% | 0.99x / 1.15x / 1.10x |
| RTX A4000 16 GB, 32K | 2 / 4 | -10.6% / -29.9% | 1.03x / 0.76x |
| Tesla P100 16 GB, 32K | 2 / 4 | -7% / -23% | 0.98x / 0.81x |

Without the draft layer (`--mtp 0`) the same slots give 1.26x to 1.64x together (3060 at 8K: 4 slots 1.64x for -5.5%
alone). A single request with drafts is already about 1.6x as fast as one without, so on a small card the main gain is
that a second user starts at once. On a card that holds every expert (R9700 32 GB, 8K prompts), 4 slots run 1.45x
one-at-a-time-with-drafts (1.59x with `--batch-mtp`).

### Prompt read beside decoding slots

> **Measured on:** R9700 32 GB, Strata 0.1.45, three decoding slots.

A 3,008-token prompt was read in 0.95 s (3,200 tok/s, 6 chunks). Each decoding slot paused at most 160 ms (one chunk)
and produced 20 tokens meanwhile. With `STRATA_Q36_READ_WINDOWS=4` they produced 65 tokens during a 1.14 s read.

### Speed work in 0.1.45

> **Measured on:** R9700 32 GB (HIP gfx1201), UD-Q4_K_XL, 5 interleaved pairs, 3 repetitions each, medians; base = the
> engine before 0.1.45.

| | Base | 0.1.45 | Ratio |
| --- | ---: | ---: | ---: |
| `--mtp 2`, 10 prompts greedy 128 | 167.7 tok/s | 183.7 | 1.10 |
| 1 / 2 / 4 / 8 requests (`--batch 8 --mtp 0`) | 85.2 / 182.4 / 271.9 / 361.1 | 122.4 / 205.1 / 303.3 / 386.9 | 1.44 / 1.12 / 1.12 / 1.07 |
| Prompt pp4096 at chunk 512 / 1024 / 2048 / 4096 | | 3,453 / 4,786 / 5,766 / 6,223 | |

UD-Q6_K_XL on the same card (every main layer's experts in VRAM, chunk stays 512): decode 80.4 to 106.9 tok/s, pp4096
2,160 to 2,200 (llama.cpp 3,486 at `-ub 2048`, 2,738 at `-ub 512`), `--mtp 2` 154.7 to 167.3. `--chunk 2048` there moves
the experts to the hybrid placement (205 of 256 per layer in VRAM): prompt 3,500 tok/s, decode 94. That is an option,
not the default.

> **Measured on:** RTX 3060 12 GB (CUDA sm_86, 20 threads), UD-Q4_K_XL in the hybrid mode (37 of 256 experts per layer
> in VRAM), Strata 0.1.45.

Decode tg128 58.5 to 59.7 tok/s (+2%), pp2048 1,958 to 2,022 (+3.3%), 10/10 prompts identical, plain and with `--mtp 2`.
llama.cpp 972b5fc CUDA (`-ncmoe 26 -t 8 -ub 2048`): pp2048 1,328, tg128 48 to 52. The GPU part of a token is the dense
Q8_0 at the card's bandwidth and the rest waits for the CPU pool, so only the small kernels' share is gained.

KL against the 512 chunk on six real 1.5K to 3.2K-token documents (48 teacher-forced rows each): chunk 1024 4.0e-3,
2048 4.2e-3, 4096 3.6e-3, top-1 agreement 44 to 48 of 48 rows. Greedy 64 tokens after those documents: the default chunk
gives the base engine's text for 2 of 6, `--prefill 512` for 6 of 6.

## Qwen3.6-35B-A3B on one or two small cards (this fork)

Qwen3.6-35B-A3B has its own engine path in Strata (`src/qwen35/q36_engine.cu`, chosen from the GGUF's
architecture `qwen35moe`): no pack, no expert profile, the Unsloth GGUF read as it is, the routed experts in VRAM
where they fit and otherwise in a pinned host arena with VRAM slots for the most-used ones and a CPU pool for the
rest (the hybrid mode), and the file's own MTP block for drafts. This page is about what this fork changed so that
it starts and runs well on 8 GB cards, and on two of them.

### What changed

- **Two cards, a layer split.** `--layer-split auto` (the default whenever two GPUs are visible; setup writes
  `"gpu": [0, 1]` and `"layer_split": "auto"` when both cards are chosen) prices the second card's free VRAM with
  the file's tensor sizes and puts the largest trailing block of layers there whose experts, dense weights, KV
  cache, the output head, the MTP block and the prompt chunk's buffers all fit, so **every expert of those layers
  is resident** and that card never streams - it may sit on a slow slot. The first card runs the other layers in
  the hybrid mode. The residual crosses once per batch (8 KiB per row) through pinned host memory; no
  peer-to-peer is needed. `--layer-split K` puts layers K.. on the second card by hand, `--layer-split 0` uses
  one card, `--split-device D` names the second card.
- **VRAM diet.** The token-embedding table stays in host memory (515 MiB), the attention-scores buffers are
  64 + 32 MB on a card under 12 GB, the split-K attention partials are sized for the batches that use them, the
  hybrid buffers for the hybrid batch, and the packed Q8_0 decode copies (a second copy of the dense matrices,
  1.8 GiB for this file) are off below 20 GB of VRAM (`STRATA_Q8_PACKED=1` turns them on).
- **Decode.** The hybrid decode runs in per-layer CUDA graphs (bit-exact with the eager path, no measurable
  speed change on an RTX 2080: the time is weight traffic, not launches). Sampled requests get their top-k from a
  device sort instead of a 1 MB logits copy per token, and **speculative decoding** with the MTP block's drafts
  verified against the sampled distribution (exact: the text is a draw from the same sampler). The CPU pool runs
  nine row tasks per thread (`--pool-tasks`), with `--pool-affinity auto|p-cores|all` and `--pin-host yes|no`,
  and on a CPU with AVX-VNNI its K-quant expert rows go through kernels that take up to 8 tokens per pass over
  the weights (bit-exact with ggml's; `STRATA_KQ_VNNI=0` keeps ggml's): the pool's time -14% on an i5-13400.

- **Re-sent conversations.** A chat client resends the whole conversation every turn, and the template renders
  the previous answer differently from the tokens the engine generated, so the held state was never an exact
  extension and every turn re-read everything. serve() now keeps checkpoints of the recurrent state in host
  memory (every GDN layer's conv history and state, ~63 MiB each, up to 8: after every full prompt chunk and 16
  tokens before a prompt's end, where the next turn diverges) and restores the latest one at or before the
  common prefix. Measured: a re-rendered follow-up after a 2,900-token prompt reads 31 tokens in 0.24 s instead
  of 2,900 in 3.3 s. `STRATA_Q36_PROMPT_TAIL=0` reads prompts in one go as before. A draft window never runs
  past the answer (the length limit or an end-of-sequence draft), so a turn that extends the answer exactly
  continues from the held state (10 tokens in 65 ms after that prompt).
- **An 8-bit KV cache** (`--kv q8_0`; setup writes it for a card under 12 GB or a context past 64K; `--kv f32` is
  upstream's). K and V rows are 8 fp16 scales and 256 int8 (272 bytes instead of 1,024): the same class as
  llama.cpp's q8_0 cache. 262K context fits on two 8 GB cards with it; 128K and 32K get more expert slots.
  Measured (engine bench, short chats): 32K 69-71 tok/s (82-107 with drafts), 128K 63-68 (94-97), 262K 58-61
  (76-79); after a 40,000-token prompt at 262K, 50-54 tok/s and the prompt read at 700-735 tok/s.
- **Elastic VRAM.** The streamed prompt path's buffers (the 2,048-row chunk and two layers of staging, ~2.1 GiB
  on the first card) exist only while a prompt longer than 32 tokens runs; between prompts their VRAM holds extra
  expert slots, filled by routing score after each such prompt (~185 ms over PCIe). Measured (engine bench, two
  cards, `--mtp 2`, 8-bit KV, before -> with it): 32K 69-71 -> 69-75 tok/s (82-107 -> 106-111 with drafts),
  128K 63-68 -> 63-72 (94-97 -> 87-106), 262K 58-61 -> 62-67 (76-79 -> 68-96, the first card's layers had no
  slot at all before); a 2,048-token prompt reads at 1,164 tok/s instead of 1,299 on the bench (its prompt
  carries the fill; through the server a 2,900-token prompt takes 3.28 s against 3.24 s).
  `STRATA_Q36_NO_ELASTIC=1` keeps the slots as placed; `STRATA_Q36_ASYNC_FILL=1` fills behind the first tokens
  (measured no faster through the server: the decode's uploads wait behind the fill's copies).
- **Long prompts.** The prompt attention sliced a chunk's queries so that their scores over the whole prefix fit
  a 64 MB buffer (10 rows a GEMM at 100K cells) and read the shared K and V once per query head. A prefix longer
  than 16K cells is now read in 16K-cell key chunks with wide query slices, one GEMM per kv head for its 8 query
  heads and an online softmax across the chunks. Measured: a 40,000-token prompt 700 -> 1,005-1,096 tok/s, a
  98,600-token prompt 224 -> 807 tok/s, the continuation identical. `STRATA_Q36_PROMPT_KEY_CHUNK_OFF=1` keeps the
  old path.
- **Long contexts.** The decode attention kernel read each cell's K row with one thread (32 scattered rows per
  warp) and decoded at 32 tok/s after a 14K prompt. The default kernel now gives one block to a chunk of 256
  cells and a kv head, reads its K and V once and scores the 8 query heads from them: 56-61 tok/s after a 14K
  prompt (61-64 with drafts). Its summation order differs from upstream's, so greedy text can drift after a few
  tokens; `STRATA_Q36_ATTN=v1` keeps upstream's kernel (bit-exact), `v2` the coalesced per-row one.

Everything else (the prompt path, the hybrid mode's slots and their rebalancing, the MTP drafting for greedy
requests) is upstream's.

### Measured on 2x RTX 2080 (8 GB) + i5-13400 + 32 GB DDR5-6000, UD-Q4_K_XL, 2026-10-16

The second card is on a PCIe 3.0 x4 slot, the first on x16 and it drives the desktop. `--max-context 32768`,
prompt chunk 2048 (the hybrid mode's), `--mtp 2`, the config `./setup.sh --yes --family qwen36 --model UD-Q4_K_XL
--gpus all` writes. The second card holds the last 6 layers and the MTP block with every expert resident
(3.34 GiB), the first card 24 of 256 experts per layer in VRAM and the rest of its 34 layers' experts in a
15.4 GiB pinned host arena. The stock engine does not start on an 8 GB card at all.

Through the server (`tools/strata_bench.py`, greedy, thinking off, medians of 3 runs):

| test | prompt | decode | first token |
| --- | ---: | ---: | ---: |
| short chat (46-token prompts, 100 tokens out) | - | 79.9 tok/s (66.8-92.2) | 0.36 s |
| 4K prompt, fresh | 1,216 tok/s | - | 3.3 s |
| 16K prompt, fresh | 1,207 tok/s | - | 13.3 s |
| agent session, turns 2-6 (3.7K to 14.5K tokens of context) | 814-977 tok/s over the new tokens, 68% reused | 83.0 tok/s (66-93) | 1.8-3.6 s |

The same suite on the 128K-context config setup writes now (8-bit KV, the elastic region, 2026-10-16 later that
day): chat decode 84.4 tok/s (80.4-99.5), first token 0.21 s; 4K prompt 1,170 tok/s; 16K prompt 1,202 tok/s;
agent session 68% reused, turns' first token 1.8-4.4 s, decode 87.4 tok/s (75.9-94.5).

The same requests to llama.cpp (b11559) with the user's own settings (`--spec-type draft-mtp --spec-draft-n-max 3
-ngl 999 -fa on -ctk q8_0 -ctv q8_0 --n-cpu-moe 32 -ts 3,1 -t 10`), both at temperature 1.0 / top-p 0.95 / top-k
20, the servers' own timings:

| request | this fork (two cards) | llama-server |
| --- | ---: | ---: |
| short chat, prose, 100 tokens | 65-84 tok/s | 44-48 tok/s |
| short chat, code | 78-93 | 57-65 |
| 3,197-token prompt (fresh) | 946-967 tok/s | 546 tok/s |
| 24 tokens right after that prompt | 50-58 | 43 |

The engine's own bench (`--bench-pp N --bench-tg 64`, 3 reps, random prompts; llama-bench the same way without
drafts): a 2,048-token prompt 1,084 / 1,289 / 1,298 tok/s against 414; greedy decode 62-70 tok/s against 45.0;
with 2 drafts 68-80. After a 14,000-token prompt: 56-61 tok/s (61-64 with drafts).

### Accuracy against llama.cpp (the same GGUF, the same PC, 2026-10-16)

Teacher-forced next-token distributions on four sequences (memorized prose, code, a document, a chat; 160 to
1,408 steps each; `tools/qwen35/tf_compare.py`): Strata's distance from llama.cpp's GPU run (b11559, the user's
llama-server placement) is the distance between llama.cpp's own GPU and CPU runs - KL mean 0.0028-0.044 against
the floor's 0.0027-0.041, top-1 agreement 96.0-98.1% against 95.8-97.5%, perplexities equal to the third digit
(code 4.322 / 4.314, document 6.621 / 6.587). Needle in a haystack (`tools/needle_bench.py`) at 8K, 32K, 64K and
120K tokens, depths 10 / 50 / 90%: 12 of 12 found by both engines, the 120K prompt read in 159 s here against 415 s there; at the 262K configuration 160K, 200K and 250K-token prompts at the same depths, 9 of 9 found (the 250K prompt read in about 9.5 minutes). GSM8K, the first 40 test problems with thinking on
(`tools/gsm8k_eval.py`, Qwen's sampling, 6,144-token limit): 37 of 40 against llama-server's 35 of 40, median
1,335 against 1,425 completion tokens, 94.7 against 60.4 tok/s. AGENTS_FORK.md has the tables.

Where a decoded token's time goes (nsys, short context): ~6.5 ms of GPU kernels (the Q8_0 dense GEMVs near the
card's bandwidth), ~5 ms on the CPU pool (32 us per missed expert, DRAM-bound), serialized by the layer
dependency; ~2 ms of host overhead. Drafts amortize all of it.

### Settings worth knowing

- `--chunk 1024` or `512`: smaller prompt buffers, so more expert slots on the first card and a few more layers on
  the second (33 / 39 slots per layer, 7 / 8 resident layers), but the prompt streams the first card's experts
  once per chunk over PCIe: 700 / 350 tok/s instead of 1,290 at 2048. Decode +3-5%.
- `--max-context`: the cache is allocated for the whole context at start and takes the first card's expert-slot
  room: with the 8-bit cache 32K leaves 38 slots per layer, 128K 18, 262K none, plus the elastic region's 36 / 35
  / 30 between prompts (at 262K every expert of the first card's layers ran on the CPU pool before it: 58-61
  tok/s plain, 76-79 with drafts). `./setup.sh --setup --context N` changes it; fp32 K/V (`--kv f32`, 45 KB per
  token) runs up to 64K here.
- `STRATA_Q36_NO_LAYER_GRAPH=1`: the eager decode; `STRATA_Q36_NO_SPEC_SAMPLING=1`: sampled requests one token at
  a time; `STRATA_Q36_NO_ELASTIC=1`, `STRATA_Q36_ASYNC_FILL=1` (above); `STRATA_KQ_VNNI=0`: ggml's CPU rows;
  `--pool-tasks N`, `--pool-affinity`, `--pin-host no`. The engine's stats line at exit says how the hybrid
  batches split between the GPU and the pool, the slot replacements and the elastic switches' time.
- Host memory: the first card's experts are pinned (15.4 GiB here) and cannot be swapped; a 24 GB swap file was
  added on this PC as a safety net for everything else.
