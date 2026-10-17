# Qwen3.6-35B-A3B on one or two small cards (this fork)

Qwen3.6-35B-A3B has its own engine path in Strata (`src/qwen35/q36_engine.cu`, chosen from the GGUF's
architecture `qwen35moe`): no pack, no expert profile, the Unsloth GGUF read as it is, the routed experts in VRAM
where they fit and otherwise in a pinned host arena with VRAM slots for the most-used ones and a CPU pool for the
rest (the hybrid mode), and the file's own MTP block for drafts. This page is about what this fork changed so that
it starts and runs well on 8 GB cards, and on two of them.

## What changed

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

## Measured on 2x RTX 2080 (8 GB) + i5-13400 + 32 GB DDR5-6000, UD-Q4_K_XL, 2026-10-16

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

Where a decoded token's time goes (nsys, short context): ~6.5 ms of GPU kernels (the Q8_0 dense GEMVs near the
card's bandwidth), ~5 ms on the CPU pool (32 us per missed expert, DRAM-bound), serialized by the layer
dependency; ~2 ms of host overhead. Drafts amortize all of it.

## Settings worth knowing

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
