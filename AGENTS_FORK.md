# AGENTS_FORK.md - project log for this fork

This fork tailors Strata to one PC and one model: **Qwen3.6-35B-A3B (Unsloth UD-Q4_K_XL)** on an
Intel Core i5-13400 (6 P + 4 E cores, AVX2), 32 GB DDR5 (31.1 GiB usable), two RTX 2080 (8 GB each, sm_75),
Linux (kernel 7.0), CUDA 13.4, driver 615.71. GPU 0 is on PCIe 3.0 x16 and drives the desktop (~390 MiB used);
GPU 1 is on PCIe 3.0 x4. No peer-to-peer between the cards (`nvidia-smi topo -p2p r`: CNS).
Goal: start at all, then run as fast as possible, at least as fast as stock Strata would with enough memory.

Measured numbers below say what they were measured on. Stock = upstream v0.1.44.1 (commit 296b59d2).

## 2026-10-16 - reproduction: the stock engine cannot start on an 8 GB card

Setup itself works: `./setup.sh --yes --family qwen36 --model UD-Q4_K_XL --no-start` compiles the engine for
sm_75 (no ready-made engine is published for v0.1.44.1) and writes `strata-qwen36-ud-q4_k_xl.json`
(`--native <gguf> --max-context 32768 --prefill auto --mtp 2`, GPU 0 only: "Qwen3.6-35B-A3B runs on one GPU so
far"). It warns that UD-Q4_K_XL "needs about 32 GB of RAM; this PC has 31 GB".

The start fails in **VRAM**, not RAM (`strata-qwen36-ud-q4_k_xl.log`):

1. Stock: the dense weights take **4.40 GiB** of the 8 GB card (Q8_0 dense 1.49 GiB + output head 0.5 GiB +
   token embeddings 0.5 GiB + a second "packed Q8_0" copy of the Q8_0 matrices, ~1.9 GiB, `STRATA_Q8_PACKED=1`
   by default for this family), then `Engine::init` fails: `cudaMalloc(&p, n * 4) failed: out of memory`
   (q36_engine.cu:1024).
2. With `STRATA_Q8_PACKED=0`: 2.62 GiB of dense weights; `Engine::init` at context 32768 and prompt chunk 2048
   (the hybrid mode's chunk) takes ~3.2 GiB more (KV cache 11 layers x 2 x 32768 x 256 fp32 = 1.375 GiB, the
   prompt attention's 256 MB + 128 MB score buffers, 2048-row activation buffers, MMQ buffers) and leaves
   **0.94 GiB free**; `place_experts` then fails on the two 464 MiB staging buffers for streamed prompt layers
   (q36_engine.cu:2110). Nothing is left for expert slots either.

So the 8 GB card never reaches the point where host memory matters. The host side would need ~19 GiB
(every expert's blob in one `cudaHostAlloc` arena, pageable `malloc` fallback) beside the mmap'd 21 GiB file.

The model (from the GGUF): 40 main layers (30 Gated DeltaNet + 10 full attention) + 1 MTP block, n_embd 2048,
256 experts top-8, expert FF 512; experts 18.74 GiB (464 MiB per layer, 1.81 MiB per expert), dense Q8_0
1.49 GiB, embeddings and head 515 MiB each; KV per token 45,056 bytes (fp32).

## 2026-10-16 - first runs of the stock engine that start (one RTX 2080, GPU 0)

Engine built by setup from this tree (sm_75, CUDA 13.4), `engine/strata --native <gguf> --mtp 2 --bench-pp N
--bench-tg 64` (the engine's own llama-bench-like loop: 3 repetitions, greedy decode one token at a time, no
MTP drafting in the loop). Host: the 18.75 GiB expert arena was pinned (`cudaHostAlloc` succeeded), MemAvailable
never fell under 9 GiB; the process RSS reads ~30 GB only because the mmap'd 21 GiB file is counted.

| run | settings | experts in VRAM per layer | prompt | decode |
| --- | --- | ---: | ---: | ---: |
| run3 | `STRATA_Q8_PACKED=0 --max-context 8192 --chunk 512` | 13 of 256 (0.95 GiB) + 1.10 GiB staging | pp512 291-304 tok/s | tg64 48.7-55.3 tok/s |
| e3 | the same + `STRATA_Q36_NO_MMQ=1` (no streamed prompt path: prompts on slots + the CPU pool) | 28 of 256 (2.05 GiB) | pp2048 143 / 438 / 372 tok/s | tg64 48.6 / 52.4 / 47.4 tok/s |

Where the decode time goes (run3, 19.2 ms per token): the CPU pool computed 257 of 320 routed-expert entries
per token (the 40 main layers x 8) in 10.4 ms (~45 GB/s of DRAM reads: 1.81 MiB per expert); the other ~9 ms
is the GPU side: the Q8_0 dense weights and the output head (2.0 GiB per token at the RTX 2080's 448 GB/s =
4.5 ms at best), plus ~800 kernel launches per token (no CUDA graph in the hybrid mode) and one host sync per
layer for the routing. Doubling the slots (13 -> 28) did not move the decode: the hit rate on the popularity
slots is still low and the GPU side is half the time.

The prompt at chunk 512 is PCIe-bound: every chunk streams all 41 layers' experts (18.6 GiB) over PCIe 3.0 x16.

### The llama.cpp comparison (the user's own settings)

`~/bin/llama.cpp/bin/llama-bench` (build f2cc7282c), the same GGUF, the user's run script's offload: `-ngl 999
-ncmoe 32 -ts 3/1 -fa on -ctk q8_0 -ctv q8_0 -t 10` (32 of 41 expert layers on the CPU, the rest split 3:1 over the
two cards), 3 repetitions:

| test | llama.cpp |
| --- | ---: |
| pp512 | 411.1 ± 5.7 tok/s |
| pp2048 | 414.0 ± 2.4 tok/s |
| tg64 | 45.0 ± 0.7 tok/s |

Target of this fork: beat both numbers clearly, with the stock engine's quality (same weights, same math).

## 2026-10-16 - design: a two-card layer split for the Qwen3.6 engine path

Why a layer split and not a second expert tier: the second RTX 2080 sits on a PCIe 3.0 x4 slot (~3 GB/s) with no
peer-to-peer, so anything streamed through it per prompt chunk is 4x slower than on the first card. With a layer
split the last layers live entirely on the second card (their dense weights, KV cache, expert slots), it streams
nothing when all of their experts are resident, and the residual crosses once per batch (8 KiB per row) through
pinned host memory. The CPU pool stays shared. Flash-Next's own split (docs/MULTI_GPU.md) is the same shape.

What changed in `src/qwen35/q36_engine.cu` (the single-device `Engine` became `Stage`, one per card, driven by a
new `Engine`):

- `--layer-split auto|K|0` and `--split-device D`. `auto` (the default whenever two GPUs are visible) prices the
  second card's free VRAM with the file's tensor sizes and takes the largest trailing block of layers whose experts,
  dense weights, KV cache (at `--max-context`), the output head, the MTP block and the chunk's buffers all fit, so
  every expert of those layers is resident there. K puts layers K.. on the second card by hand.
- Each stage allocates only its own layers' state and buffers; the MTP block, output head and logits are on the
  last stage. Expert placement is per stage (all-resident or hybrid with its own slot count and staging buffers);
  the host arena is one `cudaHostAllocPortable` block so both cards' copy engines read it.
- VRAM diet (both cards): the token-embedding table stays in host memory (515 MiB saved; a batch's rows are
  staged through pinned memory), the attention-scores buffers are 64 + 32 MB on a card under 12 GB (shrunk by
  `shrink_att` before only in one corner case), the split-K attention partials are sized for the <= 8-row batches
  that use them (16 MiB instead of 128 MiB at 32K), the hybrid buffers are sized by the hybrid batch (128 rows)
  instead of the chunk (2048), and the packed Q8_0 decode copies (1.8 GiB) are off by default on a card under
  20 GB (`STRATA_Q8_PACKED=1` turns them on).
- The MTP draft step runs as a CUDA graph when the MTP block's experts are resident, even in the hybrid mode.

Parity (greedy text, 29-token prompt, 48 tokens with `--mtp 2`, `--expert-slots 13` and `STRATA_Q36_FIXED_SLOTS=1`
so that the same experts are resident in every arm): the stock engine on one card, the new engine on one card
(with the 256 MB and with the 64 MB attention buffers) and the new engine on two cards (`--layer-split 30`)
produced the identical 48 tokens and the same 28 of 40 drafts accepted. The split is bit-exact.

### Measured: the automatic split on 2x RTX 2080, by prompt chunk (engine bench, 3 reps, before the layer graphs)

`CUDA_VISIBLE_DEVICES=0,1 build/strata --native <gguf> --max-context 32768 --mtp 2 --bench-pp 2048 --bench-tg 64`,
the chunk decides the second card's room (its buffers scale with the chunk) and the first card's expert slots:

| chunk | CUDA1 holds (all experts resident) | CUDA0 slots per layer | pp2048 (3 reps) | tg64 (3 reps) |
| ---: | --- | ---: | ---: | ---: |
| 2048 (auto) | layers 34-39 + MTP | 24 of 256 | 1084 / 1289 / 1298 tok/s | 55.4 / 55.0 / 56.9 tok/s |
| 1024 | layers 33-39 + MTP | 33 of 256 | 620 / 700 / 700 | 57.2 / 59.0 / 60.5 |
| 512 | layers 32-39 + MTP | 39 of 256 | 346 / 368 / - | 56.5 / 58.8 / - |

llama.cpp (the user's settings): pp2048 414, tg64 45.0. The prompt is PCIe-bound (the first card streams its
hybrid layers' experts once per chunk), so chunk 2048 stays the default. Decode: the CPU pool took 6.7 ms of the
17.9 ms per token at chunk 2048 (217 of 272 routed entries per token on the pool); the other ~11 ms is the GPU
side, of which the dense weights (1.5 GiB over both cards at 448 GB/s) are ~4.5 ms: the rest is per-layer launch
and sync overhead, which is what the per-layer CUDA graphs (next) address.

## 2026-10-16 - per-layer CUDA graphs: exact, but no faster; where a decoded token's time goes

The hybrid decode now runs each layer as two graphs (the mixer + router + the routing's copies to the host; the
resident and shared experts) with the CPU pool between them, and a stage tail graph (the last merge + combine, then
the head or the hand-off); a layer whose experts are all resident is one graph. ~70 graph launches per token
instead of ~700 kernel launches. Greedy text: identical to the eager path (48 tokens, 28 of 40 drafts accepted).
A verify window of one row (the drafts are clamped at a rebalance boundary) must not reuse the decode step's graph
(its GDN kernels write the state): the graphs are keyed by (verify, rows). `STRATA_Q36_NO_LAYER_GRAPH=1` or
`--no-graph` restores the eager path.

Measured (two cards, auto split, chunk 2048, tg64, 3 reps): graphs 47.8 / 49.7 / 53.6 tok/s against eager
55.4 / 55.0 / 56.9 - no gain (within noise, if anything slower). An nsys profile of a short eager decode (pp16 +
tg48, 3 reps, `--mtp 0`) says why: the GPU kernels of a token sum to ~6.5 ms (the Q8_0 dense GEMVs alone
783 ms over the run, 25.6 us each on average: near the card's bandwidth), the CPU pool ~5 ms (32 us per routed
expert entry), and they are serialized by the data dependency (a layer's combine needs the pool's rows before the
next layer's mixer). Launch overhead was never the bottleneck; the host-side overhead outside the pool is ~2 ms.
So decode on this PC is bounded by: GPU weight traffic (~1.5 GiB per token over both cards) + the CPU pool's
DRAM traffic for the missed experts. The levers left: the hit rate (VRAM for slots), the pool's efficiency, the
packed Q8_0 GEMV kernels where VRAM allows, and drafts (MTP) to amortize all of it: greedy with 2 drafts measured
78.4 tok/s over 256 tokens (68% of drafts accepted) in the engine's `--tokens` mode, 79.6 tok/s on a 160-token
`--serve` request.

### Measured: pool tasks, pool affinity, sampled requests (two cards, auto split, chunk 2048, 32K context)

Engine bench pp512 + tg64, 3 reps, and the same 64 tokens again with 2 MTP drafts (greedy):

| pool setting | tg64 | tg64 with 2 drafts | pool time for the run |
| --- | ---: | ---: | ---: |
| default (10 threads, 3 tasks per thread) | 60.8 / 64.7 / 64.3 tok/s | 63.6 / 68.1 / 67.6 | 2410 ms |
| `--pool-tasks 90` (9 per thread) | 62.0 / 67.3 / 67.3 | 67.3 / 72.3 / 71.8 | 2155 ms |
| `--pool-affinity p-cores` (6 threads) | 52.8 / 66.8 / 64.5 | 54.2 / 68.1 / 69.2 | 2697 ms |

Nine tasks per thread is the engine's default now (the E-cores' tail at each phase barrier shrinks); the E-cores
stay in the pool (the P-cores alone read less of the DRAM bandwidth).

Sampled requests through `--serve` (29-token prompt, 160 tokens, `temperature=1.0 top_p=0.95 top_k=20`, then
`temperature=0.7`): one token at a time 63.8 / 73.0 tok/s; with the new speculative sampling (the MTP block's
greedy drafts verified against the sampled distribution, exact) 68.5 / 76.3 tok/s, 49% / 53% of drafts accepted.
The greedy request on the same prompt: 79.6 tok/s (62% accepted). The top-k of a sampled request is sorted on
the device now (a radix sort of the 248,320 logits) instead of copying them to the host every token.
`STRATA_Q8_PACKED=1` cannot start on one 8 GB card even at 8K context (the staging buffers do not fit).

## 2026-10-16 - end to end: setup with both cards, the server, the repo's benchmark, llama-server

`./setup.sh --yes --family qwen36 --model UD-Q4_K_XL --gpus all --no-start` compiles the engine from this tree
(incremental), writes `"gpu": [0, 1]` and `"layer_split": "auto"` into `strata-qwen36-ud-q4_k_xl.json` and the run
script; the server starts it with `CUDA_VISIBLE_DEVICES=0,1 --layer-split auto` and `/health` says loaded.

`tools/strata_bench.py` against the running server (greedy, thinking off, medians of 3):

| test | prompt tok/s | decode tok/s | first token |
| --- | ---: | ---: | ---: |
| short chat (46-token prompts, 100 out) | - | 79.4 (67.6-92.1) | 0.25 s |
| 4K prompt, fresh | 1,263 | - | 3.19 s |
| 16K prompt, fresh | 1,225 | - | 13.1 s |
| agent session, turns 1-5 (3.8K -> 14.5K tokens) | 1,102-1,195 | 54 -> 27 | 3.3 -> 13.0 s |

Two weaknesses for agent use came out of it: (1) the engine held no reusable state between turns (0 cached
tokens: the chat template renders the previous answer differently from the tokens the engine generated, so the
new prompt is never an exact extension), so every turn re-read the whole conversation; (2) decode fell from 69
to 27 tok/s as the context grew to 14.5K tokens. (1) is addressed next with checkpoints of the recurrent state;
(2) is measured next.

The same requests to llama-server with the user's own script's settings (`--spec-type draft-mtp
--spec-draft-n-max 3 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 --n-cpu-moe 32 -ts 3,1 -c 262144 -t 10`, no mmproj) and to
the fork's server, through the OpenAI API, the servers' own timings, two runs each:

| request | fork (two cards) | llama-server |
| --- | ---: | ---: |
| short chat, prose, temperature 1.0, 100 tokens | 65-72 tok/s | 44-48 tok/s |
| short chat, code, temperature 1.0 | 84-92 | 57-65 |
| short chat, greedy | 60-89 | 45-68 |
| 3,197-token prompt (fresh) | 997-1,018 tok/s | 546 tok/s |
| decode right after that prompt (24 tokens) | 40-51 | 43 |

llama-server's prompt cache made its repeated prompts free ("prompt 4 tok"); the fork's engine read them again
every time (point (1) above).

## 2026-10-16 - checkpoints for re-sent conversations; the decode attention kernel

**Checkpoints.** The engine's prefix reuse worked only for an exact extension of the held tokens (measured
through the serve protocol: a prompt of 29 tokens + its 32-token answer + 8 new tokens reused 61). A chat turn
re-rendered by the template diverges a few tokens before the previous prompt's end (the empty think block of the
generation prompt is not rendered again), so the server's agent session reused nothing. Now serve() reads a
prompt up to its last 16 tokens in chunks (a checkpoint of the recurrent state - every GDN layer's conv history
and state matrix, hlast, ~63 MiB in host memory - after every full chunk), takes a checkpoint, then reads the
tail as a small batch (slots + the pool, no streaming); a request that is not an exact extension restores the
latest checkpoint at or before its common prefix with the held tokens and reads the rest (the KV cache keeps its
cells below the position; the recurrent state cannot step back without a checkpoint). Up to 8 checkpoints are
kept; `STRATA_Q36_PROMPT_TAIL=0` reads prompts in one go as before.

Measured (two cards, serve protocol, a 2,900-token prompt):

| request after the 2,900-token prompt | reused | read | prompt time |
| --- | ---: | ---: | ---: |
| the same + the answer + a new turn (exact extension) | 2,932 | 10 | 84 ms |
| the answer rendered differently (diverges at its first token) | 2,884 | 31 | 240 ms |
| an edit at token 2,500 | 2,048 (the chunk checkpoint) | 854 | 1.6 s |
| an edit at token 100 | 0 | 2,902 | 3.1 s |

The tail's 16 tokens take the hybrid rows path (resident experts on the GPU, the rest on the pool), not the
streamed MMQ path, so the greedy text of a prompt is no longer bit-identical to upstream's for those tokens (the
same arithmetic every generated token already uses).

**Attention.** After a 14K-token prompt the decode fell to 32 tok/s (the CPU pool's share unchanged): the
split-K decode attention kernel had every thread walk its own cell's K row (32 scattered rows per warp). A
coalesced version (each warp scores 32 cells, a K row as 32 x 8 floats across the lanes, shuffle-reduced):
14K context 37.7 / 40.3 / 40.2 tok/s (was 31.7 / 33.1 / 33.2), with 2 drafts 33.3 / 34.1 / 34.1 (was 26.9 /
27.4 / 27.5); short context tg64 65.6 / 71.2 / 72.8 (was 60.8 / 64.7 / 64.3), with drafts 80.3 / 74.6 / 79.1.
The dot product's summation order changed, so the greedy text drifts from upstream's after a few tokens (a
near-tie flipped); `STRATA_Q36_ATTN=v1` keeps upstream's kernel (the parity reference). A third version reads
each chunk's K / V once for the 8 query heads of a kv head (upstream read them 8 times over): numbers below.

The third version (`STRATA_Q36_ATTN=v3`, the default; one block per chunk and kv head, its 8 query heads' scores
from one read of the chunk's K and V): 14K context tg32 56.4 / 61.1 / 60.9 tok/s (upstream's kernel: 31.7 /
33.1 / 33.2), with 2 drafts 61.2 / 63.8 / 63.6 (upstream: 26.9 / 27.4 / 27.5); short context tg64 62.0 / 67.0 /
70.2, with drafts 71.8 / 68.2 / 77.6 (the same as v2 within noise). The long-context penalty is mostly gone:
decode after a 14K prompt is now ~90% of the short-context rate instead of ~50%. `STRATA_Q36_ATTN=v1`
reproduces the upstream greedy text exactly (checked: the 48-token parity prompt).

## 2026-10-16 - the final engine through the server (commit 2ce1bdc3 + this log)

`./setup.sh --yes --family qwen36 --model UD-Q4_K_XL --gpus all --no-start`, then `tools/strata_bench.py`
against the running server (greedy, thinking off, medians of 3):

| test | prompt tok/s | decode tok/s | first token |
| --- | ---: | ---: | ---: |
| short chat (46-token prompts, 100 out) | - | 79.9 (66.8-92.2) | 0.36 s |
| 4K prompt, fresh | 1,216 | - | 3.31 s |
| 16K prompt, fresh | 1,207 | - | 13.3 s |
| agent session, follow-up turns (3.7K -> 14.5K tokens) | 814-977 (the new tokens only) | 83.0 (66.1-93.1) | 3.43 s (1.77-3.60) |

The agent session before the checkpoints and the attention kernel: 0% reused, first token 3.3 -> 13.0 s over
the turns, decode 54 -> 27 tok/s; now 68% of the follow-up turns' tokens are reused (292 tokens per session
read again: the prompt tails and the template's divergence), the first token takes 1.8-3.6 s and decode stays at
74-90 tok/s at 14.5K tokens of context.

Sampled requests through the same server (temperature 1.0, top-p 0.95, top-k 20, the user's llama.cpp
settings): short chats 65-93 tok/s, the 3,197-token prompt 946-967 tok/s, 24 tokens after it 50-58 tok/s
(greedy: 68-69); llama-server with the user's settings: 44-68, 546, 43.

## 2026-10-16 - a review of the engine changes (four readers, two skeptics per finding), and the fixes

Twelve findings; the skeptics confirmed ten by reading the code. Fixed in the same commit as this entry:

- A last stage placed "MTP block only on the pool" beside a hybrid first stage inherited the engine-wide
  `main_hybrid` and streamed the MTP block's prompt pass through staging buffers it never allocated (exit at
  warm-up). Each stage's `main_hybrid` now follows its own plan; such a stage runs every batch on the hybrid rows
  path, as the single-card mtp_only placement does.
- serve() dropped the kept drafts after the one that ended an answer from `live`, although the engine had
  committed them all: a later request extending that text would have reused a state one or more tokens ahead.
  The kept drafts go into `live` before any of them is emitted.
- The per-layer graph path took every verify window up to 8 rows, but the hybrid buffers hold `--hybrid-rows`
  rows (128 by default; a smaller value would have overrun them): gated on the hybrid batch size too, and the
  buffers hold at least 8 rows.
- With no draft left before a rebalance boundary, the verify window's staging (the embedding rows, the position
  word) could be rewritten while the MTP call's uploads were still queued on the same card: a stream sync there.
- The AMD (HIP) build has no CUB: the device sort and the speculative sampling are CUDA-only; HIP keeps
  upstream's host sampling (not built here: no ROCm on this PC).
- The chunk rule credited the packed Q8_0 copies as returnable on two cards, where they never are.
- The DONE line's offered-drafts count used the full `--mtp` even when the step clamped it.
- setup: three cards with a two-point `--layer-split` kept both points after the list was cut to two cards (the
  server refuses such a config): "auto" with a note.
- The split planner priced the chunk's buffers per row with tile-padded sizes that init allocates once per batch,
  so it left a layer the second card could hold: priced per batch now.

Two findings were refuted (the planner's pricing order; a staging race that the hybrid MTP layer's host wait
already covers) and two are accepted as is: the greedy text of a prompt's last 16 tokens is not bit-identical to
upstream's (the hybrid rows path instead of the streamed MMQ path), and the attention kernel's sum order (see
above, `STRATA_Q36_ATTN=v1` restores both... the kernel only; `STRATA_Q36_PROMPT_TAIL=0` the prompt path).

Verified after the fixes (two cards): `STRATA_Q36_ATTN=v1` still reproduces the upstream 48-token greedy text;
`--layer-split 25` (both stages hybrid: 48 and 80 slots per layer) runs; a drafted answer that ends inside a kept
window followed by its exact extension reuses every committed token (157 of 160; the same request again
restores the tail checkpoint at 100 and reproduces the identical greedy continuation). The planner now takes 7
layers on the second card (33-39 + MTP) and leaves 25 slots per layer on the first. Engine bench, chunk 2048,
32K context: pp2048 1,195 / 1,298 / 1,300 tok/s, tg64 65.0 / 68.3 / 68.2, with 2 drafts 93.6 / 97.7 / 87.8
(40 of 48 accepted); after a 14,000-token prompt tg32 52-62, with drafts 59-65.

## 2026-10-16 - contexts beyond 32K; a fresh clone set up from scratch

Engine bench on the two cards, pp2048 + tg64, `--mtp 2`: at `--max-context 65536` the split keeps 6 layers on the
second card and 6 slots per layer on the first (the fp32 KV cache takes the slot room): pp2048 1,110 / 1,289 /
1,297 tok/s, tg64 59.3 / 63.2 / 63.1, with drafts 70.1 / 71.7 / 72.1. At 131072 the start fails (`cudaMalloc ...
out of memory` for the KV cache on the first card): this engine path stores K and V as fp32 (45 KB per token,
11 GiB at 262K). llama.cpp reaches 262K on this PC with its q8_0 KV cache (~11 KB per token). An 8-bit KV cache
for this path is the next step.

A fresh `git clone -b fork/2080x2` into another folder, `./setup.sh --yes --family qwen36 --model UD-Q4_K_XL
--no-start --data-dir <the data folder>`: both cards chosen by default, no RAM question (the family's RAM need is
28 GB now: measured 15.4 GiB pinned with two cards), the engine compiled from the clone (no ready-made engine for
0.1.44.1), the two-card config and run script written; its server loaded and answered a chat request. Nothing
else was running at the time (two instances would each need an arena and the cards).
