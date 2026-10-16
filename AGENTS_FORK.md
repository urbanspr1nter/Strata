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
