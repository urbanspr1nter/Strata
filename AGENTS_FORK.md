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
