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
  nine row tasks per thread (`--pool-tasks`), with `--pool-affinity auto|p-cores|all` and `--pin-host yes|no`.

Everything else (the prompt path, the hybrid mode's slots and their rebalancing, the MTP drafting for greedy
requests) is upstream's.

## Measured on 2x RTX 2080 (8 GB) + i5-13400 + 32 GB DDR5-6000, UD-Q4_K_XL, 2026-10-16

The second card is on a PCIe 3.0 x4 slot, the first on x16 and it drives the desktop. `--max-context 32768`,
prompt chunk 2048 (the hybrid mode's), `--mtp 2`; the engine's own bench (3 repetitions, a random prompt), and
llama.cpp (b11559) with the user's settings (`-ngl 999 --n-cpu-moe 32 -ts 3,1 -fa on -ctk q8_0 -ctv q8_0 -t 10`):

| | Strata (this fork, two cards) | llama.cpp (two cards) |
| --- | ---: | ---: |
| reads a 2,048-token prompt | 1,084 / 1,289 / 1,298 tok/s | 414 tok/s |
| writes (greedy, one token at a time) | 55-67 tok/s | 45.0 tok/s |
| writes (greedy, 2 MTP drafts) | 68-80 tok/s | - |

The stock engine does not start on an 8 GB card at all (VRAM: 4.4 GiB of dense weights with the packed copies,
then the buffers). With the split the second card holds the last 6 layers and the MTP block with every expert
resident (3.34 GiB), the first card 24 of 256 experts per layer in VRAM and the rest of its 34 layers' experts in a
15.4 GiB pinned host arena.

Where a decoded token's time goes (nsys, eager decode): ~6.5 ms of GPU kernels (the Q8_0 dense GEMVs near the
card's bandwidth), ~5 ms on the CPU pool (32 us per missed expert, DRAM-bound), serialized by the layer
dependency; ~2 ms of host overhead. Drafts amortize all of it.

## Settings worth knowing

- `--chunk 1024` or `512`: smaller prompt buffers, so more expert slots on the first card and a few more layers on
  the second (33 / 39 slots per layer, 7 / 8 resident layers), but the prompt streams the first card's experts
  once per chunk over PCIe: 700 / 350 tok/s instead of 1,290 at 2048. Decode +3-5%.
- `--max-context`: the KV cache is fp32 (45 KB per token) and is split between the cards by layer; 32K costs
  ~1.4 GiB in all.
- `STRATA_Q36_NO_LAYER_GRAPH=1`: the eager decode; `STRATA_Q36_NO_SPEC_SAMPLING=1`: sampled requests one token at
  a time; `--pool-tasks N`, `--pool-affinity`, `--pin-host no`.
- Host memory: the first card's experts are pinned (15.4 GiB here) and cannot be swapped; a 24 GB swap file was
  added on this PC as a safety net for everything else.
