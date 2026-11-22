#!/usr/bin/env python3
"""tools/qwen35/tf_compare.py - teacher-forced next-token distributions of Strata's Qwen3.6 engine and llama.cpp on
the same token sequence and the same GGUF: KL(reference || engine) per step, top-1 agreement, each engine's perplexity
and next-token accuracy on the forced tokens.  The reference is llama.cpp on the GPU (q36_llama_dump.cpp built against
a llama.cpp tree, with the same placement as a llama-server run: --llama-args); llama.cpp on the CPU alone is the
noise floor an engine is judged against (two backends of the same program).

    tf_compare.py --model GGUF --dump q36_llama_dump --work DIR --name T --ids FILE --prompt-len K --run strata|llama|cpu
    tf_compare.py --model GGUF --work DIR --name T --ids FILE --prompt-len K --report

FILE holds the sequence's token ids (llama-tokenize --ids, brackets and commas stripped); the first K are the prompt,
the rest are forced one at a time, and every step's logits go to DIR/T.<run>.bin ([step][vocab] f32, 1 MB a step).
The 2026-10-16 run on this fork is in AGENTS_FORK.md (Strata within the floor on prose, code, a document and a chat)."""
import argparse, os, subprocess, sys, time
import numpy as np
V = 248320
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="the GGUF")
ap.add_argument("--work", required=True); ap.add_argument("--name", required=True); ap.add_argument("--ids", required=True)
ap.add_argument("--prompt-len", type=int, required=True); ap.add_argument("--run", default=""); ap.add_argument("--report", action="store_true")
ap.add_argument("--strata", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build", "strata"))
ap.add_argument("--dump", default="q36_llama_dump", help="tools/qwen35/q36_llama_dump.cpp, built")
ap.add_argument("--strata-args", default="--max-context 8192 --kv q8_0 --mtp 0 --layer-split auto")
ap.add_argument("--llama-args", default="--ngl 999 --n-cpu-moe 32 --tensor-split 3,1 --threads 10 --kv q8_0 --ctx 8192 --batch 2048 --ubatch 512",
                help="the dump tool's placement for the GPU reference (here: llama-server's settings on 2x 8 GB)")
ap.add_argument("--cpu-args", default="--ngl 0 --threads 10 --ctx 8192")
ap.add_argument("--gpus", default="0,1", help="CUDA_VISIBLE_DEVICES for the runs")
a = ap.parse_args()
G = a.model
ids = [int(x) for x in open(a.ids).read().split()]
P, F = ids[:a.prompt_len], ids[a.prompt_len:]
pid = ",".join(map(str, P)); fid = ",".join(map(str, F))
base = os.path.join(a.work, a.name)
env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus)
if a.run:
    t0 = time.time()
    if a.run == "strata":
        cmd = [a.strata, "--native", G] + a.strata_args.split() + ["--tokens", pid, "--force", fid, "--dump-full-logits", base + ".strata.bin"]
    elif a.run == "llama":
        cmd = [a.dump, "--model", G, "--tokens", pid, "--force", fid, "--dump-full-logits", base + ".llama.bin"] + a.llama_args.split()
    elif a.run == "cpu":
        cmd = [a.dump, "--model", G, "--tokens", pid, "--force", fid, "--dump-full-logits", base + ".cpu.bin"] + a.cpu_args.split()
    else: sys.exit("--run strata|llama|cpu")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    open(base + f".{a.run}.log", "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    print(f"{a.name} {a.run}: rc {r.returncode} in {time.time() - t0:.0f} s", flush=True)
    sys.exit(r.returncode)
def load(kind):
    f = base + f".{kind}.bin"
    if not os.path.exists(f): return None
    x = np.fromfile(f, dtype=np.float32)
    n = x.size // V
    return x[:n * V].reshape(n, V)
def logsm(x):
    x = x.astype(np.float64); m = x.max(axis=1, keepdims=True); return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))
L = load("llama"); S = load("strata"); C = load("cpu")
n = min(len(F), *(len(x) for x in (L, S, C) if x is not None))
tgt = np.array(F[:n])
def stats(ref, q, label):
    lr, lq = logsm(ref[:n]), logsm(q[:n]); pr = np.exp(lr)
    kl = (pr * (lr - lq)).sum(axis=1)
    top1 = (ref[:n].argmax(axis=1) == q[:n].argmax(axis=1)).mean()
    nll_r = -lr[np.arange(n), tgt]; nll_q = -lq[np.arange(n), tgt]
    acc_r = (ref[:n].argmax(axis=1) == tgt).mean(); acc_q = (q[:n].argmax(axis=1) == tgt).mean()
    print(f"{a.name} {label}: n={n} KL mean {kl.mean():.5f} median {np.median(kl):.5f} p99 {np.percentile(kl, 99):.4f} max {kl.max():.4f} | "
          f"top-1 agree {100 * top1:.1f}% | PPL ref {np.exp(nll_r.mean()):.3f} vs {np.exp(nll_q.mean()):.3f} | next-token acc ref {100 * acc_r:.1f}% vs {100 * acc_q:.1f}%")
if L is not None and S is not None: stats(L, S, "llama-gpu -> strata")
if L is not None and C is not None: stats(L, C, "llama-gpu -> llama-cpu (floor)")
if C is not None and S is not None: stats(C, S, "llama-cpu -> strata")
