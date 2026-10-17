#!/usr/bin/env python3
"""tools/gsm8k_eval.py - GSM8K through an OpenAI-compatible chat server (Strata or llama-server), thinking on: the
accuracy, the completion and reasoning lengths and the server's own decode speed, one JSON line per problem.

    python tools/gsm8k_eval.py --url http://127.0.0.1:8080 --key KEY --data test.jsonl --n 40 --out strata.jsonl
    python tools/gsm8k_eval.py --url http://127.0.0.1:8000 --model Qwen3.6-35B-A3B --data test.jsonl --n 40 --out llama.jsonl

test.jsonl is the GSM8K test set (openai/grade-school-math on GitHub: grade_school_math/data/test.jsonl, 1,319
problems; --start/--n pick a slice).  Sampling as Qwen recommends for thinking (temperature 0.6, top-p 0.95, top-k 20),
one seed per problem, so two servers get the same requests; the answer is the last \\boxed{} number (else the last
number).  The 2026-10-16 run of this fork and llama.cpp b11559 on the same GGUF is in AGENTS_FORK.md."""
import argparse, json, re, sys, time, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--key", default=""); ap.add_argument("--label", default=""); ap.add_argument("--model", default="x")
ap.add_argument("--data", required=True); ap.add_argument("--n", type=int, default=40); ap.add_argument("--start", type=int, default=0)
ap.add_argument("--max-tokens", type=int, default=6144); ap.add_argument("--temp", type=float, default=0.6)
ap.add_argument("--top-p", type=float, default=0.95); ap.add_argument("--top-k", type=int, default=20)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", required=True); ap.add_argument("--timeout", type=float, default=2400)
a = ap.parse_args()
probs = [json.loads(l) for l in open(a.data, encoding="utf-8")][a.start:a.start + a.n]
def gold_of(ans): return ans.split("####")[-1].strip().replace(",", "")
def num(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return m[-1] if m else None
def pred_of(text):
    boxes = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if boxes: return num(boxes[-1])
    m = re.search(r"[Aa]nswer[^\d\-]*(-?[\d,]+(?:\.\d+)?)", text)
    if m: return num(m.group(1))
    return num(text)
def same(p, g):
    if p is None: return False
    try: return abs(float(p) - float(g)) < 1e-6
    except ValueError: return p == g
out = open(a.out, "w", encoding="utf-8")
ok = 0; tot_tok = 0; tot_reason = 0; tps = []; trunc = 0; t_all = time.time()
for i, p in enumerate(probs):
    body = {"model": a.model, "messages": [{"role": "user", "content": p["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
            "max_tokens": a.max_tokens, "temperature": a.temp, "top_p": a.top_p, "top_k": a.top_k, "seed": a.seed + i,
            "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + a.key} if a.key else {})})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as r: d = json.loads(r.read())
    except Exception as e:
        print(f"{a.label} #{a.start + i}: request failed: {e}", flush=True); out.write(json.dumps({"i": a.start + i, "error": str(e)}) + "\n"); out.flush(); continue
    wall = time.time() - t0
    ch = d["choices"][0]; msg = ch["message"]; content = msg.get("content") or ""; reasoning = msg.get("reasoning_content") or ""
    fin = ch.get("finish_reason"); usage = d.get("usage") or {}; t = d.get("timings") or {}
    gold = gold_of(p["answer"]); pred = pred_of(content) if content.strip() else pred_of(reasoning)
    good = same(pred, gold); ok += good; trunc += fin == "length"
    ctok = usage.get("completion_tokens") or t.get("predicted_n") or 0; tot_tok += ctok; tot_reason += len(reasoning)
    if t.get("predicted_per_second"): tps.append(t["predicted_per_second"])
    rec = {"i": a.start + i, "gold": gold, "pred": pred, "correct": good, "finish": fin, "completion_tokens": ctok,
           "reasoning_chars": len(reasoning), "content_chars": len(content), "wall_s": round(wall, 1), "tok_s": t.get("predicted_per_second"),
           "prompt_tokens": usage.get("prompt_tokens")}
    out.write(json.dumps(rec) + "\n"); out.flush()
    print(f"{a.label} #{a.start + i}: {'ok ' if good else 'BAD'} pred={pred} gold={gold} {ctok} tok ({len(reasoning)} reasoning chars) {fin} {wall:.0f} s {t.get('predicted_per_second', 0):.1f} tok/s", flush=True)
n = len(probs)
print(f"{a.label} SUMMARY: {ok}/{n} correct ({100.0 * ok / n:.1f}%), truncated {trunc}, mean completion {tot_tok / n:.0f} tokens, "
      f"mean reasoning {tot_reason / n:.0f} chars, decode {sum(tps) / len(tps) if tps else 0:.1f} tok/s mean, total {(time.time() - t_all) / 60:.1f} min", flush=True)
