"""Step-0 baseline eval for Moonlight-16B-A3B (Base / Instruct) on GSM8K test and MATH500.

Scoring reuses verl.utils.reward_score.default_compute_score so numbers match training reward.
GSM8K is additionally scored with a lenient last-number extractor to separate format errors from ability.

Usage:
    python eval_moonlight_step0.py --model PATH --tag NAME --prompt-mode {chat,raw} [--out-dir DIR]
"""

import argparse
import json
import os
import re
import time

import pandas as pd
from vllm import LLM, SamplingParams

from verl.utils.reward_score import default_compute_score

GSM8K_TEST = "/inspire/hdd/project/qianghuaxuexi/public/datasets/gsm8k/test.parquet"
MATH500_TEST = "/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence/data/math500/test.parquet"


def gsm8k_flexible(text, gt):
    """Last number in the response tail, ignoring commas/$ and trailing periods.

    verl's gsm8k "flexible" mode keeps the trailing period ("18." != "18"), which
    undercounts free-form answers, so we use this to measure ability independent of format.
    """
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text[-300:])
    if not nums:
        return 0.0
    pred = nums[-1].replace(",", "").rstrip(".")
    try:
        return float(abs(float(pred) - float(gt.replace(",", ""))) < 1e-6)
    except ValueError:
        return 0.0


def build_prompts(df, tokenizer, prompt_mode):
    prompts = []
    for msgs in df["prompt"]:
        msgs = [dict(m) for m in msgs]
        if prompt_mode == "chat":
            prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        else:
            prompts.append(f"Question: {msgs[-1]['content']}\nAnswer:")
    return prompts


def evaluate(llm, tokenizer, name, path, n, args):
    df = pd.read_parquet(path)
    prompts = build_prompts(df, tokenizer, args.prompt_mode)
    stop = ["\n\nQuestion:"] if args.prompt_mode == "raw" else None
    params = SamplingParams(
        n=n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop=stop,
        seed=args.seed,
    )
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    gen_time = time.time() - t0

    records = []
    stats = dict(strict=0.0, flexible=0.0, fmt=0, trunc=0, length=0, total=0)
    for row, out in zip(df.itertuples(), outputs):
        gt = row.reward_model["ground_truth"]
        for comp in out.outputs:
            text = comp.text
            res = default_compute_score(row.data_source, text, gt)
            strict = float(res["score"] if isinstance(res, dict) else res)
            if name == "gsm8k":
                flexible = gsm8k_flexible(text, gt)
                has_fmt = re.search(r"#### ?\-?[0-9\.\,]+", text) is not None
            else:
                flexible = strict
                has_fmt = "\\boxed" in text
            truncated = comp.finish_reason == "length"
            stats["strict"] += strict
            stats["flexible"] += flexible
            stats["fmt"] += int(has_fmt)
            stats["trunc"] += int(truncated)
            stats["length"] += len(comp.token_ids)
            stats["total"] += 1
            if len(records) < 50:
                records.append(dict(gt=gt, strict=strict, flexible=flexible, truncated=truncated, response=text))

    total = max(stats["total"], 1)
    summary = {
        "dataset": name,
        "n_prompts": len(df),
        "n_samples_per_prompt": n,
        f"acc_strict_avg@{n}": round(100 * stats["strict"] / total, 2),
        f"acc_flexible_avg@{n}": round(100 * stats["flexible"] / total, 2),
        "format_rate": round(100 * stats["fmt"] / total, 2),
        "truncation_rate": round(100 * stats["trunc"] / total, 2),
        "mean_response_tokens": round(stats["length"] / total, 1),
        "gen_time_s": round(gen_time, 1),
    }
    return summary, records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--prompt-mode", choices=["chat", "raw"], default="chat")
    p.add_argument("--out-dir", default="outputs/moonlight_step0")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--gsm8k-n", type=int, default=4)
    p.add_argument("--math500-n", type=int, default=4)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()
    if args.prompt_mode == "chat" and getattr(tokenizer, "chat_template", None) is None:
        raise SystemExit(f"{args.model} has no chat_template; rerun with --prompt-mode raw")

    os.makedirs(args.out_dir, exist_ok=True)
    results = {"model": args.model, "tag": args.tag, "prompt_mode": args.prompt_mode, "sampling": vars(args)}
    for name, path, n in [("gsm8k", GSM8K_TEST, args.gsm8k_n), ("math500", MATH500_TEST, args.math500_n)]:
        summary, records = evaluate(llm, tokenizer, name, path, n, args)
        results[name] = summary
        print(json.dumps(summary, ensure_ascii=False))
        with open(os.path.join(args.out_dir, f"{args.tag}_{args.prompt_mode}_{name}_samples.jsonl"), "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(args.out_dir, f"{args.tag}_{args.prompt_mode}_summary.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
