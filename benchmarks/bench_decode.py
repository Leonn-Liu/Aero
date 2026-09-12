import argparse
import time
from typing import List
import torch
from aero.models.hf_runner import HuggingFaceRunner
from aero.runtime.engine import SpeculativeEngine

DEFAULT_PROMPTS = [
    "Artificial intelligence and machine learning algorithms have transformed modern technology by",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The primary cause of climate change is the emission of greenhouse gases, which leads to"
]

def run_suite(
    engine: SpeculativeEngine,
    prompts: List[str],
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    gamma: int,
    adaptive: bool
):
    total_tokens = 0
    total_drafted = 0
    total_accepted = 0
    total_time = 0.0

    engine.adaptive_gamma = adaptive

    for p in prompts:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, tokens, _, drafted, accepted = engine.generate_with_stats(
            prompt=p,
            max_new_tokens=max_tokens,
            gamma=gamma,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        torch.cuda.synchronize()
        total_time += (time.perf_counter() - t0)
        total_tokens += len(tokens)
        total_drafted += drafted
        total_accepted += accepted

    speed = total_tokens / max(1e-5, total_time)
    acc_rate = total_accepted / max(1, total_drafted)
    return speed, acc_rate

def run_baseline(
    engine: SpeculativeEngine,
    prompts: List[str],
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float
):
    total_tokens = 0
    total_time = 0.0

    for p in prompts:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, tokens = engine.generate_pure_target(
            prompt=p,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        torch.cuda.synchronize()
        total_time += (time.perf_counter() - t0)
        total_tokens += len(tokens)

    return total_tokens / max(1e-5, total_time)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--target", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tokens", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return

    print("=" * 80)
    print(f"AERO BENCHMARK SUITE: Draft=[{args.draft}] Target=[{args.target}]")
    print(f"Device: {torch.cuda.get_device_name(0)} | Evaluated on {len(DEFAULT_PROMPTS)} Diverse Prompts")
    print("=" * 80)

    draft_runner = HuggingFaceRunner(model_name_or_path=args.draft)
    target_runner = HuggingFaceRunner(model_name_or_path=args.target)

    engine = SpeculativeEngine(
        draft_runner=draft_runner,
        target_runner=target_runner,
        adaptive_gamma=False,
        initial_gamma=4
    )

    for mode_name, temp in [("Greedy (T=0.0)", 0.0), ("Stochastic Fused CUDA (T=0.8, TopK=50, TopP=0.9)", 0.8)]:
        print(f"\n>>> MODE: {mode_name}")
        print(f"{'Configuration':<28} | {'Tokens/s':<10} | {'Speedup':<10} | {'Acceptance':<10}")
        print("-" * 66)

        base_speed = run_baseline(
            engine=engine,
            prompts=DEFAULT_PROMPTS,
            max_tokens=args.tokens,
            temperature=temp,
            top_k=50,
            top_p=0.9
        )
        print(f"{'Pure Target Baseline':<28} | {base_speed:<10.2f} | {'1.00x':<10} | {'N/A':<10}")

        for g in [2, 4, 6]:
            s_speed, acc = run_suite(
                engine=engine,
                prompts=DEFAULT_PROMPTS,
                max_tokens=args.tokens,
                temperature=temp,
                top_k=50,
                top_p=0.9,
                gamma=g,
                adaptive=False
            )
            sp = s_speed / base_speed
            print(f"{f'Speculative (gamma={g})':<28} | {s_speed:<10.2f} | {f'{sp:.2f}x':<10} | {f'{acc*100:.1f}%':<10}")

        adapt_speed, adapt_acc = run_suite(
            engine=engine,
            prompts=DEFAULT_PROMPTS,
            max_tokens=args.tokens,
            temperature=temp,
            top_k=50,
            top_p=0.9,
            gamma=4,
            adaptive=True
        )
        asp = adapt_speed / base_speed
        print(f"{'Speculative (Adaptive)':<28} | {adapt_speed:<10.2f} | {f'{asp:.2f}x':<10} | {f'{adapt_acc*100:.1f}%':<10}")
        print("-" * 66)

    print("\nBenchmark suite completed successfully.")

if __name__ == "__main__":
    main()
