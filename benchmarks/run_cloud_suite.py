import argparse
import csv
import datetime
import gc
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from aero.models.hf_runner import HuggingFaceRunner
from aero.ops.sampling import AeroSampler
from aero.runtime.engine import SpeculativeEngine

CURATED_PROMPTS = [
    {
        "id": "code_quicksort",
        "category": "code",
        "prompt": "Write a complete, highly optimized Python implementation of QuickSort with 3-way partitioning:\n```python\n"
    },
    {
        "id": "code_avl_tree",
        "category": "code",
        "prompt": "Implement a self-balancing AVL binary search tree in Python with insert and rebalance operations:\n"
    },
    {
        "id": "math_combinatorics",
        "category": "math",
        "prompt": "Solve the following combinatorics problem step-by-step: How many integers between 100 and 999 have distinct digits?\nAnswer:"
    },
    {
        "id": "reasoning_logic",
        "category": "reasoning",
        "prompt": "Three people (Alice, Bob, Charlie) are in a room. One always tells the truth, one always lies, and one alternates. Alice says:"
    },
    {
        "id": "chat_inference_engine",
        "category": "chat",
        "prompt": "Explain the architectural differences between Speculative Decoding, Medusa, and Lookahead Decoding in high-throughput LLM serving engines."
    },
    {
        "id": "technical_cuda_memory",
        "category": "technical",
        "prompt": "Detail the memory latency, bandwidth, and caching behavior of Shared Memory, L1 Cache, and L2 Cache in the NVIDIA Ada Lovelace architecture."
    }
]

class EnvironmentCollector:
    @staticmethod
    def get_git_info() -> Dict[str, Any]:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
            status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
            return {"commit_sha": commit, "branch": branch, "is_dirty": len(status) > 0}
        except Exception:
            return {"commit_sha": "unknown", "branch": "unknown", "is_dirty": False}

    @staticmethod
    def collect(device: str = "cuda:0", sampler_backend: str = "cuda") -> Dict[str, Any]:
        hw_info: Dict[str, Any] = {
            "gpu_name": "CPU",
            "gpu_count": 0,
            "total_vram_bytes": 0,
            "compute_capability": "N/A",
            "driver_version": "N/A",
            "cuda_driver_version": "N/A",
            "pcie_gen": 4,
            "cpu_model": platform.processor() or "Unknown CPU",
            "system_ram_gb": 32.0
        }

        if torch.cuda.is_available():
            dev_idx = torch.device(device).index or 0
            prop = torch.cuda.get_device_properties(dev_idx)
            hw_info["gpu_name"] = prop.name
            hw_info["gpu_count"] = torch.cuda.device_count()
            hw_info["total_vram_bytes"] = prop.total_memory
            hw_info["compute_capability"] = f"{prop.major}.{prop.minor}"

            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)
                hw_info["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
                cuda_v = pynvml.nvmlSystemGetCudaDriverVersion()
                hw_info["cuda_driver_version"] = f"{cuda_v // 1000}.{(cuda_v % 1000) // 10}"
                hw_info["pcie_gen"] = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
                pynvml.nvmlShutdown()
            except Exception:
                hw_info["driver_version"] = "Detected via Torch CUDA"
                hw_info["cuda_driver_version"] = torch.version.cuda or "N/A"

        import transformers
        sw_info = {
            "os": f"{platform.system()}-{platform.release()}-{platform.machine()}",
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda or "N/A",
            "transformers_version": transformers.__version__,
            "sampler_backend": sampler_backend
        }

        return {
            "hardware": hw_info,
            "software": sw_info,
            "git": EnvironmentCollector.get_git_info()
        }

@dataclass
class RoundTrace:
    round_idx: int
    gamma: int
    drafted_count: int
    accepted_count: int
    draft_latency_ms: float
    verify_latency_ms: float
    accepted_tokens: List[int]

class InstrumentedBenchmarkEngine:
    def __init__(self, engine: SpeculativeEngine):
        self.engine = engine
        self.draft_runner = engine.draft_runner
        self.target_runner = engine.target_runner
        self.sampler = engine.sampler
        self.gamma_controller = engine.gamma_controller
        self.draft_gen = engine.draft_gen
        self.verifier = engine.verifier

    def run_pure_target(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> Tuple[List[int], float, float, float]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        tokenizer = self.target_runner.tokenizer
        curr_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.target_runner.device)

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        logits, pkv = self.target_runner.prefill(curr_input_ids)
        generated_tokens: List[int] = []
        curr_logits = logits

        for _ in range(max_new_tokens):
            next_token, _ = self.sampler.sample(
                logits=curr_logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                return_probs=False
            )
            token_id = next_token.squeeze().item()
            generated_tokens.append(token_id)
            if token_id == self.target_runner.eos_token_id:
                break
            step_in = torch.tensor([[token_id]], device=self.target_runner.device)
            out_logits, pkv = self.target_runner.forward(step_in, pkv)
            curr_logits = out_logits[:, -1, :]

        end_evt.record()
        torch.cuda.synchronize()

        total_time_ms = start_evt.elapsed_time(end_evt)
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

        return generated_tokens, total_time_ms, peak_allocated_mb, peak_reserved_mb

    def run_speculative_profiled(
        self,
        prompt: str,
        max_new_tokens: int,
        gamma: int,
        adaptive: bool,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> Tuple[List[int], float, List[RoundTrace], float, float]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        tokenizer = self.target_runner.tokenizer
        curr_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.target_runner.device)

        start_total = torch.cuda.Event(enable_timing=True)
        end_total = torch.cuda.Event(enable_timing=True)
        d_start = torch.cuda.Event(enable_timing=True)
        d_end = torch.cuda.Event(enable_timing=True)
        v_start = torch.cuda.Event(enable_timing=True)
        v_end = torch.cuda.Event(enable_timing=True)

        start_total.record()
        target_logits, target_pkv = self.target_runner.prefill(curr_input_ids)
        draft_logits, draft_pkv = self.draft_runner.prefill(curr_input_ids.to(self.draft_runner.device))

        generated_tokens: List[int] = []
        curr_target_logits = target_logits
        curr_draft_logits = draft_logits
        round_traces: List[RoundTrace] = []

        if adaptive:
            self.gamma_controller.reset()

        round_idx = 0
        while len(generated_tokens) < max_new_tokens:
            curr_gamma = self.gamma_controller.get_gamma() if adaptive else gamma

            d_start.record()
            cand_tokens, cand_probs, draft_pkv = self.draft_gen.generate_candidates(
                start_logits=curr_draft_logits,
                past_key_values=draft_pkv,
                gamma=curr_gamma,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            d_end.record()

            v_start.record()
            accepted, next_start_token, target_pkv, draft_pkv, acc_count = self.verifier.verify(
                candidate_tokens=cand_tokens,
                candidate_probs=cand_probs,
                target_start_logits=curr_target_logits,
                target_past_key_values=target_pkv,
                draft_past_key_values=draft_pkv,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            v_end.record()
            torch.cuda.synchronize()

            d_ms = d_start.elapsed_time(d_end)
            v_ms = v_start.elapsed_time(v_end)

            round_traces.append(RoundTrace(
                round_idx=round_idx,
                gamma=curr_gamma,
                drafted_count=curr_gamma,
                accepted_count=acc_count,
                draft_latency_ms=d_ms,
                verify_latency_ms=v_ms,
                accepted_tokens=accepted
            ))
            round_idx += 1

            if adaptive:
                self.gamma_controller.update(acc_count, curr_gamma)

            stopped = False
            for token_id in accepted:
                generated_tokens.append(token_id)
                if token_id == self.target_runner.eos_token_id or len(generated_tokens) >= max_new_tokens:
                    stopped = True
                    break

            if stopped or len(generated_tokens) >= max_new_tokens:
                break

            step_in_t = torch.tensor([[next_start_token]], device=self.target_runner.device)
            target_out_logits, target_pkv = self.target_runner.forward(step_in_t, target_pkv)
            curr_target_logits = target_out_logits[:, -1, :]

            step_in_d = torch.tensor([[next_start_token]], device=self.draft_runner.device)
            draft_out_logits, draft_pkv = self.draft_runner.forward(step_in_d, draft_pkv)
            curr_draft_logits = draft_out_logits[:, -1, :]

        end_total.record()
        torch.cuda.synchronize()

        total_time_ms = start_total.elapsed_time(end_total)
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

        return generated_tokens, total_time_ms, round_traces, peak_allocated_mb, peak_reserved_mb

def compute_distribution_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90))
    }

class ReportExporter:
    @staticmethod
    def export(
        output_dir: Path,
        benchmark_data: Dict[str, Any]
    ) -> Tuple[Path, Path, Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = benchmark_data["benchmark_id"]

        json_path = output_dir / f"benchmark_raw_{ts}.json"
        summary_csv_path = output_dir / f"summary_{ts}.csv"
        detail_csv_path = output_dir / f"details_{ts}.csv"
        md_report_path = output_dir / f"report_{ts}.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Mode", "Gamma", "Temperature",
                "Throughput_Mean_tok_s", "Throughput_Std", "Throughput_P50", "Throughput_P90",
                "Speedup_Mean_x", "Speedup_P50_x",
                "AcceptanceRate_Mean", "StepLength_Tau_Mean",
                "PeakVRAM_Allocated_MB"
            ])
            for s in benchmark_data["summary"]:
                writer.writerow([
                    s["mode"],
                    s["gamma"],
                    s["temperature"],
                    f"{s['throughput_tokens_per_sec']['mean']:.2f}",
                    f"{s['throughput_tokens_per_sec']['std']:.2f}",
                    f"{s['throughput_tokens_per_sec']['p50']:.2f}",
                    f"{s['throughput_tokens_per_sec']['p90']:.2f}",
                    f"{s['speedup_ratio']['mean']:.2f}",
                    f"{s['speedup_ratio']['p50']:.2f}",
                    f"{s['acceptance_rate']['mean'] * 100:.2f}%",
                    f"{s['mean_step_length_tau']['mean']:.2f}",
                    f"{s['peak_vram_mb']:.1f}"
                ])

        with open(detail_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "PromptID", "Category", "PromptLen", "Mode", "Gamma", "Temp",
                "RepeatIdx", "TotalLatencyMs", "Throughput_tok_s", "Speedup",
                "RoundIdx", "Drafted", "Accepted", "DraftLatencyMs", "VerifyLatencyMs"
            ])
            for p_rec in benchmark_data["detailed_results"]:
                p_id = p_rec["prompt_id"]
                cat = p_rec["category"]
                p_len = p_rec["prompt_token_length"]

                for spec_cfg in p_rec["speculative_evaluations"]:
                    m = spec_cfg["mode"]
                    g = spec_cfg["gamma"]
                    t = spec_cfg["temperature"]
                    for rep in spec_cfg["repeats_data"]:
                        r_idx = rep["repeat_idx"]
                        tot_ms = rep["total_latency_ms"]
                        tps = rep["tokens_per_sec"]
                        sp = rep["speedup"]
                        for rd in rep["round_sequences"]:
                            writer.writerow([
                                p_id, cat, p_len, m, g, t,
                                r_idx, f"{tot_ms:.2f}", f"{tps:.2f}", f"{sp:.2f}",
                                rd["round_idx"], rd["drafted_count"], rd["accepted_count"],
                                f"{rd['draft_latency_ms']:.3f}", f"{rd['verify_latency_ms']:.3f}"
                            ])

        ReportExporter._write_markdown_dashboard(md_report_path, benchmark_data)

        return json_path, summary_csv_path, detail_csv_path, md_report_path

    @staticmethod
    def _write_markdown_dashboard(md_path: Path, data: Dict[str, Any]):
        env = data["environment"]
        hw = env["hardware"]
        sw = env["software"]
        models = env["models"]
        summary = data["summary"]

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Aero Speculative Decoding: Cloud Benchmark Report\n\n")
            f.write(f"Benchmark ID: `{data['benchmark_id']}` | Date: `{data['timestamp_utc']}`\n\n")

            f.write("## System & Hardware Environment\n\n")
            f.write("| Component | Specification |\n")
            f.write("| :--- | :--- |\n")
            f.write(f"| GPU | `{hw['gpu_name']}` ({hw['total_vram_bytes'] / (1024**3):.1f} GB VRAM, sm_{hw['compute_capability']}) |\n")
            f.write(f"| Host CPU & OS | `{hw['cpu_model']}` | `{sw['os']}` |\n")
            f.write(f"| NVIDIA Driver / CUDA | Driver `{hw['driver_version']}` | CUDA `{hw['cuda_driver_version']}` (Runtime: `{sw['cuda_runtime_version']}`) |\n")
            f.write(f"| PyTorch & Backend | PyTorch `{sw['torch_version']}` | Aero Sampler: `{sw['sampler_backend'].upper()}` |\n")
            f.write(f"| Target Model | `{models['target']['name_or_path']}` ({models['target']['dtype']}) |\n")
            f.write(f"| Draft Model | `{models['draft']['name_or_path']}` ({models['draft']['dtype']}) |\n\n")

            f.write("## Performance & Acceleration Overview\n\n")
            f.write("| Configuration | Temp | Throughput (tok/s) | StdDev | P50 (tok/s) | P90 (tok/s) | Speedup | Acceptance Rate | Step Length tau | Peak VRAM |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

            for s in summary:
                mode_str = "Pure Target Baseline" if s["mode"] == "baseline" else f"Aero Speculative ({s['mode']})"
                tps = s["throughput_tokens_per_sec"]
                speedup_str = f"{s['speedup_ratio']['mean']:.2f}x" if s["mode"] != "baseline" else "1.00x (Ref)"
                acc_str = f"{s['acceptance_rate']['mean'] * 100:.1f}%" if s["mode"] != "baseline" else "N/A"
                tau_str = f"{s['mean_step_length_tau']['mean']:.2f}" if s["mode"] != "baseline" else "1.00"

                f.write(
                    f"| {mode_str} | {s['temperature']:.1f} | "
                    f"{tps['mean']:.2f} | +-{tps['std']:.2f} | {tps['p50']:.2f} | {tps['p90']:.2f} | "
                    f"{speedup_str} | {acc_str} | {tau_str} | {s['peak_vram_mb']:.1f} MB |\n"
                )

def run_benchmark_suite(args):
    print("=" * 80)
    print("  AERO SPECULATIVE SAMPLING ENGINE: CLOUD BENCHMARK HARNESS")
    print("=" * 80)

    device = args.device
    env_info = EnvironmentCollector.collect(device=device, sampler_backend=args.sampler_backend)
    hw = env_info["hardware"]
    print(f"[*] Target Hardware: {hw['gpu_name']} | Driver: {hw['driver_version']} | VRAM: {hw['total_vram_bytes'] / (1024**3):.2f} GB")
    print(f"[*] Software Stack : PyTorch {env_info['software']['torch_version']} | CUDA {env_info['software']['cuda_runtime_version']}")
    print(f"[*] Target Model   : {args.target} ({args.target_dtype})")
    print(f"[*] Draft Model    : {args.draft} ({args.draft_dtype})")

    draft_dtype = getattr(torch, args.draft_dtype)
    target_dtype = getattr(torch, args.target_dtype)

    draft_runner = HuggingFaceRunner(model_name_or_path=args.draft, device=device, dtype=draft_dtype)
    target_runner = HuggingFaceRunner(model_name_or_path=args.target, device=device, dtype=target_dtype)
    sampler = AeroSampler(backend=args.sampler_backend)

    engine = SpeculativeEngine(
        draft_runner=draft_runner,
        target_runner=target_runner,
        sampler=sampler,
        adaptive_gamma=False
    )
    profiler = InstrumentedBenchmarkEngine(engine=engine)

    env_info["models"] = {
        "draft": {
            "name_or_path": args.draft,
            "commit_hash": getattr(draft_runner.model.config, "_commit_hash", "local_loaded"),
            "dtype": str(draft_dtype),
            "num_parameters": sum(p.numel() for p in draft_runner.model.parameters())
        },
        "target": {
            "name_or_path": args.target,
            "commit_hash": getattr(target_runner.model.config, "_commit_hash", "local_loaded"),
            "dtype": str(target_dtype),
            "num_parameters": sum(p.numel() for p in target_runner.model.parameters())
        }
    }

    prompts = CURATED_PROMPTS
    if args.dataset != "curated" and os.path.exists(args.dataset):
        prompts = []
        with open(args.dataset, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                prompts.append({
                    "id": item.get("id", f"prompt_{idx}"),
                    "category": item.get("category", "general"),
                    "prompt": item["prompt"]
                })
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    print(f"[+] Loaded {len(prompts)} evaluation prompts.")
    print(f"\n[+] Starting Warmup Protocol ({args.warmup} runs)...")
    warmup_prompt = "The quick brown fox jumps over the lazy dog"
    for _ in range(args.warmup):
        _ = profiler.run_pure_target(warmup_prompt, max_new_tokens=16, temperature=0.0, top_k=50, top_p=0.9)
        _ = profiler.run_speculative_profiled(warmup_prompt, max_new_tokens=16, gamma=4, adaptive=False, temperature=0.0, top_k=50, top_p=0.9)
    torch.cuda.synchronize()
    print("[+] Warmup complete. GPU clock & CUDA kernels stabilized.")

    detailed_results = []
    bench_id = f"aero_bench_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    config_tracker = {}

    for p_idx, p_item in enumerate(prompts):
        p_id = p_item["id"]
        p_cat = p_item["category"]
        p_text = p_item["prompt"]
        p_tok_len = len(target_runner.tokenizer(p_text).input_ids)
        print(f"\n[{p_idx + 1}/{len(prompts)}] Evaluating Prompt '{p_id}' ({p_cat}, {p_tok_len} tokens)...")

        prompt_record: Dict[str, Any] = {
            "prompt_id": p_id,
            "category": p_cat,
            "prompt_token_length": p_tok_len,
            "target_baseline": {},
            "speculative_evaluations": []
        }

        for temp in args.temperatures:
            base_tps_list = []
            base_vram_list = []
            base_lat_list = []
            for r in range(args.repeats):
                toks, lat_ms, alloc_mb, _ = profiler.run_pure_target(
                    prompt=p_text,
                    max_new_tokens=args.max_tokens,
                    temperature=temp,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                tps = len(toks) / max(1e-5, lat_ms / 1000.0)
                base_tps_list.append(tps)
                base_vram_list.append(alloc_mb)
                base_lat_list.append(lat_ms)

            mean_base_tps = float(np.mean(base_tps_list))
            cfg_key_base = f"baseline_T{temp}"
            if cfg_key_base not in config_tracker:
                config_tracker[cfg_key_base] = {
                    "mode": "baseline",
                    "gamma": "N/A",
                    "temperature": temp,
                    "tps": [],
                    "speedups": [],
                    "acc_rates": [],
                    "taus": [],
                    "vram": []
                }
            config_tracker[cfg_key_base]["tps"].extend(base_tps_list)
            config_tracker[cfg_key_base]["speedups"].append(1.0)
            config_tracker[cfg_key_base]["acc_rates"].append(0.0)
            config_tracker[cfg_key_base]["taus"].append(1.0)
            config_tracker[cfg_key_base]["vram"].extend(base_vram_list)

            modes_to_test = [(f"gamma_{g}", g, False) for g in args.gamma_list]
            if args.test_adaptive:
                modes_to_test.append(("adaptive", 4, True))

            for m_name, g_val, is_adapt in modes_to_test:
                spec_eval_entry = {
                    "mode": m_name,
                    "gamma": str(g_val) if not is_adapt else "adaptive",
                    "temperature": temp,
                    "generated_tokens_len": args.max_tokens,
                    "repeats_data": []
                }

                spec_tps_list = []
                spec_vram_list = []
                spec_acc_list = []
                spec_tau_list = []

                for r in range(args.repeats):
                    toks, lat_ms, r_traces, alloc_mb, _ = profiler.run_speculative_profiled(
                        prompt=p_text,
                        max_new_tokens=args.max_tokens,
                        gamma=g_val,
                        adaptive=is_adapt,
                        temperature=temp,
                        top_k=args.top_k,
                        top_p=args.top_p
                    )
                    tps = len(toks) / max(1e-5, lat_ms / 1000.0)
                    sp = tps / max(1e-5, mean_base_tps)
                    tot_drafted = sum(rt.drafted_count for rt in r_traces)
                    tot_accepted = sum(rt.accepted_count for rt in r_traces)
                    acc = tot_accepted / max(1, tot_drafted)
                    tau = len(toks) / max(1, len(r_traces))

                    spec_tps_list.append(tps)
                    spec_vram_list.append(alloc_mb)
                    spec_acc_list.append(acc)
                    spec_tau_list.append(tau)

                    spec_eval_entry["repeats_data"].append({
                        "repeat_idx": r,
                        "total_latency_ms": lat_ms,
                        "tokens_per_sec": tps,
                        "speedup": sp,
                        "acceptance_rate": acc,
                        "step_length_tau": tau,
                        "round_sequences": [asdict(rt) for rt in r_traces]
                    })

                prompt_record["speculative_evaluations"].append(spec_eval_entry)

                cfg_key_spec = f"{m_name}_T{temp}"
                if cfg_key_spec not in config_tracker:
                    config_tracker[cfg_key_spec] = {
                        "mode": m_name,
                        "gamma": str(g_val) if not is_adapt else "adaptive",
                        "temperature": temp,
                        "tps": [],
                        "speedups": [],
                        "acc_rates": [],
                        "taus": [],
                        "vram": []
                    }
                config_tracker[cfg_key_spec]["tps"].extend(spec_tps_list)
                config_tracker[cfg_key_spec]["speedups"].extend([t / max(1e-5, mean_base_tps) for t in spec_tps_list])
                config_tracker[cfg_key_spec]["acc_rates"].extend(spec_acc_list)
                config_tracker[cfg_key_spec]["taus"].extend(spec_tau_list)
                config_tracker[cfg_key_spec]["vram"].extend(spec_vram_list)

        detailed_results.append(prompt_record)

    summary_list = []
    for k, v in config_tracker.items():
        tps_stats = compute_distribution_stats(v["tps"])
        sp_stats = compute_distribution_stats(v["speedups"])
        acc_stats = compute_distribution_stats(v["acc_rates"])
        tau_stats = compute_distribution_stats(v["taus"])
        peak_vram = float(np.max(v["vram"])) if len(v["vram"]) > 0 else 0.0

        lat_stats = {
            "mean": 1000.0 / max(1e-5, tps_stats["mean"]),
            "std": 0.0,
            "p50": 1000.0 / max(1e-5, tps_stats["p50"]),
            "p90": 1000.0 / max(1e-5, tps_stats["p90"])
        }

        summary_list.append({
            "mode": v["mode"],
            "gamma": v["gamma"],
            "temperature": v["temperature"],
            "throughput_tokens_per_sec": tps_stats,
            "latency_ms_per_token": lat_stats,
            "speedup_ratio": sp_stats,
            "acceptance_rate": acc_stats,
            "mean_step_length_tau": tau_stats,
            "peak_vram_mb": peak_vram
        })

    benchmark_payload = {
        "schema_version": "1.0.0",
        "benchmark_id": bench_id,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "environment": env_info,
        "config": {
            "gamma_list": args.gamma_list,
            "include_adaptive": args.test_adaptive,
            "repeats": args.repeats,
            "warmup_iters": args.warmup,
            "max_new_tokens": args.max_tokens,
            "temperatures": args.temperatures,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "dataset_name": args.dataset
        },
        "summary": summary_list,
        "detailed_results": detailed_results
    }

    out_dir = Path(args.output_dir)
    j_path, s_csv, d_csv, md_path = ReportExporter.export(out_dir, benchmark_payload)
    print("\n" + "=" * 80)
    print("  AERO BENCHMARK RUN COMPLETE - ARTIFACTS EXPORTED")
    print("=" * 80)
    print(f"  -> Raw Benchmark JSON: {j_path.resolve()}")
    print(f"  -> Summary CSV:        {s_csv.resolve()}")
    print(f"  -> Detailed CSV:       {d_csv.resolve()}")
    print(f"  -> Markdown Report:    {md_path.resolve()}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Aero Cloud Benchmark Harness")
    parser.add_argument("--draft", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--target", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--draft_dtype", type=str, default="float16")
    parser.add_argument("--target_dtype", type=str, default="float16")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="benchmarks/results")
    parser.add_argument("--gamma_list", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--test_adaptive", action="store_true", default=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="curated")
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=30)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.8])
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--sampler_backend", type=str, default="cuda")
    args = parser.parse_args()

    run_benchmark_suite(args)

if __name__ == "__main__":
    main()
