import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from uuid import uuid4

import mlflow
import pandas as pd
import torch
import yaml
from transformers import set_seed

from src.data_loader import get_dataset
from src.evaluator import BenchmarkEvaluator
from src.quantizer import QuantizationLoader


def read_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    required = {
        "models",
        "precisions",
        "datasets",
        "batch_sizes",
        "split",
        "sample_limit",
        "seed",
        "max_length",
        "max_new_tokens",
        "warmup_batches",
        "repeats",
        "device",
        "output_dir",
        "mlflow_tracking_uri",
        "mlflow_experiment",
    }
    if set(config) != required:
        raise ValueError(f"Configuration keys must be exactly: {sorted(required)}")
    for key in ("models", "precisions", "datasets", "batch_sizes"):
        if not isinstance(config[key], list) or not config[key]:
            raise ValueError(f"{key} must be a nonempty list")
    for model in config["models"]:
        if not isinstance(model, dict) or set(model) != {"id", "revision"}:
            raise ValueError("Each model requires id and revision")
        if not all(
            isinstance(value, str) and value.strip() for value in model.values()
        ):
            raise ValueError("Model identifiers and revisions must be nonempty strings")
    if any(mode not in {"fp16", "int8", "nf4"} for mode in config["precisions"]):
        raise ValueError("precisions must contain fp16, int8, or nf4")
    if any(name != "ag_news" for name in config["datasets"]):
        raise ValueError("datasets must contain ag_news")
    for key in ("max_length", "max_new_tokens", "warmup_batches", "repeats"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_length"] < 64:
        raise ValueError("max_length must be at least 64")
    for key in ("seed", "device"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if any(type(size) is not int or size < 1 for size in config["batch_sizes"]):
        raise ValueError("batch_sizes must contain positive integers")
    limit = config["sample_limit"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("sample_limit must be positive or null")
    for key in ("split", "output_dir", "mlflow_tracking_uri", "mlflow_experiment"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    for key in ("models", "precisions", "datasets", "batch_sizes"):
        items = [json.dumps(item, sort_keys=True) for item in config[key]]
        if len(set(items)) != len(items):
            raise ValueError(f"Duplicate entries in {key}")
    return config


def export_csv(frame, path):
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def environment():
    packages = (
        "torch",
        "transformers",
        "bitsandbytes",
        "accelerate",
        "datasets",
        "pynvml",
        "scikit-learn",
        "pandas",
        "seaborn",
        "matplotlib",
        "mlflow",
        "streamlit",
        "pyyaml",
    )
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_cuda": str(torch.version.cuda),
    }
    for package in packages:
        result[f"version_{package}"] = importlib.metadata.version(package)
    try:
        result["driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        result["driver"] = "unavailable"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Single-GPU SLM quantization benchmark"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    try:
        config = read_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    if not torch.cuda.is_available():
        parser.error(
            "A CUDA GPU is required; CPU fallback would invalidate comparisons"
        )
    device = config["device"]
    if device >= torch.cuda.device_count():
        parser.error("The configured CUDA device is unavailable")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output_dir = Path(args.output_dir or config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    invocation = uuid4().hex
    results_path = output_dir / f"metrics_{invocation}.csv"
    samples_path = output_dir / f"samples_{invocation}.csv"
    config_digest = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    metadata = environment()
    metadata.update(
        gpu=torch.cuda.get_device_name(device),
        gpu_total_gib=torch.cuda.get_device_properties(device).total_memory / 1024**3,
        config_sha256=config_digest,
        invocation=invocation,
    )
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", config["mlflow_tracking_uri"])
    )
    mlflow.set_experiment(config["mlflow_experiment"])
    rows = []
    sample_frames = []
    failures = 0
    for specification in config["models"]:
        for precision in config["precisions"]:
            for dataset_name in config["datasets"]:
                for batch_size in config["batch_sizes"]:
                    model = tokenizer = evaluator = dataset = None
                    set_seed(config["seed"])
                    gc.collect()
                    torch.cuda.empty_cache()
                    row = {
                        "invocation": invocation,
                        "model": specification["id"],
                        "revision": specification["revision"],
                        "precision": precision,
                        "dataset": dataset_name,
                        "split": config["split"],
                        "batch_size": batch_size,
                        "seed": config["seed"],
                        "max_length": config["max_length"],
                        "max_new_tokens": config["max_new_tokens"],
                        "repeats": config["repeats"],
                        "warmup_batches": config["warmup_batches"],
                        **metadata,
                    }
                    with mlflow.start_run(
                        run_name=f"{specification['id']}-{precision}-b{batch_size}"
                    ) as run:
                        row["run_id"] = run.info.run_id
                        mlflow.log_params(row)
                        mlflow.log_dict(config, "config.json")
                        try:
                            model, tokenizer = QuantizationLoader(
                                specification["id"],
                                precision,
                                device,
                                specification["revision"],
                            ).load()
                            dataset = get_dataset(
                                dataset_name,
                                config["split"],
                                config["sample_limit"],
                                tokenizer=tokenizer,
                                max_length=config["max_length"],
                                seed=config["seed"],
                            )
                            row["resolved_revision"] = (
                                getattr(model.config, "_commit_hash", None)
                                or specification["revision"]
                            )
                            row["dataset_fingerprint"] = dataset._fingerprint
                            row["sample_ids_sha256"] = hashlib.sha256(
                                json.dumps(list(dataset["sample_id"])).encode()
                            ).hexdigest()
                            mlflow.log_params(
                                {
                                    key: row[key]
                                    for key in (
                                        "resolved_revision",
                                        "dataset_fingerprint",
                                        "sample_ids_sha256",
                                    )
                                }
                            )
                            evaluator = BenchmarkEvaluator(
                                model,
                                tokenizer,
                                device,
                                config["max_new_tokens"],
                                config["warmup_batches"],
                                config["repeats"],
                            )
                            metrics = (
                                evaluator.evaluate(dataset, batch_size)
                                .iloc[0]
                                .to_dict()
                            )
                            mlflow.log_metrics(
                                {
                                    key: float(value)
                                    for key, value in metrics.items()
                                    if math.isfinite(float(value))
                                }
                            )
                            row.update(metrics, status="ok", error="")
                            samples = evaluator.samples_dataframe().assign(
                                run_id=row["run_id"],
                                invocation=invocation,
                                model=row["model"],
                                precision=precision,
                                batch_size=batch_size,
                                dataset=dataset_name,
                            )
                            sample_frames.append(samples)
                        except Exception as error:
                            failures += 1
                            row.update(
                                status="failed",
                                error=f"{type(error).__name__}: {error}",
                            )
                            mlflow.set_tag("failure", row["error"][:5000])
                            mlflow.end_run(status="FAILED")
                        finally:
                            evaluator = model = tokenizer = dataset = None
                            gc.collect()
                            torch.cuda.empty_cache()
                    rows.append(row)
                    export_csv(pd.DataFrame(rows), results_path)
                    if sample_frames:
                        export_csv(
                            pd.concat(sample_frames, ignore_index=True), samples_path
                        )
                    with mlflow.start_run(run_id=row["run_id"]):
                        mlflow.log_table(pd.DataFrame([row]), "metrics.json")
                        if row["status"] == "ok":
                            mlflow.log_table(sample_frames[-1], "samples.json")
                        else:
                            mlflow.end_run(status="FAILED")
                    print(
                        pd.DataFrame([row])
                        .reindex(
                            columns=[
                                "model",
                                "precision",
                                "batch_size",
                                "status",
                                "accuracy",
                                "perplexity",
                                "tokens_per_second",
                                "peak_vram_gib",
                                "error",
                            ]
                        )
                        .to_string(index=False),
                        flush=True,
                    )
    print(f"Metrics: {results_path}", flush=True)
    if sample_frames:
        print(f"Samples: {samples_path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
