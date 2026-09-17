import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from sklearn.metrics import accuracy_score

from src.data_loader import CLASS_CODES, CLASS_NAMES


class BenchmarkEvaluator:
    def __init__(
        self,
        model,
        tokenizer,
        device=0,
        max_new_tokens=32,
        warmup_batches=2,
        repeats=3,
    ):
        for name, value in (
            ("max_new_tokens", max_new_tokens),
            ("warmup_batches", warmup_batches),
            ("repeats", repeats),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(f"cuda:{device}")
        self.max_new_tokens = max_new_tokens
        self.warmup_batches = warmup_batches
        self.repeats = repeats
        self.samples = []
        self.result = None

    def _pad(self, sequences, left=False):
        width = max(map(len, sequences))
        ids = torch.full(
            (len(sequences), width), self.tokenizer.pad_token_id, dtype=torch.long
        )
        mask = torch.zeros_like(ids)
        for index, sequence in enumerate(sequences):
            start = width - len(sequence) if left else 0
            ids[index, start : start + len(sequence)] = torch.tensor(sequence)
            mask[index, start : start + len(sequence)] = 1
        return {
            "input_ids": ids.to(self.device),
            "attention_mask": mask.to(self.device),
        }

    def _generate(self, inputs):
        return self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=False,
            output_scores=False,
            output_logits=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=None,
            forced_eos_token_id=None,
        )

    def _classification(self, prompts, targets):
        scores = []
        for code in CLASS_CODES:
            candidate = self.tokenizer.encode(code, add_special_tokens=False)
            if not candidate:
                raise ValueError("A class code produced no tokens")
            sequences = [prompt + candidate for prompt in prompts]
            inputs = self._pad(sequences)
            labels = inputs["input_ids"].clone()
            labels[inputs["attention_mask"] == 0] = -100
            for row, prompt in enumerate(prompts):
                labels[row, : len(prompt)] = -100
            logits = self.model(**inputs, use_cache=False).logits
            shifted = labels[:, 1:]
            selected = shifted != -100
            token_losses = functional.cross_entropy(
                logits[:, :-1][selected].float(),
                shifted[selected],
                reduction="none",
            )
            scores.append(-token_losses.reshape(len(prompts), -1).sum(dim=1))
        scores = torch.stack(scores, dim=1)
        target_tensor = torch.tensor(targets, device=self.device)
        loss = functional.cross_entropy(scores, target_tensor, reduction="sum")
        return scores.argmax(dim=1).tolist(), float(loss)

    @torch.inference_mode()
    def evaluate(self, dataset, batch_size):
        if type(batch_size) is not int or batch_size < 1 or len(dataset) == 0:
            raise ValueError("A positive batch size and nonempty dataset are required")
        self.result = None
        self.samples = []
        context = getattr(self.model.config, "max_position_embeddings", None)
        longest = max(len(ids) for ids in dataset["input_ids"])
        if context and longest + self.max_new_tokens > context:
            raise ValueError("Input length plus generation exceeds model context")
        warmup = dataset[: min(batch_size, len(dataset))]
        inputs = self._pad(warmup["input_ids"], left=True)
        for _ in range(self.warmup_batches):
            self._generate(inputs)
            self._classification(warmup["prompt_ids"], warmup["label"])
        del inputs
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        total_nll = 0.0
        total_tokens = 0
        classification_nll = 0.0
        predictions = []
        references = []
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start : start + batch_size]
            inputs = self._pad(batch["input_ids"])
            logits = self.model(**inputs, use_cache=False).logits
            valid = inputs["attention_mask"][:, 1:].bool()
            count = int(valid.sum())
            if count:
                loss = functional.cross_entropy(
                    logits[:, :-1][valid].float(),
                    inputs["input_ids"][:, 1:][valid],
                    reduction="sum",
                )
                total_nll += float(loss)
                total_tokens += count
            del logits, inputs
            predicted, loss = self._classification(batch["prompt_ids"], batch["label"])
            classification_nll += loss
            predictions.extend(predicted)
            references.extend(batch["label"])
            for index, prediction in enumerate(predicted):
                self.samples.append(
                    {
                        "sample_id": batch["sample_id"][index],
                        "text": batch["text"][index],
                        "reference": CLASS_NAMES[batch["label"][index]],
                        "prediction": CLASS_NAMES[prediction],
                        "continuation": "",
                    }
                )
        if total_tokens == 0:
            raise ValueError("No next-token targets are available")
        durations = []
        batch_latencies = []
        generated_tokens = 0
        for repeat in range(self.repeats):
            duration = 0.0
            for start in range(0, len(dataset), batch_size):
                batch = dataset[start : start + batch_size]
                inputs = self._pad(batch["input_ids"], left=True)
                torch.cuda.synchronize(self.device)
                begin = time.perf_counter()
                output = self._generate(inputs)
                torch.cuda.synchronize(self.device)
                elapsed = time.perf_counter() - begin
                duration += elapsed
                batch_latencies.append(elapsed * 1000)
                continuation = output[:, inputs["input_ids"].shape[1] :]
                if continuation.shape[1] != self.max_new_tokens:
                    raise RuntimeError(
                        "Generation stopped before the fixed token budget"
                    )
                generated_tokens += continuation.numel()
                if repeat == 0:
                    decoded = self.tokenizer.batch_decode(
                        continuation, skip_special_tokens=True
                    )
                    for index, text in enumerate(decoded):
                        self.samples[start + index]["continuation"] = text
                del inputs, output, continuation
            durations.append(duration)
        loss = total_nll / total_tokens
        perplexity = (
            math.exp(loss) if loss < math.log(np.finfo(float).max) else math.inf
        )
        if not all(
            math.isfinite(value) for value in (loss, perplexity, classification_nll)
        ):
            raise FloatingPointError("Nonfinite quality metrics were produced")
        total_seconds = sum(durations)
        per_repeat_tokens = generated_tokens / self.repeats
        self.result = {
            "accuracy": accuracy_score(references, predictions),
            "classification_loss": classification_nll / len(dataset),
            "loss": loss,
            "perplexity": perplexity,
            "perplexity_tokens": total_tokens,
            "sample_count": len(dataset),
            "generated_tokens": generated_tokens,
            "generation_seconds": total_seconds,
            "tokens_per_second": generated_tokens / total_seconds,
            "tokens_per_second_std": float(
                np.std([per_repeat_tokens / value for value in durations])
            ),
            "latency_ms_per_sample": total_seconds
            * 1000
            / (len(dataset) * self.repeats),
            "batch_latency_ms_mean": float(np.mean(batch_latencies)),
            "batch_latency_ms_p50": float(np.percentile(batch_latencies, 50)),
            "batch_latency_ms_p95": float(np.percentile(batch_latencies, 95)),
            "peak_vram_gib": torch.cuda.max_memory_allocated(self.device) / 1024**3,
        }
        return self.to_dataframe()

    def to_dataframe(self):
        if self.result is None:
            raise RuntimeError("Evaluation has not completed successfully")
        return pd.DataFrame([self.result])

    def samples_dataframe(self):
        if self.result is None:
            raise RuntimeError("Evaluation has not completed successfully")
        return pd.DataFrame(self.samples)
