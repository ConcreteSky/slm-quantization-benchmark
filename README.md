# SLM Quantization Benchmark

## Abstract

An empirical framework is provided for controlled comparison of FP16, LLM.int8, and four-bit Normal Float (NF4) inference on small causal language models. Classification quality, article language-model loss, generation throughput, and peak allocated GPU memory are measured under a shared protocol. Results are exported to CSV and recorded in MLflow. Accuracy–latency tradeoffs and sample predictions are displayed through Streamlit. No benchmark scores are bundled or implied by the implementation.

## Experimental setup

The default matrix is defined by `microsoft/Phi-3-mini-4k-instruct` and `google/gemma-2-2b`, three precision modes, and batch sizes of 1 and 4. A shuffled subset of 256 AG News test articles is selected with seed 42. Input lengths are capped at 512 model-specific tokens. Exactly 32 continuation tokens are generated per article, with two warmup batches and three complete timed repetitions. Greedy decoding, eager attention, FP16 arithmetic for unquantized layers, and disabled TF32 are held constant. NF4 weights are configured with double quantization and FP16 compute. Integer-eight weights are configured through LLM.int8 with its default outlier handling. These modes are supported by the [Transformers bitsandbytes integration](https://huggingface.co/docs/transformers/v4.56.2/quantization/bitsandbytes).

Only one CUDA GPU is used per run. CPU and disk offloading are excluded. Model loading and tokenization are excluded from timing. Each matrix cell is loaded independently; GPU memory is released between cells. A failed load, out-of-memory event, invalid configuration, or nonfinite metric is reported explicitly. Failed cells are retained with an error and absent quality metrics; a nonzero CLI exit status is returned if any cell fails. Infrastructure failures in MLflow or artifact storage terminate execution visibly.

Python 3.10–3.12 and an NVIDIA GPU with sufficient VRAM are required. Approximately 8 GB is occupied by the Phi-3 FP16 weights alone; additional memory is required for activations, logits, and generation caches. GPU capacity must be determined for the selected batch and sequence lengths. A 4 GB GPU cannot accommodate the complete default matrix. Model access must be enabled under the applicable Hugging Face licenses, including the gated Gemma repository. Credentials are supplied through `HF_TOKEN`, never through configuration or source files.

The Docker base is fixed to `nvidia/cuda:12.1.0-runtime-ubuntu22.04`. The pinned PyTorch 2.8 wheel supplies its own newer CUDA libraries; its effective CUDA version is recorded in each run. A host driver compatible with that wheel, plus NVIDIA Container Toolkit, is required. The base image tag must not be interpreted as the effective PyTorch CUDA version. Direct dependencies are pinned; exact installed package versions, Python version, GPU, driver, and effective CUDA version are also recorded. Transitive dependencies and the base image digest are not locked.

Model revisions are configurable and default to `main`. Resolved model commit hashes, dataset fingerprints, selected article-ID hashes, and configuration hashes are exported. For a repeatable study, immutable model commit hashes must be substituted for `main`, the dataset cache must be retained, and exported fingerprints must be checked. Equivalent tokenization is applied within each model's precision comparisons. Absolute perplexity values across different tokenizers must not be interpreted as a controlled quantization comparison. Statistical significance is not inferred from timing repetition alone; multi-seed studies must be executed separately.

## Dataset and classification protocol

[AG News](https://huggingface.co/datasets/fancyzhx/ag_news) is distributed with 120,000 training articles and 7,600 test articles across World, Sports, Business, and Sci/Tech. The test split is selected by default. Original row IDs are retained before seeded shuffling so the same examples can be compared across precision modes. The supported dataset aliases are `ag_news` and `fancyzhx/ag_news` in the loader; the CLI configuration uses `ag_news`.

`get_dataset(dataset_name, split, sample_limit)` returns a tokenized Hugging Face Dataset. An optional `tokenizer` argument is supplied by the CLI for the evaluated model; the Phi-3 tokenizer is loaded when that argument is omitted. Unpadded article token IDs, attention masks, classification prompt IDs, raw text, labels, and original sample IDs are returned. All rows are used when `sample_limit` is null; an oversized limit is capped at the split size.

A common plain-text prompt is constructed with the mappings A: World, B: Sports, C: Business, D: Sci/Tech. Article tokens are truncated to preserve the instruction and final category marker. Each candidate suffix (` A`, ` B`, ` C`, ` D`) is tokenized separately and appended to the prompt. Its conditional log probability is summed over all suffix tokens. The class with the highest score is selected; classification cross-entropy is computed by normalizing the four scores. No fine-tuning, demonstrations, model-specific chat template, or output-string parsing is applied. Candidate tokenization can differ between models, and this fixed prompting protocol is not a claim of optimal downstream accuracy.

## Metrics

For article tokens with valid next-token targets, the token-weighted negative log likelihood and perplexity are defined as

\[
\mathcal{L}_{LM}=-\frac{1}{N}\sum_{t=1}^{N}\log p_\theta(x_t\mid x_{<t}),
\qquad \mathrm{PPL}=\exp(\mathcal{L}_{LM}).
\]

The first token and padding positions are excluded from target counts. Each truncated article is scored independently, with no context carried across articles. This is truncated-article perplexity, rather than full-corpus sliding-window perplexity; context effects are discussed in the [Transformers perplexity documentation](https://huggingface.co/docs/transformers/v4.56.2/perplexity). Loss is weighted by valid target counts, rather than by batch means. Article language-model loss (`loss`) is distinct from four-class cross-entropy (`classification_loss`).

For class scores \(s_{j,c}\), predicted labels and classification metrics are defined as

\[
\hat y_j=\arg\max_c s_{j,c},\qquad
\mathrm{Accuracy}=\frac{1}{M}\sum_{j=1}^{M}\mathbf{1}[\hat y_j=y_j],
\qquad
\mathcal{L}_{cls}=-\frac{1}{M}\sum_{j=1}^{M}\log\frac{e^{s_{j,y_j}}}{\sum_c e^{s_{j,c}}}.
\]

For synchronized generation time \(T\), total generated tokens \(G\), article count \(M\), and repetition count \(R\), throughput and amortized latency are defined as

\[
\mathrm{Throughput}=G/T\quad\text{tokens/s},\qquad
\mathrm{Latency}_{sample}=1000T/(MR)\quad\text{ms/sample}.
\]

Prefill and autoregressive decoding are included in timing; tokenization, host-to-device input transfer, decoding to text, classification scoring, and perplexity scoring are excluded. CUDA is synchronized before and after each generation batch. A fixed generation budget is enforced, including beyond normal end-of-sequence predictions. Continuations are therefore controlled workloads, not conversational responses. Throughput standard deviation is calculated across complete repetition rates with population normalization. Batch mean, median, and 95th-percentile latencies are also exported. Amortized sample latency is not time to first token, inter-token latency, or individual request latency.

Peak allocated VRAM is measured with `torch.cuda.max_memory_allocated` after warmup and a peak-counter reset, covering quality evaluation and timed generation with model weights resident. It is reported in GiB. CUDA context, allocator reservations, external allocations, and other processes are excluded; this value is not total board utilization or model-weight size.

## Local execution

From the repository root, the environment is installed and the benchmark is executed as follows:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.benchmark_cli --config config.yaml
python -m src.benchmark_cli --config config.yaml --output-dir results
```

On Windows, activation is performed with `.venv\Scripts\Activate.ps1`. A CUDA-enabled PyTorch installation must be present. Configuration is validated before the matrix is executed. Precision names in YAML are `fp16`, `int8`, and `nf4`; `4bit` is additionally accepted by `QuantizationLoader` as an NF4 alias.

Each invocation produces `metrics_<invocation>.csv` and, when evaluation succeeds, `samples_<invocation>.csv` under the output directory. Exports are atomically replaced after each completed cell, and unique invocation names prevent overwriting prior studies. Previously completed cells remain available after interruption; automatic resumption is not implemented. All sampled predictions and first-repetition continuations are retained. Runtime CSV exports, MLflow storage, environment files, and caches are generated during execution and are not repository source files.

MLflow parameters include experimental settings and environment versions. Configuration, metric tables, and sample tables are recorded as run artifacts. The default tracking database is `sqlite:///mlflow.db`; a remote tracking URI can be configured. An `MLFLOW_TRACKING_URI` environment variable is given precedence over the YAML value. Tracking is inspected with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
streamlit run app/streamlit_app.py
```

A result directory or uploaded CSV pair is selected in the dashboard. Failed runs are displayed separately. Successful runs are filtered by available experimental settings. Lower latency and higher accuracy define the displayed Pareto frontier. Peak allocated VRAM is compared across models, batch sizes, and precision modes. Article-level class predictions and raw article continuations are shown for matching run IDs. Sample exports contain dataset text and should be handled under the dataset's terms.

## Docker execution

From a Linux shell with NVIDIA Container Toolkit configured, the image is built and the benchmark is launched as follows. `HF_TOKEN` is expected to have been supplied through the shell environment when gated model access is required.

```bash
docker build -t slm-quantization-benchmark .
mkdir -p results
docker run --rm --gpus all -e HF_TOKEN -e MLFLOW_TRACKING_URI=sqlite:////tracking/mlflow.db -v "$PWD/results:/benchmark/results" -v slm-hf-cache:/cache/huggingface -v slm-mlflow:/tracking -v slm-artifacts:/benchmark/mlruns slm-quantization-benchmark
```

The `/tracking` and `/benchmark/mlruns` volumes preserve the tracking database and run artifacts respectively. CSV results are exposed to the dashboard through the host results directory:

```bash
docker run --rm -p 8501:8501 -v "$PWD/results:/benchmark/results:ro" slm-quantization-benchmark python3.10 -m streamlit run app/streamlit_app.py --server.address=0.0.0.0
```

The dashboard is then available at `http://localhost:8501`. No GPU is required for CSV visualization. Model and dataset downloads require network access unless caches have already been populated.

## Repository scope

The source tree is limited to `README.md`, `requirements.txt`, `Dockerfile`, `config.yaml`, the `src` package (`__init__.py`, `data_loader.py`, `quantizer.py`, `evaluator.py`, `benchmark_cli.py`), and `app/streamlit_app.py`. Python source contains executable statements only, without comments or docstrings. Runtime results are not committed as empirical evidence unless the corresponding experiments have actually been executed and reviewed.
