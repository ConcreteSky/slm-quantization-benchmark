from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


def pareto_frontier(frame):
    values = frame[["latency_ms_per_sample", "accuracy"]].to_numpy()
    keep = []
    for latency, accuracy in values:
        dominated = (
            (values[:, 0] <= latency)
            & (values[:, 1] >= accuracy)
            & ((values[:, 0] < latency) | (values[:, 1] > accuracy))
        ).any()
        keep.append(not dominated)
    return frame.loc[keep].sort_values("latency_ms_per_sample")


def main():
    st.set_page_config(page_title="SLM Quantization Benchmark", layout="wide")
    st.title("SLM Quantization Benchmark")
    metrics_upload = st.sidebar.file_uploader("Metrics CSV", type="csv")
    samples_upload = st.sidebar.file_uploader("Samples CSV", type="csv")
    directory = Path(st.sidebar.text_input("Results directory", "results"))
    candidates = sorted(directory.glob("metrics_*.csv")) if directory.is_dir() else []
    if metrics_upload is not None:
        source = metrics_upload
    elif candidates:
        source = Path(
            st.sidebar.selectbox("Metrics export", [str(p) for p in candidates])
        )
    else:
        st.info(
            "Upload a metrics CSV or select a directory containing benchmark exports."
        )
        return
    try:
        frame = pd.read_csv(source)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Metrics could not be loaded: {error}")
        return
    required = {"model", "precision", "batch_size", "dataset", "run_id", "status"}
    if not required.issubset(frame.columns):
        st.error(
            f"Required columns are missing: {sorted(required - set(frame.columns))}"
        )
        return
    failed = frame.loc[frame["status"] != "ok"]
    if not failed.empty:
        with st.expander(f"Failed runs ({len(failed)})"):
            st.dataframe(failed, hide_index=True)
    frame = frame.loc[frame["status"] == "ok"].copy()
    if frame.empty:
        st.warning("No successful benchmark results are available.")
        return
    numeric = [
        "accuracy",
        "latency_ms_per_sample",
        "tokens_per_second",
        "peak_vram_gib",
    ]
    if not set(numeric).issubset(frame.columns):
        st.error(
            "Successful rows require accuracy, latency, throughput, and VRAM metrics."
        )
        return
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = np.isfinite(frame[numeric]).all(axis=1)
    valid &= frame["accuracy"].between(0, 1)
    valid &= (frame[numeric[1:]] > 0).all(axis=1)
    if not valid.all():
        st.warning("Rows containing invalid numeric metrics were excluded.")
    frame = frame.loc[valid]
    if frame.empty:
        st.warning("No valid metrics remain.")
        return
    for column in (
        "invocation",
        "dataset",
        "split",
        "seed",
        "max_length",
        "max_new_tokens",
        "gpu",
    ):
        if column in frame and frame[column].nunique() > 1:
            selection = st.sidebar.selectbox(column, frame[column].dropna().unique())
            frame = frame.loc[frame[column] == selection]
    selected_models = st.sidebar.multiselect(
        "Models", frame["model"].unique(), default=frame["model"].unique()
    )
    frame = frame.loc[frame["model"].isin(selected_models)]
    if frame.empty:
        st.info("Select at least one model.")
        return
    st.caption(
        "Latency is amortized generation time per sample, including prefill. "
        "Throughput counts generated tokens. VRAM is peak PyTorch allocated memory."
    )
    frontier = pareto_frontier(frame)
    left, right = st.columns(2)
    with left:
        st.subheader("Accuracy versus latency")
        figure, axes = plt.subplots()
        sns.scatterplot(
            data=frame,
            x="latency_ms_per_sample",
            y="accuracy",
            hue="precision",
            style="model",
            size="batch_size",
            ax=axes,
        )
        axes.plot(
            frontier["latency_ms_per_sample"],
            frontier["accuracy"],
            "k--",
            label="Pareto frontier",
        )
        axes.set(
            xlabel="Amortized generation latency (ms/sample)",
            ylabel="Classification accuracy",
        )
        axes.legend(fontsize="small")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
    with right:
        st.subheader("Peak allocated VRAM")
        chart = frame.assign(
            configuration=frame["model"] + " / b" + frame["batch_size"].astype(str)
        )
        figure, axes = plt.subplots()
        sns.barplot(
            data=chart,
            x="configuration",
            y="peak_vram_gib",
            hue="precision",
            errorbar=None,
            ax=axes,
        )
        axes.set(xlabel="Model / batch size", ylabel="Peak allocated VRAM (GiB)")
        axes.tick_params(axis="x", rotation=35)
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
    st.dataframe(frame, hide_index=True)
    st.subheader("Sample inference comparisons")
    if samples_upload is not None:
        sample_source = samples_upload
    elif isinstance(source, Path):
        sample_source = source.with_name(source.name.replace("metrics_", "samples_", 1))
        if not sample_source.is_file():
            st.info("A matching samples CSV is unavailable.")
            return
    else:
        st.info(
            "Upload the matching samples CSV to inspect predictions and continuations."
        )
        return
    try:
        samples = pd.read_csv(sample_source)
        needed = {
            "run_id",
            "sample_id",
            "text",
            "reference",
            "prediction",
            "continuation",
            "model",
            "precision",
            "batch_size",
        }
        if not needed.issubset(samples.columns):
            raise ValueError(
                f"Missing sample columns: {sorted(needed - set(samples.columns))}"
            )
        samples = samples.loc[samples["run_id"].isin(frame["run_id"])]
        if samples.empty:
            st.info("No samples match the selected runs.")
            return
        sample_id = st.selectbox("Article ID", sorted(samples["sample_id"].unique()))
        selected = samples.loc[samples["sample_id"] == sample_id]
        st.write(selected.iloc[0]["text"])
        st.dataframe(
            selected[
                [
                    "model",
                    "precision",
                    "batch_size",
                    "reference",
                    "prediction",
                    "continuation",
                ]
            ],
            hide_index=True,
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Samples could not be loaded: {error}")


if __name__ == "__main__":
    main()
