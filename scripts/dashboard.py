from __future__ import annotations

import json
import queue
import re
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import discover_dataset
from src.utils import load_config

st.set_page_config(
    page_title="Fuzzy Fusion | Research Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    [data-testid="stSidebar"] { background: #101b32; }
    [data-testid="stSidebar"] * { color: #eef3ff; }
    .hero {
        padding: 1.5rem 1.8rem; border-radius: 18px; color: white;
        background: linear-gradient(120deg, #15294b, #255b78 65%, #498f91);
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { margin: .45rem 0 0; opacity: .86; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e6eaf1;
        padding: 14px 16px; border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

config = load_config(ROOT / "config.yaml")
configured_data = Path(config["data"]["root"])
if not configured_data.is_absolute():
    configured_data = ROOT / configured_data
uploaded_data = ROOT / "kvasir-dataset"
default_data = uploaded_data if uploaded_data.is_dir() else configured_data

with st.sidebar:
    st.markdown("## 🧪 Fuzzy Fusion")
    st.caption("Colorectal image classification research dashboard")
    data_text = st.text_input("Dataset folder on this computer", str(default_data))
    st.caption("Use a folder containing one subfolder per image class.")
    st.divider()
    st.markdown("**Experiment setup**")
    st.write(f"Extractors: {len(config['features']['extractors'])}")
    st.write(f"Classifier: {config['classifier']['type']}")
    st.write(f"Device: {config['features']['device']}")
    st.caption("The dataset stays on your computer; nothing is uploaded.")

data_root = Path(data_text).expanduser()
if not data_root.is_absolute():
    data_root = ROOT / data_root

samples = []
class_names: list[str] = []
dataset_error = None
try:
    samples, class_names = discover_dataset(data_root)
except (FileNotFoundError, ValueError) as exc:
    dataset_error = str(exc)

st.markdown(
    """
    <div class="hero">
      <h1>Fuzzy-Guided Dynamic Feature Fusion</h1>
      <p>Inspect your dataset, run the experiment, and explore model performance in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if dataset_error:
    st.error(f"Dataset not ready: {dataset_error}")
    st.info("Choose a valid dataset folder in the sidebar. It must contain image files grouped into class folders.")
else:
    label_counts = Counter(sample.label for sample in samples)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Images", f"{len(samples):,}")
    col2.metric("Classes", len(class_names))
    col3.metric("Images per class", f"{min(label_counts.values())}–{max(label_counts.values())}")
    col4.metric("Feature extractors", len(config["features"]["extractors"]))

    st.markdown("### Dataset overview")
    overview_col, run_col = st.columns([1.3, 1])
    with overview_col:
        counts_df = pd.DataFrame(
            [{"Class": name, "Images": label_counts[name]} for name in class_names]
        )
        st.dataframe(counts_df, use_container_width=True, hide_index=True)
        st.caption(f"Dataset location: `{data_root.resolve()}`")

    with run_col:
        st.markdown("#### Run the full experiment")
        st.write(
            "The pipeline computes image-quality scores, extracts backbone features, "
            "compares fusion methods, and saves reports and plots."
        )
        if st.button("▶  Run experiment", type="primary", use_container_width=True):
            progress = st.progress(0, text="Starting experiment…")
            status = st.empty()
            log_box = st.empty()
            log_lines: list[str] = []
            command = [
                sys.executable,
                "-u",
                str(ROOT / "scripts" / "run_experiment.py"),
                "--config",
                str(ROOT / "config.yaml"),
                "--data",
                str(data_root.resolve()),
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
            chunks: queue.Queue[bytes | None] = queue.Queue()

            def read_process_output() -> None:
                assert process.stdout is not None
                try:
                    while True:
                        chunk = process.stdout.read(1024)
                        if not chunk:
                            break
                        chunks.put(chunk)
                finally:
                    chunks.put(None)

            reader = threading.Thread(target=read_process_output, daemon=True)
            reader.start()
            pending = ""
            stage = "Starting experiment…"
            fraction = 0.0
            completed_classifiers = 0
            log_box_placeholder = st.container()
            stream_finished = False

            while not stream_finished or reader.is_alive():
                try:
                    chunk = chunks.get(timeout=0.2)
                except queue.Empty:
                    chunk = b""
                if chunk is None:
                    stream_finished = True
                elif chunk:
                    pending += chunk.decode("utf-8", errors="replace")
                    segments = re.split(r"[\r\n]", pending)
                    pending = segments.pop()
                    for segment in segments:
                        line = segment.strip()
                        if not line:
                            continue
                        quality_match = re.search(
                            r"Computing image-quality indicators:\s*(\d+)%.*?(\d+)/(\d+)",
                            line,
                        )
                        feature_match = re.search(
                            r"Extracting deep features:\s*(\d+)%.*?(\d+)/(\d+)",
                            line,
                        )
                        if quality_match:
                            fraction = 0.2 * int(quality_match.group(1)) / 100
                            stage = f"Computing image-quality indicators: {quality_match.group(2)}/{quality_match.group(3)}"
                        elif feature_match:
                            fraction = 0.2 + 0.6 * int(feature_match.group(1)) / 100
                            stage = f"Extracting deep features: {feature_match.group(2)}/{feature_match.group(3)}"
                        elif "Loaded cached features" in line:
                            fraction = max(fraction, 0.8)
                            stage = "Loaded cached image features"
                            log_lines.append(line)
                        elif line.startswith("Training classifier for:"):
                            completed_classifiers += 1
                            fraction = 0.8 + 0.18 * completed_classifiers / 5
                            stage = line
                            log_lines.append(line)
                        elif line.startswith(("Found ", "Using device:", "Saved results to")):
                            log_lines.append(line)
                            stage = line
                        if len(log_lines) > 12:
                            log_lines = log_lines[-12:]
                progress.progress(min(fraction, 0.99), text=stage)
                status.info(stage)
                with log_box_placeholder:
                    if log_lines:
                        st.code("\n".join(log_lines), language="text")

            reader.join()
            return_code = process.wait()
            if pending.strip():
                log_lines.append(pending.strip())
            if return_code == 0:
                progress.progress(1.0, text="Experiment complete")
                st.success("Experiment completed successfully. Results below have been refreshed.")
                st.rerun()
            else:
                progress.empty()
                st.error(f"Experiment failed with exit code {return_code}. Review the output below.")
                if log_lines:
                    st.code("\n".join(log_lines[-20:]), language="text")

    output_dir = Path(config["output"]["dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    summary_path = output_dir / "metrics_summary.csv"
    metrics_path = output_dir / "metrics.json"
    quality_path = output_dir / "per_image_quality_and_weights.csv"

    st.divider()
    st.markdown("### Experiment results")
    if not summary_path.exists():
        st.info("Run the experiment to see the model comparison, plots, and per-image analysis here.")
    else:
        summary_df = pd.read_csv(summary_path)
        metrics = {}
        if metrics_path.exists():
            with metrics_path.open(encoding="utf-8") as handle:
                metrics = json.load(handle)
        st.caption(f"Latest saved results: `{summary_path}`")
        if metrics:
            reported_images = metrics.get("n_images")
            image_count = f"{reported_images:,}" if isinstance(reported_images, int) else "unknown"
            st.caption(
                f"Report contains {image_count} images and "
                f"{len(metrics.get('class_names', []))} classes."
            )

        best_row = summary_df.sort_values("test_f1_macro", ascending=False).iloc[0]
        fuzzy_row = summary_df.loc[summary_df["method"] == "fuzzy_dynamic"]
        fuzzy_f1 = float(fuzzy_row.iloc[0]["test_f1_macro"]) if not fuzzy_row.empty else None
        metric_cols = st.columns(4)
        metric_cols[0].metric("Best method", str(best_row["method"]))
        metric_cols[1].metric("Best test accuracy", f"{best_row['test_accuracy']:.1%}")
        metric_cols[2].metric("Best test macro F1", f"{best_row['test_f1_macro']:.1%}")
        if fuzzy_f1 is not None:
            metric_cols[3].metric("Fuzzy dynamic macro F1", f"{fuzzy_f1:.1%}")

        tab_compare, tab_plots, tab_images = st.tabs(
            ["📊 Model comparison", "📈 Plots", "🖼️ Per-image analysis"]
        )
        with tab_compare:
            st.dataframe(
                summary_df.style.format(
                    {
                        "test_accuracy": "{:.2%}",
                        "test_precision_macro": "{:.2%}",
                        "test_recall_macro": "{:.2%}",
                        "test_f1_macro": "{:.2%}",
                        "val_accuracy": "{:.2%}",
                        "val_f1_macro": "{:.2%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download comparison CSV",
                summary_path.read_bytes(),
                file_name="metrics_summary.csv",
                mime="text/csv",
            )

        with tab_plots:
            plots_dir = output_dir / "plots"
            plot_files = sorted(plots_dir.glob("*.png")) if plots_dir.is_dir() else []
            if plot_files:
                for start in range(0, len(plot_files), 2):
                    cols = st.columns(2)
                    for col, image_path in zip(cols, plot_files[start : start + 2]):
                        col.image(str(image_path), caption=image_path.stem.replace("_", " ").title())
            else:
                st.info("No plots are available yet. Run the experiment to generate them.")

        with tab_images:
            if quality_path.exists():
                quality_df = pd.read_csv(quality_path)
                class_filter = st.selectbox("Filter by class", ["All classes", *class_names])
                visible_df = quality_df
                if class_filter != "All classes":
                    visible_df = visible_df[visible_df["label"] == class_filter]
                st.dataframe(visible_df.head(500), use_container_width=True, hide_index=True)
                st.caption(
                    f"Showing up to 500 of {len(visible_df):,} matching image rows. "
                    "Columns include quality indicators and each extractor's fuzzy weight."
                )
                st.download_button(
                    "Download per-image CSV",
                    quality_path.read_bytes(),
                    file_name="per_image_quality_and_weights.csv",
                    mime="text/csv",
                )
            else:
                st.info("Per-image quality data appears after a completed experiment.")
