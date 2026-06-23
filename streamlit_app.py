"""
streamlit_app.py
================
Interactive Streamlit UI for the pure-numpy Eigenfaces (PCA/SVD) library.

Run from inside the eigenfaces/ folder:
    streamlit run streamlit_app.py

The app imports the shared `eigenfaces_core` module (same directory) and
exposes 6 tabs: Dataset, Training, Visualisasi, Identifikasi, Verifikasi,
Evaluasi. All heavy work is done by eigenfaces_core; this file is purely
presentation + caching + memory hygiene.

Pure PCA / SVD only - NO face_recognition / sklearn.decomposition / dlib /
MTCNN / Facenet / opencv_face. Allowed: numpy, matplotlib, PIL/cv2 I/O,
psutil (optional).
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import os
import sys
import gc
import time
import tempfile
import zipfile
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import streamlit as st

# Make sure we can import the core lib (same folder)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eigenfaces_core as ec


# --------------------------------------------------------------------------- #
# Page config + custom CSS
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Eigenfaces PCA/SVD",
    page_icon=":material/face_retouching_natural:",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --bg:        #0f172a;
    --panel:     #1e293b;
    --panel-2:   #243046;
    --card:      #1b2740;
    --accent:    #14b8a6;   /* teal */
    --accent-2:  #f59e0b;   /* amber */
    --danger:    #ef4444;
    --ok:        #22c55e;
    --text:      #e2e8f0;
    --muted:     #94a3b8;
    --border:    #334155;
}

html, body, [class*="css"] {
    color: var(--text);
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

/* App background gradient */
.stApp {
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(20,184,166,0.10), transparent 60%),
        radial-gradient(1200px 600px at 100% 0%, rgba(245,158,11,0.08), transparent 55%),
        var(--bg);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1322 0%, #111c30 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown h1 {
    font-size: 1.35rem;
    color: var(--accent);
    margin-bottom: 0;
    font-weight: 700;
    letter-spacing: -0.02em;
}
section[data-testid="stSidebar"] .stMarkdown p.subtitle {
    color: var(--muted);
    font-size: 0.78rem;
    margin-top: 0;
    margin-bottom: 1rem;
}

/* Cards */
.ef-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.30);
    margin-bottom: 16px;
}
.ef-card h3 {
    margin-top: 0;
    color: var(--accent);
    font-size: 1.05rem;
    letter-spacing: -0.01em;
}
.ef-card h4 {
    margin-top: 0;
    color: var(--accent-2);
    font-size: 0.95rem;
}
.ef-muted { color: var(--muted); font-size: 0.85rem; }
.ef-mono { font-family: 'JetBrains Mono','Menlo','Consolas',monospace; font-size: 0.82rem; }

/* Metric tiles */
.ef-metric {
    background: linear-gradient(135deg, rgba(20,184,166,0.10), rgba(20,184,166,0.02));
    border: 1px solid rgba(20,184,166,0.25);
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
}
.ef-metric .val { font-size: 1.8rem; font-weight: 800; color: var(--accent); }
.ef-metric .lbl { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }

/* Verdict badges */
.ef-badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}
.ef-badge.same { background: rgba(34,197,94,0.15); color: var(--ok); border: 1px solid rgba(34,197,94,0.45); }
.ef-badge.diff { background: rgba(239,68,68,0.15); color: var(--danger); border: 1px solid rgba(239,68,68,0.45); }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #0d9488 100%);
    color: #0b1322;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    padding: 8px 18px;
    transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(20,184,166,0.35);
    color: #0b1322;
}
.stButton > button:disabled {
    background: #334155;
    color: #64748b;
    cursor: not-allowed;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: var(--panel);
    border-radius: 10px 10px 0 0;
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 8px 18px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(20,184,166,0.25), rgba(20,184,166,0.05)) !important;
    color: var(--accent) !important;
    border-color: var(--accent) !important;
}
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent); }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* Streamlit subheaders */
h1, h2, h3 { color: var(--text) !important; letter-spacing: -0.02em; }
h2 { border-bottom: 1px solid var(--border); padding-bottom: 6px; }

/* Sticky footer */
.ef-footer {
    position: fixed;
    left: 0; right: 0; bottom: 0;
    background: rgba(11,19,34,0.92);
    backdrop-filter: blur(8px);
    border-top: 1px solid var(--border);
    padding: 8px 22px;
    font-size: 0.78rem;
    color: var(--muted);
    z-index: 9999;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.ef-footer .brand { color: var(--accent); font-weight: 700; }
.blockchain-spacer { height: 48px; }

/* File uploader polish */
.stFileUploader > section {
    background: var(--panel);
    border: 1px dashed var(--border);
    border-radius: 12px;
    padding: 12px;
}

/* Dataframe tweaks */
.stDataFrame, .stTable { background: var(--panel); border-radius: 10px; }
</style>

<div class="ef-footer">
    <span><span class="brand">Eigenfaces — PCA/SVD</span> &nbsp;|&nbsp;
          pure numpy &nbsp;|&nbsp; Turk-Pentland dual trick &nbsp;|&nbsp;
          backend: <span class="ef-mono">{backend}</span> &nbsp;|&nbsp;
          psutil: <span class="ef-mono">{psutil}</span></span>
    <span class="ef-mono">RAM: <span id="ram">{ram}</span> MB</span>
</div>
<div class="blockchain-spacer"></div>
"""
# Render CSS once at the top — placeholders filled per-rerun via safe replace.
_css_filled = (CUSTOM_CSS
               .replace("{backend}", ec.get_backend())
               .replace("{psutil}", "ON" if ec.has_psutil() else "OFF")
               .replace("{ram}", f"{ec.mem_used_mb():.1f}" if ec.has_psutil() else "—"))
st.markdown(_css_filled, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
IMAGE_SIZE_OPTIONS = {
    "112x92 (AT&T)": (112, 92),
    "64x64": (64, 64),
    "100x100": (100, 100),
}

RAM_CEILING_MB = 4096.0  # 4 GB


def _parse_image_size(label: str) -> Tuple[int, int]:
    return IMAGE_SIZE_OPTIONS[label]


def _close_fig(fig) -> None:
    """Close matplotlib figure and aggressively collect RAM if heavy."""
    try:
        plt.close(fig)
    except Exception:
        pass
    if ec.has_psutil() and ec.mem_used_mb() > 500.0:
        gc.collect()


def _maybe_gc() -> None:
    if ec.has_psutil() and ec.mem_used_mb() > 500.0:
        ec.gc_collect()


def _load_image_safe(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path)
    except Exception:
        return None


def _card(title: str, body_md: str) -> None:
    """Render a styled card."""
    st.markdown(
        f"""
        <div class="ef-card">
            <h3>{title}</h3>
            {body_md}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_tile(value: str, label: str) -> str:
    return f"""
    <div class="ef-metric">
        <div class="val">{value}</div>
        <div class="lbl">{label}</div>
    </div>"""


# --------------------------------------------------------------------------- #
# Cached heavy ops
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Memuat dataset...")
def cached_load_dataset(folder: str, image_size_tuple: Tuple[int, int]):
    """Cache dataset load keyed by (path, image_size). Returns (X, y, names, paths)."""
    X, y, label_names, paths = ec.load_dataset(
        folder,
        image_size=image_size_tuple,
        recursive=True,
        dtype=np.float32,
    )
    return X, y, label_names, paths


@st.cache_resource(
    show_spinner="Melatih Eigenfaces (PCA/SVD)...",
    hash_funcs={np.ndarray: lambda arr: id(arr)},
)
def cached_fit_eigenfaces(
    X: np.ndarray,
    variance_threshold: float,
    whiten: bool,
    image_size_tuple: Tuple[int, int],
    n_components_override: int,
):
    """Cache fit keyed by id(X) + variance + whiten + image_size + n_components."""
    if n_components_override and int(n_components_override) > 0:
        model = ec.fit_eigenfaces(
            X,
            n_components=int(n_components_override),
            image_size=image_size_tuple,
            whiten=bool(whiten),
            dtype=np.float32,
        )
    else:
        model = ec.fit_eigenfaces(
            X,
            variance_threshold=float(variance_threshold),
            image_size=image_size_tuple,
            whiten=bool(whiten),
            dtype=np.float32,
        )
    return model


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> dict:
    """Render sidebar; return dict of params."""
    with st.sidebar:
        st.markdown("# Eigenfaces — PCA/SVD")
        st.markdown('<p class="subtitle">Pure numpy implementation, memory-safe</p>',
                    unsafe_allow_html=True)

        st.markdown("### Parameter")
        image_size_label = st.selectbox(
            "Ukuran gambar",
            list(IMAGE_SIZE_OPTIONS.keys()),
            index=0,
            help="AT&T/ORL standar 112x92. Pilih sesuai dataset.",
        )
        image_size_tuple = _parse_image_size(image_size_label)

        variance_threshold = st.slider(
            "Variance threshold (k auto)",
            min_value=0.50, max_value=0.99, step=0.01, value=0.95,
            help="Pilih k komponen terkecil yang menjelaskan ≥ threshold varians.",
        )
        whiten = st.checkbox("Whiten (ZCA-style scaling)", value=False,
                             help="Skala eigenvector dengan 1/√λ. Rekonstruksi tetap benar (whiten_scale disimpan terpisah).")
        metric = st.radio(
            "Metric",
            options=["cosine", "euclidean"],
            index=0,
            horizontal=True,
            help="Cosine: similaritas tinggi = same. Euclidean: jarak rendah = same.",
        )
        factor = st.slider(
            "Threshold factor (auto-tune)",
            min_value=0.0, max_value=1.0, step=0.05, value=0.5,
            help="faktor interpolasi intra vs inter-class untuk auto-threshold.",
        )
        n_components_override = st.number_input(
            "Override k komponen (0 = auto)",
            min_value=0, max_value=2000, value=0, step=1,
            help="Jika > 0, gunakan jumlah komponen ini (abaikan variance threshold).",
        )

        st.markdown("---")
        st.markdown("### Memory")
        ram_mb = ec.mem_used_mb() if ec.has_psutil() else -1.0
        if ram_mb >= 0:
            pct = min(ram_mb / RAM_CEILING_MB, 1.0)
            st.progress(pct, text=f"RAM: {ram_mb:.1f} / {RAM_CEILING_MB:.0f} MB")
        else:
            st.info("psutil tidak tersedia — monitoring RAM off.")

        st.markdown("---")
        if st.button("Reset Session", use_container_width=True,
                     help="Bersihkan dataset & model dari session_state + cache."):
            # Clear all caches
            try:
                cached_load_dataset.clear()
                cached_fit_eigenfaces.clear()
            except Exception:
                pass
            for k in list(st.session_state.keys()):
                try:
                    del st.session_state[k]
                except Exception:
                    pass
            st.success("Session direset.")
            st.rerun()

    return dict(
        image_size_label=image_size_label,
        image_size_tuple=image_size_tuple,
        variance_threshold=variance_threshold,
        whiten=whiten,
        metric=metric,
        factor=factor,
        n_components_override=int(n_components_override),
        ram_mb=ram_mb,
    )


# --------------------------------------------------------------------------- #
# Tab 1 — Dataset
# --------------------------------------------------------------------------- #
def tab_dataset(params: dict) -> None:
    st.header("1. Dataset")

    st.markdown(
        '<div class="ef-card"><h3>Sumber dataset</h3>'
        '<p class="ef-muted">Unggah ZIP berisi subfolder per identitas, atau '
        'masukkan path folder lokal. Struktur yang didukung: '
        '<span class="ef-mono">root/s1/*.pgm</span>, '
        '<span class="ef-mono">root/identity_name/*.jpg</span>, atau flat.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_up, col_path = st.columns(2)
    with col_up:
        uploaded_zip = st.file_uploader(
            "Unggah ZIP dataset",
            accept_multiple_files=False,
            type=["zip"],
            key="zip_uploader",
        )
    with col_path:
        local_path = st.text_input(
            "Atau path folder lokal:",
            value="",
            placeholder="/content/att_faces  atau  /kaggle/input/orl_faces",
        )

    # Resolve dataset folder
    dataset_folder: Optional[str] = None
    if uploaded_zip is not None:
        try:
            tmp_dir = tempfile.mkdtemp(prefix="eigenfaces_zip_")
            zip_path = os.path.join(tmp_dir, uploaded_zip.name)
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmp_dir)
            # Find the directory containing images (skip top-level if it's a wrapper)
            dataset_folder = ec.detect_dataset(tmp_dir, interactive=False)
            st.session_state["dataset_folder"] = dataset_folder
            st.session_state["dataset_tmp_dir"] = tmp_dir
            st.success(f"ZIP diekstrak ke: `{tmp_dir}`\n\nDataset terdeteksi: `{dataset_folder}`")
        except Exception as e:
            st.error(f"Gagal ekstrak ZIP: {e}")
    elif local_path.strip():
        try:
            dataset_folder = ec.detect_dataset(local_path.strip(), interactive=False)
            st.session_state["dataset_folder"] = dataset_folder
            st.success(f"Dataset terdeteksi: `{dataset_folder}`")
        except Exception as e:
            st.error(f"Deteksi gagal: {e}")

    folder = st.session_state.get("dataset_folder")
    if not folder:
        st.info("Belum ada dataset. Unggah ZIP atau isi path folder di atas.")
        return

    # Show detection info
    st.markdown("---")
    st.subheader("Info dataset")

    try:
        n_imgs = ec.count_images(folder)
        dist = ec.get_subfolder_distribution(folder)
        n_id = len(dist)
    except Exception as e:
        st.error(f"Gagal membaca info dataset: {e}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gambar", n_imgs)
    c2.metric("Identitas", n_id)
    c3.metric("Backend I/O", ec.get_backend())
    c4.metric("RAM (sekarang)",
              f"{ec.mem_used_mb():.0f} MB" if ec.has_psutil() else "—")

    if dist:
        st.markdown("**Distribusi per identitas:**")
        # Table: identity -> count
        rows = [{"identitas": k, "gambar": v} for k, v in sorted(dist.items())]
        st.table(rows)

    # Load button
    st.markdown("---")
    if st.button("Load Dataset", type="primary", use_container_width=False):
        try:
            with st.spinner("Memuat gambar ke matriks X..."):
                X, y, label_names, paths = cached_load_dataset(
                    folder, params["image_size_tuple"]
                )
            st.session_state["X"] = X
            st.session_state["y"] = y
            st.session_state["label_names"] = label_names
            st.session_state["paths"] = paths
            st.success(f"Loaded: X={X.shape}, y={y.shape}, identitas={len(label_names)}, "
                       f"RAM={ec.mem_used_mb():.0f} MB")
        except Exception as e:
            st.error(f"Load gagal: {e}")

    # If already loaded, show preview
    X = st.session_state.get("X")
    y = st.session_state.get("y")
    paths = st.session_state.get("paths")
    label_names = st.session_state.get("label_names")
    if X is not None and paths is not None:
        st.markdown("---")
        st.subheader("Preview 8 gambar pertama")
        n_prev = min(8, X.shape[0])
        cols = st.columns(4)
        for i in range(n_prev):
            with cols[i % 4]:
                img = ec.vector_to_image(X[i], params["image_size_tuple"])
                lbl = label_names[y[i]] if label_names is not None else str(y[i])
                st.image(img, caption=f"#{i} • {lbl}", use_container_width=True)

        _card(
            "Ringkasan matriks",
            f"<p class='ef-muted'>X shape:</p>"
            f"<p class='ef-mono'>{X.shape} &nbsp; dtype={X.dtype}</p>"
            f"<p class='ef-muted'>Total sampel: <b>{X.shape[0]}</b> &nbsp;|&nbsp; "
            f"Dimensi: <b>{X.shape[1]}</b> &nbsp;|&nbsp; "
            f"Identitas: <b>{len(label_names) if label_names else '?'}</b> &nbsp;|&nbsp; "
            f"RAM: <b>{ec.mem_used_mb():.0f} MB</b></p>",
        )


# --------------------------------------------------------------------------- #
# Tab 2 — Training
# --------------------------------------------------------------------------- #
def tab_training(params: dict) -> None:
    st.header("2. Training")

    X = st.session_state.get("X")
    y = st.session_state.get("y")
    label_names = st.session_state.get("label_names")

    if X is None or y is None:
        st.warning("Belum ada dataset. Buka tab 1 dan klik Load Dataset dulu.")
        return

    # Parameter summary
    body = (
        f"<p class='ef-muted'>image_size: <span class='ef-mono'>{params['image_size_label']}</span> "
        f"({params['image_size_tuple'][0]}x{params['image_size_tuple'][1]})</p>"
        f"<p class='ef-muted'>variance_threshold: <span class='ef-mono'>{params['variance_threshold']:.2f}</span> &nbsp;|&nbsp; "
        f"whiten: <span class='ef-mono'>{params['whiten']}</span> &nbsp;|&nbsp; "
        f"metric: <span class='ef-mono'>{params['metric']}</span> &nbsp;|&nbsp; "
        f"factor: <span class='ef-mono'>{params['factor']:.2f}</span></p>"
        f"<p class='ef-muted'>n_components_override: <span class='ef-mono'>{params['n_components_override']}</span> "
        f"(0 = auto dari variance threshold)</p>"
        f"<p class='ef-muted'>X shape: <span class='ef-mono'>{X.shape}</span> &nbsp; dtype={X.dtype}</p>"
    )
    _card("Parameter training", body)

    if st.button("Train Eigenfaces", type="primary"):
        try:
            t0 = time.time()
            with st.spinner("Melatih model PCA/SVD (Turk-Pentland dual trick bila N<d)..."):
                model = cached_fit_eigenfaces(
                    X,
                    params["variance_threshold"],
                    params["whiten"],
                    params["image_size_tuple"],
                    params["n_components_override"],
                )
            elapsed = time.time() - t0

            # Attach train_labels + label_names for downstream use
            model.train_labels = y
            model.label_names = label_names

            # Resolve thresholds via vectorized auto-tuning
            euc_thr, cos_thr = ec.resolve_thresholds(
                model, y, params["metric"], None, None, factor=params["factor"]
            )

            st.session_state["model"] = model
            st.session_state["train_time"] = elapsed
            st.session_state["euc_thr"] = float(euc_thr)
            st.session_state["cos_thr"] = float(cos_thr)
            st.success(f"Training selesai dalam {elapsed:.2f}s.")
            _maybe_gc()
        except Exception as e:
            st.error(f"Training gagal: {e}")
            import traceback
            st.expander("Traceback").code(traceback.format_exc())

    model = st.session_state.get("model")
    if model is None:
        st.info("Klik 'Train Eigenfaces' untuk melatih model.")
        return

    elapsed = st.session_state.get("train_time", 0.0)
    euc_thr = st.session_state.get("euc_thr")
    cos_thr = st.session_state.get("cos_thr")

    # Results
    rep = model.report
    k = model.eigenfaces.shape[0]
    total = rep.total_components if rep else k
    method = rep.method if rep else "?"
    var_kept = float(model.cumulative_variance[-1]) if model.cumulative_variance.size else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Metode", method)
    m2.metric("k / total", f"{k} / {total}")
    m3.metric("Variansi ditangkap", f"{var_kept*100:.2f}%")
    m4.metric("Waktu", f"{elapsed:.2f}s")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("n_train", model.n_train)
    m6.metric("Whiten", "ON" if model.whiten_scale is not None else "OFF")
    m7.metric("RAM setelah", f"{ec.mem_used_mb():.0f} MB" if ec.has_psutil() else "—")
    m8.metric("n_components_95",
              rep.n_components_95 if rep else "?")

    _card(
        "Thresholds otomatis",
        f"<p class='ef-muted'>Euclidean thr: <span class='ef-mono'>{euc_thr:.4f}</span> "
        f"(distance ≤ thr = same)</p>"
        f"<p class='ef-muted'>Cosine thr: <span class='ef-mono'>{cos_thr:.4f}</span> "
        f"(similarity ≥ thr = same)</p>"
        f"<p class='ef-muted'>Metric aktif: <span class='ef-mono'>{params['metric']}</span> "
        f"— factor: <span class='ef-mono'>{params['factor']:.2f}</span></p>",
    )


# --------------------------------------------------------------------------- #
# Tab 3 — Visualisasi
# --------------------------------------------------------------------------- #
def tab_visualization(params: dict) -> None:
    st.header("3. Visualisasi")

    model: Optional[ec.EigenFaceModel] = st.session_state.get("model")
    X = st.session_state.get("X")
    y = st.session_state.get("y")
    label_names = st.session_state.get("label_names")

    if model is None or X is None:
        st.warning("Model belum dilatih. Buka tab 2 dulu.")
        return

    image_size = params["image_size_tuple"]

    # --- Mean face ---
    st.subheader("Mean Face")
    mean_img = ec.vector_to_image(model.mean_face, image_size)
    c1, c2 = st.columns([1, 3])
    with c1:
        st.image(mean_img, caption="mean_face", use_container_width=True, clamp=True)
    with c2:
        _card(
            "Statistik mean face",
            f"<p class='ef-muted'>Shape: <span class='ef-mono'>{model.mean_face.shape}</span></p>"
            f"<p class='ef-muted'>min: <span class='ef-mono'>{model.mean_face.min():.2f}</span> &nbsp; "
            f"max: <span class='ef-mono'>{model.mean_face.max():.2f}</span> &nbsp; "
            f"mean: <span class='ef-mono'>{model.mean_face.mean():.2f}</span></p>"
            f"<p class='ef-muted'>Setiap eigenface dilatih pada (x - mean_face).</p>",
        )

    st.markdown("---")

    # --- Eigenfaces ---
    st.subheader("Eigenfaces (komponen utama)")
    k_avail = model.eigenfaces.shape[0]
    n_show = st.slider("Jumlah eigenfaces ditampilkan", 1, max(1, min(32, k_avail)),
                       min(8, k_avail))
    cols = st.columns(4)
    for i in range(n_show):
        with cols[i % 4]:
            ef = model.eigenfaces[i]
            img = ec.vector_to_image(ef, image_size)
            evr = float(model.explained_variance_ratio[i]) * 100 if i < len(model.explained_variance_ratio) else 0.0
            st.image(img, caption=f"EF #{i+1} • {evr:.2f}%", use_container_width=True)

    st.markdown("---")

    # --- Reconstruction ---
    st.subheader("Rekonstruksi bertahap")
    n_samples = X.shape[0]
    idx = st.slider("Pilih sampel", 0, n_samples - 1, 0)
    ks_wish = [1, 5, 10, 25, 50, 100]
    ks = [k for k in ks_wish if k <= k_avail]
    if not ks:
        ks = [k_avail]

    try:
        recs = ec.reconstruct_up_to(model, X[idx], ks)
    except Exception as e:
        st.error(f"Rekonstruksi gagal: {e}")
        recs = {}

    orig_img = ec.vector_to_image(X[idx], image_size)
    lbl = label_names[y[idx]] if label_names is not None and y is not None else "?"

    n_cols = 1 + len(ks)
    rcols = st.columns(n_cols)
    with rcols[0]:
        st.image(orig_img, caption=f"Original #{idx} • {lbl}", use_container_width=True, clamp=True)
    for j, k in enumerate(ks):
        if k in recs:
            with rcols[j + 1]:
                img = ec.vector_to_image(recs[k], image_size)
                # Compute relative L2 error vs original
                try:
                    diff = recs[k].astype(np.float64) - X[idx].astype(np.float64)
                    rel = np.linalg.norm(diff) / max(np.linalg.norm(X[idx].astype(np.float64)), 1e-9)
                    cap = f"k={k}\nrel err {rel*100:.1f}%"
                except Exception:
                    cap = f"k={k}"
                st.image(img, caption=cap, use_container_width=True, clamp=True)

    st.markdown("---")

    # --- Variance plots ---
    st.subheader("Variansi")

    evr = np.asarray(model.explained_variance_ratio, dtype=np.float64)
    cum = np.asarray(model.cumulative_variance, dtype=np.float64)
    eigvals = np.asarray(model.eigenvalues, dtype=np.float64)

    try:
        p1, p2, p3 = st.columns(3)

        with p1:
            fig1, ax = plt.subplots(figsize=(4.2, 3.2))
            ax.semilogy(np.arange(1, len(eigvals) + 1), eigvals, marker=".",
                        color="#14b8a6", linestyle="-")
            ax.set_title("Scree (log eigenvalues)", color="#e2e8f0", fontsize=10)
            ax.set_xlabel("komponen", color="#94a3b8", fontsize=8)
            ax.set_ylabel("λ (log)", color="#94a3b8", fontsize=8)
            ax.tick_params(colors="#94a3b8", labelsize=7)
            ax.grid(True, alpha=0.25, color="#334155")
            for s in ax.spines.values():
                s.set_color("#334155")
            fig1.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#0f172a")
            st.pyplot(fig1)
            _close_fig(fig1)

        with p2:
            fig2, ax = plt.subplots(figsize=(4.2, 3.2))
            kk = min(len(evr), 30)
            ax.bar(np.arange(1, kk + 1), evr[:kk] * 100, color="#f59e0b", alpha=0.85)
            ax.set_title("Per-component variance (%)", color="#e2e8f0", fontsize=10)
            ax.set_xlabel("komponen", color="#94a3b8", fontsize=8)
            ax.set_ylabel("% varians", color="#94a3b8", fontsize=8)
            ax.tick_params(colors="#94a3b8", labelsize=7)
            ax.grid(True, alpha=0.25, axis="y", color="#334155")
            for s in ax.spines.values():
                s.set_color("#334155")
            fig2.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#0f172a")
            st.pyplot(fig2)
            _close_fig(fig2)

        with p3:
            fig3, ax = plt.subplots(figsize=(4.2, 3.2))
            xs = np.arange(1, len(cum) + 1)
            ax.plot(xs, cum * 100, color="#14b8a6", linewidth=2.0)
            ax.axhline(95, color="#f59e0b", linestyle="--", linewidth=1.2, label="95%")
            ax.axhline(99, color="#ef4444", linestyle=":", linewidth=1.2, label="99%")
            ax.set_title("Cumulative variance", color="#e2e8f0", fontsize=10)
            ax.set_xlabel("komponen", color="#94a3b8", fontsize=8)
            ax.set_ylabel("% kumulatif", color="#94a3b8", fontsize=8)
            ax.tick_params(colors="#94a3b8", labelsize=7)
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.25, color="#334155")
            ax.legend(loc="lower right", fontsize=7, facecolor="#1e293b",
                      edgecolor="#334155", labelcolor="#e2e8f0")
            for s in ax.spines.values():
                s.set_color("#334155")
            fig3.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#0f172a")
            st.pyplot(fig3)
            _close_fig(fig3)
    except Exception as e:
        st.error(f"Plot variansi gagal: {e}")

    _maybe_gc()


# --------------------------------------------------------------------------- #
# Tab 4 — Identifikasi (nearest neighbor)
# --------------------------------------------------------------------------- #
def tab_identify(params: dict) -> None:
    st.header("4. Identifikasi (Nearest Neighbor)")

    model: Optional[ec.EigenFaceModel] = st.session_state.get("model")
    paths = st.session_state.get("paths")
    y = st.session_state.get("y")
    label_names = st.session_state.get("label_names")

    if model is None or paths is None or y is None:
        st.warning("Model & dataset belum siap. Selesaikan tab 1 dan 2 dulu.")
        return

    metric = params["metric"]
    euc_thr = st.session_state.get("euc_thr", 4000.0)
    cos_thr = st.session_state.get("cos_thr", 0.85)

    uploaded = st.file_uploader(
        "Unggah wajah query",
        type=["png", "jpg", "jpeg", "bmp", "pgm"],
        key="identify_query",
    )

    if uploaded is None:
        st.info("Unggah sebuah gambar wajah untuk diidentifikasi.")
        return

    # Save uploaded to temp file, load via core
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}")
        tmp.write(uploaded.getbuffer())
        tmp.close()
        query_vec = ec.load_single_image(tmp.name, image_size=params["image_size_tuple"])
    except Exception as e:
        st.error(f"Gagal membaca gambar query: {e}")
        return
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    # Project & match
    try:
        wq = ec.project_image(model, query_vec)
        match = ec.nearest_neighbor_match(
            wq, model.weights, y, label_names=label_names, metric=metric,
        )
    except Exception as e:
        st.error(f"Proyeksi / matching gagal: {e}")
        return

    # Decision
    if metric == "euclidean":
        accept = match["best_euclidean"] <= euc_thr
        primary_val = match["best_euclidean"]
        thr = euc_thr
    else:
        accept = match["best_cosine"] >= cos_thr
        primary_val = match["best_cosine"]
        thr = cos_thr

    # Show query + best match
    q_img = ec.vector_to_image(query_vec, params["image_size_tuple"])
    best_idx = match["best_index"]
    best_path = paths[best_idx]
    try:
        best_vec = ec.load_single_image(best_path, image_size=params["image_size_tuple"])
        b_img = ec.vector_to_image(best_vec, params["image_size_tuple"])
    except Exception:
        b_img = q_img

    c1, c2 = st.columns(2)
    with c1:
        _card("Query",
              f"<p class='ef-muted'>Nama file: <span class='ef-mono'>{uploaded.name}</span></p>")
        st.image(q_img, caption="Query", use_container_width=True, clamp=True)
    with c2:
        badge_cls = "same" if accept else "diff"
        badge_txt = "ACCEPT" if accept else "REJECT"
        _card(
            f"Best match — {match['best_label_name']}",
            f"<p class='ef-muted'>Index: <span class='ef-mono'>{best_idx}</span> &nbsp;|&nbsp; "
            f"Path: <span class='ef-mono'>{os.path.basename(best_path)}</span></p>"
            f"<p class='ef-muted'>Metric: <span class='ef-mono'>{metric}</span> &nbsp;|&nbsp; "
            f"value: <span class='ef-mono'>{primary_val:.4f}</span> &nbsp;|&nbsp; "
            f"threshold: <span class='ef-mono'>{thr:.4f}</span></p>"
            f"<p><span class='ef-badge {badge_cls}'>{badge_txt}</span></p>"
            f"<p class='ef-muted'>Euclidean: <span class='ef-mono'>{match['best_euclidean']:.4f}</span> "
            f"&nbsp;|&nbsp; Cosine: <span class='ef-mono'>{match['best_cosine']:.4f}</span></p>",
        )
        st.image(b_img, caption=f"Best: {match['best_label_name']}", use_container_width=True, clamp=True)

    # Top-5
    st.markdown("---")
    st.subheader("Top-5 matches")
    all_euc = match["all_euclidean"]
    all_cos = match["all_cosine"]
    if metric == "euclidean":
        order = np.argsort(all_euc)[:5]
    else:
        order = np.argsort(-all_cos)[:5]

    tcols = st.columns(5)
    for j, idx2 in enumerate(order):
        with tcols[j]:
            try:
                vec2 = ec.load_single_image(paths[idx2], image_size=params["image_size_tuple"])
                img2 = ec.vector_to_image(vec2, params["image_size_tuple"])
            except Exception:
                img2 = np.zeros((params["image_size_tuple"][0], params["image_size_tuple"][1]),
                                dtype=np.uint8)
            name = label_names[y[idx2]] if label_names else str(y[idx2])
            e_val = float(all_euc[idx2])
            c_val = float(all_cos[idx2])
            st.image(img2, caption=f"#{j+1} • {name}\ne={e_val:.2f}\nc={c_val:.3f}",
                     use_container_width=True, clamp=True)

    _maybe_gc()


# --------------------------------------------------------------------------- #
# Tab 5 — Verifikasi (compare two faces)
# --------------------------------------------------------------------------- #
def tab_verify(params: dict) -> None:
    st.header("5. Verifikasi (bandingkan dua wajah)")

    model: Optional[ec.EigenFaceModel] = st.session_state.get("model")
    if model is None:
        st.warning("Model belum dilatih. Buka tab 2 dulu.")
        return

    metric = params["metric"]
    euc_thr = st.session_state.get("euc_thr", 4000.0)
    cos_thr = st.session_state.get("cos_thr", 0.85)

    cA, cB = st.columns(2)
    with cA:
        upA = st.file_uploader("Wajah A", type=["png", "jpg", "jpeg", "bmp", "pgm"],
                               key="verify_A")
    with cB:
        upB = st.file_uploader("Wajah B", type=["png", "jpg", "jpeg", "bmp", "pgm"],
                               key="verify_B")

    if upA is None or upB is None:
        st.info("Unggah dua gambar wajah untuk dibandingkan.")
        return

    # Load both
    def _load_uploaded(up, name):
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{name}")
            tmp.write(up.getbuffer())
            tmp.close()
            vec = ec.load_single_image(tmp.name, image_size=params["image_size_tuple"])
            return vec
        except Exception as e:
            st.error(f"Gagal membaca {name}: {e}")
            return None
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    vecA = _load_uploaded(upA, "A")
    vecB = _load_uploaded(upB, "B")
    if vecA is None or vecB is None:
        return

    # Project both
    try:
        wA = ec.project_image(model, vecA)
        wB = ec.project_image(model, vecB)
        result = ec.compare(
            wA, wB,
            metric=metric,
            euclidean_threshold=euc_thr,
            cosine_threshold=cos_thr,
            ref_label="A",
            query_label="B",
        )
    except Exception as e:
        st.error(f"Verifikasi gagal: {e}")
        return

    imgA = ec.vector_to_image(vecA, params["image_size_tuple"])
    imgB = ec.vector_to_image(vecB, params["image_size_tuple"])

    # Side-by-side images
    col1, col2 = st.columns(2)
    with col1:
        st.image(imgA, caption=f"Wajah A — {upA.name}", use_container_width=True, clamp=True)
    with col2:
        st.image(imgB, caption=f"Wajah B — {upB.name}", use_container_width=True, clamp=True)

    # Verdict card
    same = result.same_person
    badge_cls = "same" if same else "diff"
    badge_txt = "SAME PERSON" if same else "DIFFERENT PERSON"

    thr = result.threshold
    _card(
        "Verdict",
        f"<p><span class='ef-badge {badge_cls}'>{badge_txt}</span></p>"
        f"<p class='ef-muted'>Metric aktif: <span class='ef-mono'>{result.decision_metric}</span></p>"
        f"<p class='ef-muted'>Threshold: <span class='ef-mono'>{thr:.4f}</span></p>"
        f"<p class='ef-muted'>Euclidean distance: <span class='ef-mono'>{result.euclidean_distance:.4f}</span></p>"
        f"<p class='ef-muted'>Cosine similarity: <span class='ef-mono'>{result.cosine_similarity:.4f}</span></p>"
        f"<p class='ef-muted'>Cosine distance: <span class='ef-mono'>{result.cosine_distance:.4f}</span></p>"
        f"<p class='ef-muted'>ref_label: <span class='ef-mono'>{result.ref_label}</span> "
        f"&nbsp;|&nbsp; query_label: <span class='ef-mono'>{result.query_label}</span></p>",
    )

    _maybe_gc()


# --------------------------------------------------------------------------- #
# Tab 6 — Evaluasi (gallery)
# --------------------------------------------------------------------------- #
def tab_evaluate(params: dict) -> None:
    st.header("6. Evaluasi (Gallery)")

    model: Optional[ec.EigenFaceModel] = st.session_state.get("model")
    y = st.session_state.get("y")
    if model is None or y is None or model.weights is None:
        st.warning("Model belum dilatih. Buka tab 2 dulu.")
        return

    metric = params["metric"]
    euc_thr = st.session_state.get("euc_thr", 4000.0)
    cos_thr = st.session_state.get("cos_thr", 0.85)

    _card(
        "Konfigurasi evaluasi",
        f"<p class='ef-muted'>N samples: <span class='ef-mono'>{model.weights.shape[0]}</span> &nbsp;|&nbsp; "
        f"k components: <span class='ef-mono'>{model.weights.shape[1]}</span></p>"
        f"<p class='ef-muted'>Metric: <span class='ef-mono'>{metric}</span> &nbsp;|&nbsp; "
        f"euc_thr: <span class='ef-mono'>{euc_thr:.4f}</span> &nbsp;|&nbsp; "
        f"cos_thr: <span class='ef-mono'>{cos_thr:.4f}</span></p>"
        f"<p class='ef-muted'>Semua pasangan upper-triangle (N(N-1)/2) dievaluasi secara vektorisasi.</p>",
    )

    if st.button("Run Gallery Evaluation", type="primary"):
        try:
            with st.spinner("Menghitung pairwise matrix + metrik..."):
                rep = ec.evaluate_gallery(
                    model.weights, y,
                    metric=metric,
                    euclidean_threshold=euc_thr,
                    cosine_threshold=cos_thr,
                    chunk=256,
                )
            st.session_state["gallery_rep"] = rep
            st.success("Evaluasi selesai.")
            _maybe_gc()
        except Exception as e:
            st.error(f"Evaluasi gagal: {e}")
            import traceback
            st.expander("Traceback").code(traceback.format_exc())

    rep = st.session_state.get("gallery_rep")
    if rep is None:
        st.info("Klik 'Run Gallery Evaluation' untuk menjalankan.")
        return

    # Metric tiles
    g1, g2, g3, g4 = st.columns(4)
    g1.markdown(_metric_tile(f"{rep.accuracy*100:.2f}%", "Accuracy"), unsafe_allow_html=True)
    g2.markdown(_metric_tile(f"{rep.precision*100:.2f}%", "Precision"), unsafe_allow_html=True)
    g3.markdown(_metric_tile(f"{rep.recall*100:.2f}%", "Recall"), unsafe_allow_html=True)
    g4.markdown(_metric_tile(f"{rep.f1*100:.2f}%", "F1"), unsafe_allow_html=True)

    # Confusion counts
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TP", rep.tp)
    c2.metric("FP", rep.fp)
    c3.metric("TN", rep.tn)
    c4.metric("FN", rep.fn)

    # Confusion matrix 2x2
    st.markdown("---")
    st.subheader("Confusion matrix (2x2)")
    try:
        cm = np.array([[rep.tp, rep.fn],
                       [rep.fp, rep.tn]], dtype=np.float64)
        fig_cm, ax = plt.subplots(figsize=(4.2, 3.6))
        im = ax.imshow(cm, cmap="YlGnBu", aspect="auto")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Same", "Pred Diff"], color="#94a3b8", fontsize=9)
        ax.set_yticklabels(["Actual Same", "Actual Diff"], color="#94a3b8", fontsize=9)
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, f"{int(cm[ii, jj])}",
                        ha="center", va="center",
                        color="#0f172a" if cm[ii, jj] > cm.max() / 2 else "#e2e8f0",
                        fontsize=14, fontweight="bold")
        ax.set_title("Confusion Matrix", color="#e2e8f0", fontsize=11)
        fig_cm.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")
        for s in ax.spines.values():
            s.set_color("#334155")
        st.pyplot(fig_cm)
        _close_fig(fig_cm)
    except Exception as e:
        st.error(f"Plot confusion matrix gagal: {e}")

    # Distribution histograms (intra vs inter)
    st.markdown("---")
    st.subheader("Distribusi intra vs inter-class")
    try:
        col_e, col_c = st.columns(2)
        with col_e:
            fig_e, ax = plt.subplots(figsize=(5.2, 3.4))
            bins = 40
            ax.hist(rep.intra_euclid, bins=bins, alpha=0.6, color="#22c55e",
                    label=f"intra (mean={rep.intra_euclid_mean:.2f})")
            ax.hist(rep.inter_euclid, bins=bins, alpha=0.6, color="#ef4444",
                    label=f"inter (mean={rep.inter_euclid_mean:.2f})")
            if metric == "euclidean":
                ax.axvline(euc_thr, color="#f59e0b", linestyle="--", linewidth=1.5,
                           label=f"thr={euc_thr:.2f}")
            ax.set_title("Euclidean distance", color="#e2e8f0", fontsize=10)
            ax.set_xlabel("distance", color="#94a3b8", fontsize=8)
            ax.set_ylabel("count", color="#94a3b8", fontsize=8)
            ax.tick_params(colors="#94a3b8", labelsize=7)
            ax.grid(True, alpha=0.25, color="#334155")
            ax.legend(fontsize=7, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
            for s in ax.spines.values():
                s.set_color("#334155")
            fig_e.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#0f172a")
            st.pyplot(fig_e)
            _close_fig(fig_e)

        with col_c:
            fig_c, ax = plt.subplots(figsize=(5.2, 3.4))
            bins = 40
            ax.hist(rep.intra_cosine, bins=bins, alpha=0.6, color="#22c55e",
                    label=f"intra (mean={rep.intra_cosine_mean:.3f})")
            ax.hist(rep.inter_cosine, bins=bins, alpha=0.6, color="#ef4444",
                    label=f"inter (mean={rep.inter_cosine_mean:.3f})")
            if metric == "cosine":
                ax.axvline(cos_thr, color="#f59e0b", linestyle="--", linewidth=1.5,
                           label=f"thr={cos_thr:.3f}")
            ax.set_title("Cosine similarity", color="#e2e8f0", fontsize=10)
            ax.set_xlabel("similarity", color="#94a3b8", fontsize=8)
            ax.set_ylabel("count", color="#94a3b8", fontsize=8)
            ax.tick_params(colors="#94a3b8", labelsize=7)
            ax.grid(True, alpha=0.25, color="#334155")
            ax.legend(fontsize=7, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
            for s in ax.spines.values():
                s.set_color("#334155")
            fig_c.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#0f172a")
            st.pyplot(fig_c)
            _close_fig(fig_c)
    except Exception as e:
        st.error(f"Plot distribusi gagal: {e}")

    # Means summary
    _card(
        "Rangkuman statistik",
        f"<p class='ef-muted'>intra_euclid_mean: <span class='ef-mono'>{rep.intra_euclid_mean:.4f}</span> &nbsp;|&nbsp; "
        f"inter_euclid_mean: <span class='ef-mono'>{rep.inter_euclid_mean:.4f}</span></p>"
        f"<p class='ef-muted'>intra_cosine_mean: <span class='ef-mono'>{rep.intra_cosine_mean:.4f}</span> &nbsp;|&nbsp; "
        f"inter_cosine_mean: <span class='ef-mono'>{rep.inter_cosine_mean:.4f}</span></p>"
        f"<p class='ef-muted'>Total pairs: <span class='ef-mono'>{rep.total}</span> "
        f"(same: {rep.n_same}, diff: {rep.n_diff})</p>",
    )

    _maybe_gc()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    params = render_sidebar()

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Dataset",
        "2. Training",
        "3. Visualisasi",
        "4. Identifikasi",
        "5. Verifikasi",
        "6. Evaluasi",
    ])

    with tab1:
        tab_dataset(params)
    with tab2:
        tab_training(params)
    with tab3:
        tab_visualization(params)
    with tab4:
        tab_identify(params)
    with tab5:
        tab_verify(params)
    with tab6:
        tab_evaluate(params)


if __name__ == "__main__":
    main()
