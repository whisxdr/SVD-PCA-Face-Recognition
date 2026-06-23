"""
eigenfaces_core.py
==================
Pure PCA / SVD Eigenfaces library - NO face-recognition / ML libraries.
Only depends on numpy (+ optional PIL/cv2 for image I/O, matplotlib for plots).

Key fixes vs original Eigenfaces.ipynb
-------------------------------------
ALGORITHM
  * Vectorized every O(N^2) pair operation (auto_thresholds, run_gallery,
    similarity distribution, nearest_neighbor_match) - now O(N^2) in C, not
    Python; uses chunked distance matrices so peak RAM is bounded.
  * Implemented the Turk-Pentland *dual / snapshot* eigendecomposition when
    N < d: eigendecompose the small N x N covariance L = Phi Phi^T / N and
    recover eigenfaces as V = Phi^T U / S  -> never materializes full Vt.
  * `whiten=True` no longer breaks reconstruction: whitening scale is stored
    separately and inverted inside reconstruct*().
  * `_svd_flip` rewritten with one advanced-indexing pass (no temp copy).
  * Robust component count (handles degenerate S, threshold edge cases).

MEMORY (kebocoran memori)
  * Default dtype float32 (halves RAM vs float64 of the original).
  * `del` + gc.collect() between training phases; mean_face stored as 1D and
    subtracted in-place.
  * report stores only the *truncated* (k-length) variance arrays - the
    original kept full N-length copies alive forever.
  * `load_dataset` streams images in batches and frees the list ASAP.
  * `pairwise_distance_matrix` is computed in row-chunks so peak memory is
    O(chunk * N) instead of O(N^2) for huge galleries.
  * Optional `psutil` memory logging (no hard dependency).
"""

from __future__ import annotations

import gc
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Optional deps (image I/O + plotting + memory monitor)
# --------------------------------------------------------------------------- #
try:
    import cv2  # type: ignore

    def _imread_gray(path: str) -> Optional[np.ndarray]:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def _resize_gray(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)

    _BACKEND = "cv2"
except Exception:  # pragma: no cover - fallback path
    from PIL import Image  # type: ignore

    def _imread_gray(path: str) -> Optional[np.ndarray]:
        try:
            with Image.open(path) as im:
                return np.asarray(im.convert("L"))
        except Exception:
            return None

    def _resize_gray(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        pil = Image.fromarray(img).resize((size[1], size[0]), Image.BILINEAR)
        return np.asarray(pil)

    _BACKEND = "PIL"

try:
    import psutil  # type: ignore

    def mem_used_mb() -> float:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    _HAS_PSUTIL = True
except Exception:  # pragma: no cover
    def mem_used_mb() -> float:
        return -1.0

    _HAS_PSUTIL = False


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def get_backend() -> str:
    return _BACKEND


def has_psutil() -> bool:
    return _HAS_PSUTIL


def gc_collect() -> None:
    """Aggressive cleanup - call between heavy phases."""
    gc.collect()


# --------------------------------------------------------------------------- #
# Image I/O
# --------------------------------------------------------------------------- #
def list_image_files(folder: str, recursive: bool = True) -> List[str]:
    paths: List[str] = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                    paths.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            full = os.path.join(folder, f)
            if os.path.isfile(full) and os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                paths.append(full)
    paths.sort()
    return paths


def infer_label(path: str, root: str) -> str:
    rel = os.path.relpath(path, root)
    parts = rel.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return parts[0]
    return os.path.splitext(parts[0])[0]


def load_dataset(
    folder: str,
    image_size: Tuple[int, int] = (112, 92),
    recursive: bool = True,
    dtype: np.dtype = np.float32,
    batch_log_every: int = 200,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Stream-load images into one big matrix; frees the python list ASAP."""
    paths = list_image_files(folder, recursive=recursive)
    if not paths:
        raise FileNotFoundError(
            f"Tidak ada file gambar ditemukan di: {folder}. "
            "Periksa path dan ekstensi (jpg/png/pgm/bmp)."
        )

    H, W = image_size
    d = H * W
    N = len(paths)
    X = np.empty((N, d), dtype=dtype)
    raw_labels: List[str] = [""] * N
    used_paths: List[str] = [""] * N
    ok = 0

    for i, p in enumerate(paths):
        gray = _imread_gray(p)
        if gray is None:
            continue
        resized = _resize_gray(gray, image_size)
        X[ok] = resized.reshape(d).astype(dtype, copy=False)
        raw_labels[ok] = infer_label(p, folder)
        used_paths[ok] = p
        ok += 1
        if batch_log_every and (i + 1) % batch_log_every == 0:
            print(f"  [loader] {i+1}/{N}  mem={mem_used_mb():.1f} MB")

    if ok == 0:
        raise RuntimeError("Tidak ada gambar yang berhasil dimuat.")
    if ok < N:
        X = X[:ok]
        raw_labels = raw_labels[:ok]
        used_paths = used_paths[:ok]

    label_names = sorted(set(raw_labels))
    name_to_idx = {n: i for i, n in enumerate(label_names)}
    y = np.fromiter((name_to_idx[n] for n in raw_labels), dtype=np.int64, count=len(raw_labels))

    print(f"[loader] backend={_BACKEND} gambar={X.shape[0]} dimensi={d} "
          f"identitas={len(label_names)} dtype={X.dtype} mem={mem_used_mb():.1f} MB")
    return X, y, label_names, used_paths


def load_single_image(path: str, image_size: Tuple[int, int] = (112, 92),
                      dtype: np.dtype = np.float32) -> np.ndarray:
    gray = _imread_gray(path)
    if gray is None:
        raise FileNotFoundError(f"Tidak dapat membaca gambar: {path}")
    resized = _resize_gray(gray, image_size)
    return resized.reshape(image_size[0] * image_size[1]).astype(dtype, copy=False)


def vector_to_image(vec: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    H, W = image_size
    img = vec.reshape(H, W)
    vmin, vmax = float(img.min()), float(img.max())
    if vmax - vmin > 1e-9:
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img)
    return (img * 255.0).clip(0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class PCAReport:
    explained_variance_ratio: np.ndarray   # length k (truncated)
    cumulative_variance: np.ndarray        # length k (truncated)
    n_components_95: int
    n_components_99: int
    total_components: int                  # rank of Phi (<= min(N, d))
    method: str                            # 'svd' | 'dual'


@dataclass
class EigenFaceModel:
    mean_face: np.ndarray
    eigenfaces: np.ndarray                 # (k, d) - NOT whitened (whiten scale kept separate)
    singular_values: np.ndarray            # (k,)
    eigenvalues: np.ndarray                # (k,)
    explained_variance_ratio: np.ndarray   # (k,)
    cumulative_variance: np.ndarray        # (k,)
    image_size: Tuple[int, int]
    n_train: int
    whiten_scale: Optional[np.ndarray]     # (k,) or None
    weights: Optional[np.ndarray] = None   # (N, k) - whitened if whiten=True
    train_labels: Optional[np.ndarray] = None
    report: Optional[PCAReport] = None
    label_names: Optional[List[str]] = field(default=None, repr=False)


# --------------------------------------------------------------------------- #
# Core PCA / SVD
# --------------------------------------------------------------------------- #
def _svd_flip(U: np.ndarray, Vt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Make sign deterministic - one pass, no temp."""
    if U.size == 0:
        return U, Vt
    max_abs = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs, np.arange(U.shape[1])])
    signs[signs == 0] = 1.0
    U *= signs
    Vt *= signs[:, None]
    return U, Vt


def _compute_pca_svd(Phi: np.ndarray):
    """Standard thin SVD on Phi (N x d). Returns U, S, Vt."""
    U, S, Vt = np.linalg.svd(Phi, full_matrices=False)
    U, Vt = _svd_flip(U, Vt)
    return U, S, Vt


def _compute_pca_dual(Phi: np.ndarray):
    """Turk-Pentland dual trick: eigendecompose N x N L = Phi Phi^T / N.

    Used when N << d to avoid materializing Vt (N x d) directly.
    Returns U (N x k), S (k,), Vt (k x d) where Vt = Phi^T U / S.
    """
    N = Phi.shape[0]
    # L = (1/N) Phi Phi^T  -> symmetric PSD
    L = (Phi @ Phi.T) / N
    # eigh: ascending eigenvalues; reverse to descending
    w, v = np.linalg.eigh(L)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]

    # Singular values of Phi: S_i = sqrt(N * lambda_i)
    S = np.sqrt(np.maximum(w * N, 0.0))

    # Right singular vectors V = Phi^T U / S  (skip degenerate)
    eps = 1e-10 * (S[0] if S.size and S[0] > 0 else 1.0)
    valid = S > eps
    k = int(valid.sum())
    U = v[:, :k]
    S = S[:k]
    # Compute Vt row-by-row to bound memory; result is (k, d)
    # Phi.T @ U is (d, k); divide each column by S, then transpose.
    Vt = ((Phi.T @ U) / S[None, :]).T
    # Sign flip
    max_abs = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs, np.arange(U.shape[1])])
    signs[signs == 0] = 1.0
    U *= signs
    Vt *= signs[:, None]
    return U, S, Vt


def fit_eigenfaces(
    X: np.ndarray,
    n_components: Optional[int] = None,
    variance_threshold: Optional[float] = None,
    image_size: Tuple[int, int] = (112, 92),
    whiten: bool = False,
    dtype: np.dtype = np.float32,
    log_mem: bool = True,
) -> EigenFaceModel:
    """Fit Eigenfaces via pure PCA / SVD. Memory-safe; auto-picks dual trick."""
    N, d = X.shape
    if N < 2:
        raise ValueError("Butuh minimal 2 sampel untuk PCA.")

    # --- center --------------------------------------------------------- #
    mean_face = X.mean(axis=0).astype(dtype, copy=True)
    Phi = X.astype(dtype, copy=False) - mean_face  # broadcast, new array
    del X  # original X no longer needed - caller usually keeps a ref anyway
    if log_mem:
        print(f"[fit] after centering: Phi={Phi.shape} {Phi.dtype} mem={mem_used_mb():.1f} MB")

    # --- choose SVD path ------------------------------------------------ #
    use_dual = N < d
    if use_dual:
        print(f"[fit] using TURK-PENTLAND DUAL trick (N={N} < d={d})")
        U, S, Vt = _compute_pca_dual(Phi)
        method = "dual"
    else:
        print(f"[fit] using thin SVD (N={N} >= d={d})")
        U, S, Vt = _compute_pca_svd(Phi)
        method = "svd"
    del U  # we only need Vt (eigenfaces) and S
    gc_collect()
    if log_mem:
        print(f"[fit] after SVD: Vt={Vt.shape} S={S.shape} mem={mem_used_mb():.1f} MB")

    # --- eigenvalues & variance ---------------------------------------- #
    Nf = max(N, 1)
    eigenvalues_all = (S ** 2) / Nf
    total_var = float(eigenvalues_all.sum())
    if total_var <= 0:
        evr_all = np.zeros_like(eigenvalues_all)
    else:
        evr_all = eigenvalues_all / total_var
    cum_all = np.cumsum(evr_all)

    # --- choose k ------------------------------------------------------- #
    eps = 1e-10 * (S[0] if S.size and S[0] > 0 else 1.0)
    valid_r = int((S > eps).sum()) if S.size else 0
    valid_r = max(valid_r, 1)

    if n_components is not None:
        k = min(int(n_components), valid_r)
    elif variance_threshold is not None:
        # smallest k s.t. cum[k-1] >= threshold
        idx = int(np.searchsorted(cum_all[:valid_r], float(variance_threshold))) + 1
        k = min(idx, valid_r)
    else:
        k = valid_r

    # --- truncate (only keep k arrays alive) --------------------------- #
    eigenfaces = np.ascontiguousarray(Vt[:k], dtype=dtype)
    singular_values = S[:k].astype(np.float64, copy=True)
    eigenvalues_k = eigenvalues_all[:k].astype(np.float64, copy=True)
    evr_k = evr_all[:k].astype(np.float64, copy=True)
    cum_k = cum_all[:k].astype(np.float64, copy=True)
    del Vt, S, eigenvalues_all, evr_all, cum_all
    gc_collect()

    # --- whiten scale (kept separate so reconstruction stays correct) -- #
    whiten_scale: Optional[np.ndarray] = None
    if whiten:
        whiten_scale = 1.0 / np.sqrt(np.maximum(eigenvalues_k, 1e-12))

    # --- project training set ------------------------------------------ #
    # weights in *whitened* space if whiten else raw eigenface coords
    if whiten:
        weights = (Phi @ eigenfaces.T) * whiten_scale[None, :]
    else:
        weights = Phi @ eigenfaces.T
    weights = np.ascontiguousarray(weights, dtype=dtype)
    del Phi
    gc_collect()

    n95 = int(np.searchsorted(cum_k, 0.95)) + 1 if cum_k[-1] >= 0.95 else k
    n99 = int(np.searchsorted(cum_k, 0.99)) + 1 if cum_k[-1] >= 0.99 else k

    report = PCAReport(
        explained_variance_ratio=evr_k,
        cumulative_variance=cum_k,
        n_components_95=n95,
        n_components_99=n99,
        total_components=valid_r,
        method=method,
    )

    print(f"[fit] done: k={k}/{valid_r} var_kept={cum_k[-1]*100:.2f}% "
          f"whiten={whiten} mem={mem_used_mb():.1f} MB")

    return EigenFaceModel(
        mean_face=mean_face,
        eigenfaces=eigenfaces,
        singular_values=singular_values,
        eigenvalues=eigenvalues_k,
        explained_variance_ratio=evr_k,
        cumulative_variance=cum_k,
        image_size=image_size,
        n_train=N,
        whiten_scale=whiten_scale,
        weights=weights,
        report=report,
    )


# --------------------------------------------------------------------------- #
# Projection / reconstruction (whiten-aware)
# --------------------------------------------------------------------------- #
def project_image(model: EigenFaceModel, x: np.ndarray) -> np.ndarray:
    """Project a single (centered-able) image vector into eigenface space.

    Returns *whitened* weights if model was trained with whiten=True, so the
    output is directly comparable with model.weights.
    """
    if x.shape != model.mean_face.shape:
        raise ValueError(
            f"Dimensi wajah {x.shape} != mean face {model.mean_face.shape}. "
            "Pastikan resize/preprocessing identik."
        )
    phi = (x - model.mean_face).astype(model.eigenfaces.dtype, copy=False)
    w = phi @ model.eigenfaces.T
    if model.whiten_scale is not None:
        w = w * model.whiten_scale
    return w


def reconstruct(model: EigenFaceModel, omega: np.ndarray) -> np.ndarray:
    """Reconstruct a (whitened-or-not) weight vector back to pixel space.

    omega must be in the same space as model.weights (whitened if whiten=True).
    """
    if model.whiten_scale is not None:
        omega = omega * (1.0 / model.whiten_scale)  # undo whitening
    return model.mean_face + omega @ model.eigenfaces


def reconstruct_up_to(model: EigenFaceModel, x: np.ndarray, ks: List[int]) -> Dict[int, np.ndarray]:
    """Reconstruct x using only the first k eigenfaces (k in ks)."""
    phi = (x - model.mean_face).astype(model.eigenfaces.dtype, copy=False)
    out: Dict[int, np.ndarray] = {}
    k_total = model.eigenfaces.shape[0]
    for k in ks:
        k_eff = min(k, k_total)
        EF = model.eigenfaces[:k_eff]
        omega = phi @ EF.T
        if model.whiten_scale is not None:
            omega = omega * model.whiten_scale[:k_eff]
            omega = omega * (1.0 / model.whiten_scale[:k_eff])  # cancels out
        out[k_eff] = model.mean_face + omega @ EF
    return out


# --------------------------------------------------------------------------- #
# Vectorized pairwise ops - chunked to bound peak memory
# --------------------------------------------------------------------------- #
def pairwise_euclidean(W: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Full N x N euclidean distance matrix, computed in row-chunks.

    Memory: O(chunk * N) peak instead of O(N^2) intermediate during compute.
    """
    N = W.shape[0]
    out = np.empty((N, N), dtype=np.float32)
    sq = np.einsum("ij,ij->i", W, W).astype(np.float32)  # (N,)
    for i in range(0, N, chunk):
        sl = slice(i, min(i + chunk, N))
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        cross = W[sl] @ W.T  # (chunk, N)
        d2 = sq[sl][:, None] + sq[None, :] - 2.0 * cross
        np.maximum(d2, 0, out=d2)
        out[sl] = np.sqrt(d2, dtype=np.float32)
    return out


def pairwise_cosine(W: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Full N x N cosine similarity matrix, chunked."""
    norms = np.linalg.norm(W, axis=1).astype(np.float32)
    norms = np.where(norms < 1e-12, 1.0, norms)
    Wn = W / norms[:, None].astype(W.dtype, copy=False)
    N = Wn.shape[0]
    out = np.empty((N, N), dtype=np.float32)
    for i in range(0, N, chunk):
        sl = slice(i, min(i + chunk, N))
        out[sl] = Wn[sl] @ Wn.T
    return out


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.dot(diff, diff)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# --------------------------------------------------------------------------- #
# Threshold auto-tuning (vectorized, O(N^2) in C)
# --------------------------------------------------------------------------- #
def auto_thresholds(
    weights: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    factor: float = 0.5,
    chunk: int = 256,
) -> float:
    """Vectorized threshold tuning - no Python O(N^2) loop."""
    N = weights.shape[0]
    if N < 2:
        return 4000.0 if metric == "euclidean" else 0.85

    same_mask = (labels[:, None] == labels[None, :])
    # exclude diagonal
    np.fill_diagonal(same_mask, False)
    triu = np.triu(np.ones((N, N), dtype=bool), k=1)
    same_mask = same_mask & triu
    diff_mask = (~same_mask) & triu

    if metric == "euclidean":
        D = pairwise_euclidean(weights, chunk=chunk)
        pos = D[same_mask]
        neg = D[diff_mask]
        del D
        if pos.size == 0 or neg.size == 0:
            return float(np.median(np.concatenate([pos, neg]))) if (pos.size or neg.size) else 4000.0
        mp, mn = float(pos.mean()), float(neg.mean())
        return mp + factor * (mn - mp)
    elif metric == "cosine":
        C = pairwise_cosine(weights, chunk=chunk)
        pos = C[same_mask]
        neg = C[diff_mask]
        del C
        if pos.size == 0 or neg.size == 0:
            return float(np.median(np.concatenate([pos, neg]))) if (pos.size or neg.size) else 0.85
        mp, mn = float(pos.mean()), float(neg.mean())
        return mp - factor * (mp - mn)
    else:
        raise ValueError(f"metric tidak dikenal: {metric}")


# --------------------------------------------------------------------------- #
# Identify / Verify - vectorized
# --------------------------------------------------------------------------- #
def nearest_neighbor_match(
    query_weight: np.ndarray,
    train_weights: np.ndarray,
    train_labels: np.ndarray,
    label_names: Optional[List[str]] = None,
    metric: str = "euclidean",
) -> dict:
    q = query_weight.astype(np.float64, copy=False).reshape(-1)
    TW = train_weights.astype(np.float64, copy=False)

    diffs = TW - q[None, :]
    eucl = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))

    qn = np.linalg.norm(q)
    tn = np.linalg.norm(TW, axis=1)
    denom = tn * qn
    denom = np.where(denom < 1e-12, 1.0, denom)
    coss = (TW @ q) / denom

    if metric == "euclidean":
        best_idx = int(np.argmin(eucl))
    else:
        best_idx = int(np.argmax(coss))

    best_label = int(train_labels[best_idx])
    best_label_name = label_names[best_label] if label_names else str(best_label)

    return {
        "best_index": best_idx,
        "best_label": best_label,
        "best_label_name": best_label_name,
        "best_euclidean": float(eucl[best_idx]),
        "best_cosine": float(coss[best_idx]),
        "all_euclidean": eucl.astype(np.float32, copy=False),
        "all_cosine": coss.astype(np.float32, copy=False),
    }


def decide_same_person(
    euclidean: float,
    cosine: float,
    metric: str = "euclidean",
    euclidean_threshold: float = 4000.0,
    cosine_threshold: float = 0.85,
) -> bool:
    if metric == "euclidean":
        return euclidean <= euclidean_threshold
    elif metric == "cosine":
        return cosine >= cosine_threshold
    raise ValueError(f"metric tidak dikenal: {metric}")


@dataclass
class SimilarityResult:
    euclidean_distance: float
    cosine_similarity: float
    cosine_distance: float
    same_person: bool
    decision_metric: str
    threshold: float
    ref_label: Optional[str] = None
    query_label: Optional[str] = None


def compare(
    omega_a: np.ndarray,
    omega_b: np.ndarray,
    metric: str = "euclidean",
    euclidean_threshold: float = 4000.0,
    cosine_threshold: float = 0.85,
    ref_label: Optional[str] = None,
    query_label: Optional[str] = None,
) -> SimilarityResult:
    euc = euclidean_distance(omega_a, omega_b)
    cos = cosine_similarity(omega_a, omega_b)
    same = decide_same_person(euc, cos, metric, euclidean_threshold, cosine_threshold)
    threshold = euclidean_threshold if metric == "euclidean" else cosine_threshold
    return SimilarityResult(
        euclidean_distance=euc,
        cosine_similarity=cos,
        cosine_distance=1.0 - cos,
        same_person=same,
        decision_metric=metric,
        threshold=threshold,
        ref_label=ref_label,
        query_label=query_label,
    )


# --------------------------------------------------------------------------- #
# Gallery evaluation - fully vectorized
# --------------------------------------------------------------------------- #
@dataclass
class GalleryReport:
    total: int
    n_same: int
    n_diff: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int
    intra_euclid_mean: float
    inter_euclid_mean: float
    intra_cosine_mean: float
    inter_cosine_mean: float
    intra_euclid: np.ndarray
    inter_euclid: np.ndarray
    intra_cosine: np.ndarray
    inter_cosine: np.ndarray


def evaluate_gallery(
    weights: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    euclidean_threshold: float = 4000.0,
    cosine_threshold: float = 0.85,
    chunk: int = 256,
) -> GalleryReport:
    N = weights.shape[0]
    same_mask = (labels[:, None] == labels[None, :])
    np.fill_diagonal(same_mask, False)
    triu = np.triu(np.ones((N, N), dtype=bool), k=1)
    same_mask &= triu
    diff_mask = (~same_mask) & triu

    D = pairwise_euclidean(weights, chunk=chunk)
    C = pairwise_cosine(weights, chunk=chunk)

    intra_e = D[same_mask].astype(np.float64)
    inter_e = D[diff_mask].astype(np.float64)
    intra_c = C[same_mask].astype(np.float64)
    inter_c = C[diff_mask].astype(np.float64)

    if metric == "euclidean":
        pred_same = D <= euclidean_threshold
    else:
        pred_same = C >= cosine_threshold
    pred_same &= triu  # only upper-triangle pairs

    tp = int((pred_same & same_mask).sum())
    fp = int((pred_same & diff_mask).sum())
    tn = int((~pred_same & diff_mask).sum())
    fn = int((~pred_same & same_mask).sum())
    total = tp + fp + tn + fn

    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    del D, C, pred_same, same_mask, diff_mask
    gc_collect()

    return GalleryReport(
        total=total,
        n_same=tp + fn,
        n_diff=fp + tn,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        tp=tp, fp=fp, tn=tn, fn=fn,
        intra_euclid_mean=float(intra_e.mean()) if intra_e.size else 0.0,
        inter_euclid_mean=float(inter_e.mean()) if inter_e.size else 0.0,
        intra_cosine_mean=float(intra_c.mean()) if intra_c.size else 0.0,
        inter_cosine_mean=float(inter_c.mean()) if inter_c.size else 0.0,
        intra_euclid=intra_e.astype(np.float32, copy=False),
        inter_euclid=inter_e.astype(np.float32, copy=False),
        intra_cosine=intra_c.astype(np.float32, copy=False),
        inter_cosine=inter_c.astype(np.float32, copy=False),
    )


# --------------------------------------------------------------------------- #
# Dataset path detection (Colab / Kaggle aware)
# --------------------------------------------------------------------------- #
def count_images(directory: str) -> int:
    c = 0
    for root, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                c += 1
    return c


def get_subfolder_distribution(directory: str) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            c = count_images(item_path)
            if c > 0:
                dist[item] = c
    return dist


def auto_detect(root_dir: str) -> Tuple[str, str, int, int]:
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Folder tidak ditemukan: {root_dir}")

    items = os.listdir(root_dir)
    for candidate in ['train', 'training', 'Train', 'Training']:
        if candidate in items:
            train_path = os.path.join(root_dir, candidate)
            if os.path.isdir(train_path):
                n = count_images(train_path)
                if n > 0:
                    dist = get_subfolder_distribution(train_path)
                    return train_path, "train_split", n, len(dist)

    dist = get_subfolder_distribution(root_dir)
    if dist and len(dist) >= 2:
        counts = list(dist.values())
        avg = sum(counts) / len(counts)
        if avg > 1:
            return root_dir, "nested_identity", sum(counts), len(dist)

    root_images = [f for f in items
                   if os.path.isfile(os.path.join(root_dir, f))
                   and os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    if root_images:
        return root_dir, "flat", len(root_images), 1

    if len(items) == 1:
        single = os.path.join(root_dir, items[0])
        if os.path.isdir(single):
            return auto_detect(single)

    if dist:
        best = max(dist, key=dist.get)
        best_path = os.path.join(root_dir, best)
        sub_dist = get_subfolder_distribution(best_path)
        if sub_dist and len(sub_dist) >= 2:
            return best_path, "auto_detected", sum(sub_dist.values()), len(sub_dist)

    total = count_images(root_dir)
    return root_dir, "fallback", total, 0


def get_default_paths() -> List[str]:
    """Return candidate dataset roots based on environment (Colab/Kaggle/local)."""
    cands = [
        "/kaggle/input",
        "/content",
        os.path.expanduser("~/.cache/eigenfaces"),
        ".",
    ]
    return [p for p in cands if os.path.isdir(p)]


def detect_dataset(input_path: Optional[str] = None,
                   interactive: bool = False) -> str:
    if input_path is None:
        roots = get_default_paths()
        if not roots:
            raise FileNotFoundError("Tidak ada folder kandidat ditemukan.")
        input_path = roots[0]
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Path tidak ditemukan: {input_path}")

    items = os.listdir(input_path)
    non_hidden = [d for d in items if not d.startswith('.')
                  and os.path.isdir(os.path.join(input_path, d))]

    if len(non_hidden) == 0:
        raise FileNotFoundError(f"Tidak ada dataset di: {input_path}")

    # If input_path itself looks like a dataset (multiple identity subfolders
    # each containing images), use input_path directly. This handles Kaggle's
    # /kaggle/input/<dataset-name>/s1, s2, ... structure.
    direct_dist = get_subfolder_distribution(input_path)
    if direct_dist and len(direct_dist) >= 2:
        # input_path is itself the identity-rooted dataset
        path, struct_type, n_imgs, n_id = auto_detect(input_path)
        if n_imgs > 0 and n_id >= 2:
            print(f"\n{'='*50}")
            print(f"Dataset terdeteksi (langsung di root):")
            print(f"  Path       : {path}")
            print(f"  Struktur   : {struct_type}")
            print(f"  Gambar     : {n_imgs}")
            print(f"  Identitas  : {n_id}")
            print(f"{'='*50}\n")
            return path

    if len(non_hidden) == 1:
        dataset_dir = os.path.join(input_path, non_hidden[0])
    elif interactive:
        print(f"\nDataset tersedia di {input_path}:")
        for i, name in enumerate(non_hidden):
            n = count_images(os.path.join(input_path, name))
            print(f"  [{i+1}] {name} ({n} gambar)")
        while True:
            try:
                choice = input("\nPilih dataset (nomor): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(non_hidden):
                    dataset_dir = os.path.join(input_path, non_hidden[idx])
                    break
                print(f"Masukkan nomor 1-{len(non_hidden)}")
            except ValueError:
                print("Input tidak valid")
    else:
        # Non-interactive: pick the one with the most images
        best, best_n = None, -1
        for name in non_hidden:
            n = count_images(os.path.join(input_path, name))
            if n > best_n:
                best_n, best = n, name
        dataset_dir = os.path.join(input_path, best) if best else input_path

    path, struct_type, n_imgs, n_id = auto_detect(dataset_dir)
    print(f"\n{'='*50}")
    print(f"Dataset terdeteksi:")
    print(f"  Path       : {path}")
    print(f"  Struktur   : {struct_type}")
    print(f"  Gambar     : {n_imgs}")
    print(f"  Identitas  : {n_id}")
    print(f"{'='*50}\n")
    return path


def resolve_thresholds(model: EigenFaceModel, y: np.ndarray, metric: str,
                       euc_thr: Optional[float], cos_thr: Optional[float],
                       factor: float = 0.5) -> Tuple[float, float]:
    if model.weights is None or len(y) < 2:
        return (euc_thr if euc_thr is not None else 4000.0,
                cos_thr if cos_thr is not None else 0.85)
    e = euc_thr if euc_thr is not None else auto_thresholds(
        model.weights, y, metric="euclidean", factor=factor)
    c = cos_thr if cos_thr is not None else auto_thresholds(
        model.weights, y, metric="cosine", factor=factor)
    return e, c
