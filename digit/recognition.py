"""Recognition: live touch detection, contact localisation and a learned classifier.

The live parts (touch / no-touch and where the contact is) use only geometry
from :mod:`digit.processing`, so they work with no training data.  The learned
part is a thin, honest wrapper around scikit-learn: collect a few labelled
frames, train k-NN / SVM / logistic regression, measure accuracy on a held-out
split, and predict live.  See README for measured numbers on this sensor.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .processing import (
    contact_mask,
    contact_stats,
    grid_cell,
    localize,
    magnitude,
)


# ---------------------------------------------------------------------------
# live touch detection / localisation


@dataclass
class TouchResult:
    touch: bool
    area_px: int
    area_frac: float
    centroid: Optional[Tuple[float, float]]
    normalized_centroid: Optional[Tuple[float, float]]
    grid_cell: Optional[str]
    force: float
    peak_mag: float = 0.0
    n_blobs: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "touch": self.touch,
            "area_px": self.area_px,
            "area_frac": round(self.area_frac, 4),
            "centroid": None
            if self.centroid is None
            else [round(self.centroid[0], 1), round(self.centroid[1], 1)],
            "centroid_norm": None
            if self.normalized_centroid is None
            else [round(self.normalized_centroid[0], 3), round(self.normalized_centroid[1], 3)],
            "cell": self.grid_cell,
            "force": round(self.force, 1),
            "peak_mag": round(self.peak_mag, 2),
            "n_blobs": self.n_blobs,
        }


class TouchDetector:
    """Threshold-based touch detector with optional background cancellation.

    ``min_peak`` is an optional contrast gate: a candidate blob is only reported
    as touch if its ``peak_mag`` (the strongest per-pixel change) reaches it.
    This separates a **localised** contact from a broad, low-contrast
    disturbance such as cable strain or a small shift of the sensor, which can
    light up the difference over a wide area without any sharp contact edge.
    """

    def __init__(
        self,
        reference: np.ndarray,
        threshold: float = 10.0,
        min_area: int = 40,
        region: Optional[np.ndarray] = None,
        min_peak: float = 0.0,
    ) -> None:
        self.reference = reference
        self.threshold = float(threshold)
        self.min_area = int(min_area)
        self.min_peak = float(min_peak)
        from .processing import active_region

        self.region = active_region(reference) if region is None else region
        self.region_area = int((self.region > 0).sum())

    def detect(self, frame: np.ndarray) -> TouchResult:
        signed = frame.astype(np.float32) - self.reference.astype(np.float32)
        mask = contact_mask(
            signed, threshold=self.threshold, min_area=self.min_area, region=self.region
        )
        stats = contact_stats(mask, signed, region=self.region)
        centroid = stats.centroid
        norm = localize(stats, frame.shape)
        touch = bool(stats.touch) and stats.peak_mag >= self.min_peak
        return TouchResult(
            touch=touch,
            area_px=stats.area_px,
            area_frac=stats.area_frac,
            centroid=centroid,
            normalized_centroid=norm,
            grid_cell=grid_cell(norm),
            force=stats.force,
            peak_mag=stats.peak_mag,
            n_blobs=stats.n_blobs,
        )


# ---------------------------------------------------------------------------
# feature extraction for the learned classifier


def _hu_features(mask: np.ndarray) -> np.ndarray:
    moments = cv2.moments(mask, binaryImage=True)
    hu = cv2.HuMoments(moments).flatten().astype(np.float64)
    # log-scale so the huge dynamic range does not dominate
    return (-np.sign(hu) * np.log10(np.abs(hu) + 1e-30)).astype(np.float32)


def extract_features(
    frame: np.ndarray,
    reference: np.ndarray,
    region: Optional[np.ndarray] = None,
    grid: Tuple[int, int] = (12, 12),
    threshold: float = 10.0,
    min_area: int = 40,
) -> np.ndarray:
    """Hand-crafted feature vector for one frame against a reference.

    Components:

    * ``grid`` x ``grid`` pooled change magnitude, L2-normalised  (144 dims)
    * global magnitude mean / max / std                            (3 dims)
    * contact mask geometry: area fraction, bbox aspect, centroid  (3 dims)
    * log Hu moments of the contact mask                           (7 dims)
    * mean BGR of the frame and of the change inside the mask      (6 dims)

    The vector is deterministic and cheap enough for live use.
    """
    signed = frame.astype(np.float32) - reference.astype(np.float32)
    mag = magnitude(signed)
    if region is not None:
        mag = mag * (region > 0)

    pooled = cv2.resize(mag, grid, interpolation=cv2.INTER_AREA).reshape(-1)
    pooled = pooled / (np.linalg.norm(pooled) + 1e-9)

    mask = contact_mask(signed, threshold=threshold, min_area=min_area, region=region)
    stats = contact_stats(mask, signed, region=region)

    geom = [float(stats.area_frac), 0.0, 0.0]
    if stats.bbox is not None:
        _, _, w, h = stats.bbox
        geom[1] = float(w) / max(1.0, float(h))  # aspect ratio
    if stats.centroid is not None:
        height, width = frame.shape[:2]
        geom[2] = float(stats.centroid[0] / width)

    hu = _hu_features(mask) if mask.any() else np.zeros(7, np.float32)

    frame_mean = frame.reshape(-1, frame.shape[-1]).mean(axis=0) / 255.0
    if mask.any():
        diff_inside = signed[mask > 0].mean(axis=0) / 255.0
    else:
        diff_inside = np.zeros(3, np.float32)

    features = np.concatenate(
        [
            pooled.astype(np.float32),
            np.array([mag.mean(), mag.max(), mag.std()], np.float32) / 255.0,
            np.array(geom, np.float32),
            hu.astype(np.float32),
            frame_mean.astype(np.float32),
            diff_inside.astype(np.float32),
        ]
    )
    return np.nan_to_num(features)


@dataclass
class FeatureExtractor:
    """Configurable :func:`extract_features` bound to a reference + region."""

    reference: np.ndarray
    region: Optional[np.ndarray] = None
    grid: Tuple[int, int] = (12, 12)
    threshold: float = 10.0
    min_area: int = 40

    def __post_init__(self) -> None:
        if self.region is None:
            from .processing import active_region

            self.region = active_region(self.reference)

    def extract(self, frame: np.ndarray) -> np.ndarray:
        return extract_features(
            frame,
            self.reference,
            region=self.region,
            grid=self.grid,
            threshold=self.threshold,
            min_area=self.min_area,
        )

    @property
    def dim(self) -> int:
        return int(self.extract(np.zeros_like(self.reference)).shape[0])


# ---------------------------------------------------------------------------
# learned classifier


class DigitClassifier:
    """scikit-learn classifier over :func:`extract_features` vectors.

    ``method`` is one of ``"knn"``, ``"svm"`` or ``"logreg"``.  Feature
    standardisation is fitted with the model and stored in the pickle.
    """

    def __init__(self, method: str = "knn", **kwargs) -> None:
        self.method = method
        self.kwargs = kwargs
        self.model = None
        self.scaler = None
        self.classes_: List[str] = []

    def _make_model(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC

        if self.method == "knn":
            return KNeighborsClassifier(n_neighbors=self.kwargs.get("n_neighbors", 3), weights="distance")
        if self.method == "svm":
            return SVC(
                C=self.kwargs.get("C", 3.0),
                kernel=self.kwargs.get("kernel", "rbf"),
                probability=True,
                random_state=0,
            )
        if self.method == "logreg":
            return LogisticRegression(max_iter=1000, C=self.kwargs.get("C", 1.0))
        raise ValueError(f"unknown method {self.method!r} (use knn, svm or logreg)")

    def fit(self, X: np.ndarray, y: Sequence[str]) -> "DigitClassifier":
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, np.float32)
        y = np.asarray(list(y))
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        self.classes_ = sorted(set(y.tolist()))
        self.scaler = StandardScaler().fit(X)
        self.model = self._make_model()
        self.model.fit(self.scaler.transform(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("classifier is not fitted")
        X = np.atleast_2d(np.asarray(X, np.float32))
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None or not hasattr(self.model, "predict_proba"):
            return None
        X = np.atleast_2d(np.asarray(X, np.float32))
        return self.model.predict_proba(self.scaler.transform(X))

    def score(self, X: np.ndarray, y: Sequence[str]) -> float:
        if self.model is None:
            raise RuntimeError("classifier is not fitted")
        X = np.atleast_2d(np.asarray(X, np.float32))
        return float(self.model.score(self.scaler.transform(X), np.asarray(list(y))))

    def predict_one(self, features: np.ndarray) -> Tuple[str, float]:
        label = str(self.predict(features)[0])
        proba = self.predict_proba(features)
        conf = 1.0
        if proba is not None:
            conf = float(np.max(proba))
        return label, conf

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "method": self.method,
                    "kwargs": self.kwargs,
                    "model": self.model,
                    "scaler": self.scaler,
                    "classes": self.classes_,
                },
                fh,
            )

    @classmethod
    def load(cls, path: str) -> "DigitClassifier":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        obj = cls(method=blob.get("method", "knn"), **blob.get("kwargs", {}))
        obj.model = blob["model"]
        obj.scaler = blob["scaler"]
        obj.classes_ = blob.get("classes", [])
        return obj


# ---------------------------------------------------------------------------
# dataset helpers


@dataclass
class FeatureDataset:
    X: np.ndarray
    y: List[str]

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, np.float32)
        self.y = [str(v) for v in self.y]

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def add(self, features: np.ndarray, label: str) -> None:
        self.X = np.vstack([self.X, np.asarray(features, np.float32).reshape(1, -1)])
        self.y.append(str(label))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, X=self.X, y=np.asarray(self.y, dtype=object))

    @classmethod
    def load(cls, path: str) -> "FeatureDataset":
        blob = np.load(path, allow_pickle=True)
        return cls(blob["X"], list(blob["y"]))

    def summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for label in self.y:
            out[label] = out.get(label, 0) + 1
        return out


def evaluate(
    X: np.ndarray,
    y: Sequence[str],
    method: str = "knn",
    test_size: float = 0.35,
    seed: int = 0,
    **kwargs,
) -> Dict[str, object]:
    """Train/test split evaluation returning accuracy and confusion matrix.

    Small datasets are common here, so the split is stratified when possible.
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split

    X = np.asarray(X, np.float32)
    y = np.asarray(list(y))
    stratify = y if len(set(y.tolist())) > 1 else None
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=stratify
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    clf = DigitClassifier(method=method, **kwargs).fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)
    pred = clf.predict(X_te)
    labels = sorted(set(y.tolist()))
    cm = confusion_matrix(y_te, pred, labels=labels).tolist()
    return {
        "method": method,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "accuracy": float(acc),
        "labels": labels,
        "confusion": cm,
        "classifier": clf,
    }
