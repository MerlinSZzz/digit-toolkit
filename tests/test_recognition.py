"""Sensor-free tests for recognition (touch, localisation, classifier)."""

import numpy as np

from digit import recognition as R
from digit import synth


def test_touch_detector_no_touch(reference):
    detector = R.TouchDetector(reference)
    result = detector.detect(reference.copy())
    assert not result.touch
    assert result.area_px == 0
    assert result.centroid is None


def test_touch_detector_and_localisation(reference, offset_contact):
    frame, _ = offset_contact
    detector = R.TouchDetector(reference, threshold=10.0, min_area=40)
    result = detector.detect(frame)
    h, w = reference.shape[:2]
    assert result.touch
    assert result.normalized_centroid is not None
    assert abs(result.normalized_centroid[0] - 0.7) < 0.05
    assert abs(result.normalized_centroid[1] - 0.3) < 0.05
    assert result.grid_cell is not None


def test_feature_extractor_is_deterministic(reference, center_contact):
    frame, _ = center_contact
    extractor = R.FeatureExtractor(reference)
    a = extractor.extract(frame)
    b = extractor.extract(frame)
    assert a.shape == b.shape
    assert np.allclose(a, b)
    assert extractor.dim == a.shape[0]


def test_synthetic_dataset_positions_are_distinct(reference):
    frames, labels, infos = synth.synthesize_dataset(
        reference, classes=synth.POSITION_CLASSES, n_per_class=3, seed=1
    )
    assert len(frames) == len(labels) == len(infos)
    for label in ("top-left", "bottom-right", "center"):
        centers = [synth.true_centroid(i) for l, i in zip(labels, infos) if l == label]
        h, w = reference.shape[:2]
        if label == "top-left":
            assert all(c[0] < w / 2 and c[1] < h / 2 for c in centers)
        if label == "bottom-right":
            assert all(c[0] > w / 2 and c[1] > h / 2 for c in centers)


def test_classifier_learns_synthetic_shapes(reference):
    frames, labels, _ = synth.synthesize_dataset(
        reference, classes=synth.SHAPE_CLASSES, n_per_class=8, seed=2
    )
    extractor = R.FeatureExtractor(reference)
    X = np.stack([extractor.extract(f) for f in frames])
    report = R.evaluate(X, labels, method="svm", test_size=0.35, seed=0)
    assert report["accuracy"] >= 0.8
    assert "none" in report["labels"]


def test_classifier_save_load_roundtrip(reference, tmp_path):
    frames, labels, _ = synth.synthesize_dataset(reference, n_per_class=4, seed=3)
    extractor = R.FeatureExtractor(reference)
    X = np.stack([extractor.extract(f) for f in frames])
    clf = R.DigitClassifier(method="knn").fit(X, labels)
    path = str(tmp_path / "model.pkl")
    clf.save(path)
    loaded = R.DigitClassifier.load(path)
    assert loaded.predict(X[:2]).tolist() == clf.predict(X[:2]).tolist()


def test_feature_dataset_roundtrip(reference, tmp_path):
    frames, labels, _ = synth.synthesize_dataset(reference, n_per_class=2, seed=4)
    extractor = R.FeatureExtractor(reference)
    X = np.stack([extractor.extract(f) for f in frames])
    dataset = R.FeatureDataset(X, labels)
    path = str(tmp_path / "data.npz")
    dataset.save(path)
    loaded = R.FeatureDataset.load(path)
    assert len(loaded) == len(dataset)
    assert sorted(loaded.y) == sorted(dataset.y)
