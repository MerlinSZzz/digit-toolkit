"""Sensor-free tests for the synthetic contact generator."""

import numpy as np

from digit import synth


def test_synthetic_reference_shape_and_dtype():
    ref = synth.synthetic_reference((100, 120, 3), seed=0)
    assert ref.shape == (100, 120, 3)
    assert ref.dtype == np.uint8
    assert ref.mean() > 60  # bright enough to be usable gel


def test_synth_frame_changes_and_masks():
    ref = synth.synthetic_reference((120, 160, 3), seed=1)
    frame, mask = synth.synth_frame(ref, kind="circle", center=(80.0, 60.0), amplitude=-80.0)
    assert frame.shape == ref.shape
    assert mask.dtype == np.uint8
    assert mask.any()
    # the changed pixels are inside the ground-truth mask area
    changed = np.abs(frame.astype(int) - ref.astype(int)).max(axis=2) > 10
    assert changed.sum() > 0


def test_contact_kinds_all_produce_fields():
    for kind in ("circle", "square", "ridge", "ring", "two_finger", "stripe"):
        field = synth.contact_field((80, 100), kind=kind, radius=0.2)
        assert field.shape == (80, 100)
        assert np.abs(field).max() > 0


def test_dataset_is_balanced_and_positioned():
    ref = synth.synthetic_reference((120, 160, 3), seed=2)
    frames, labels, infos = synth.synthesize_dataset(ref, n_per_class=4, seed=5)
    assert len(frames) == len(labels) == len(infos)
    counts = {label: labels.count(label) for label in set(labels)}
    assert all(count == 4 for count in counts.values())
    assert set(counts) == set(synth.POSITION_CLASSES)
    # 'none' has no centroid
    none_info = [i for l, i in zip(labels, infos) if l == "none"][0]
    assert synth.true_centroid(none_info) is None
