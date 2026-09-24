"""Shared pytest fixtures. Everything here is sensor-free."""

import numpy as np
import pytest

from digit import synth


@pytest.fixture(scope="session")
def reference() -> np.ndarray:
    return synth.synthetic_reference(seed=0)


@pytest.fixture
def center_contact(reference):
    h, w = reference.shape[:2]
    frame, mask = synth.synth_frame(reference, kind="circle", center=(w / 2, h / 2), amplitude=-70.0)
    return frame, mask


@pytest.fixture
def offset_contact(reference):
    h, w = reference.shape[:2]
    frame, mask = synth.synth_frame(reference, kind="circle", center=(0.7 * w, 0.3 * h), amplitude=-70.0)
    return frame, mask
