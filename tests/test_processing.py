"""Sensor-free tests for digit.processing."""

import numpy as np

from digit import processing as P


def test_difference_of_identical_frames_is_zero(reference):
    visual, signed = P.difference(reference, reference, gain=3.0)
    assert np.allclose(signed, 0.0, atol=1e-6)
    assert np.all(visual == 128)


def test_magnitude_matches_channels():
    signed = np.zeros((4, 4, 3), np.float32)
    signed[1, 1] = [3.0, -9.0, 4.0]
    mag = P.magnitude(signed)
    assert mag[1, 1] == 9.0
    assert mag.sum() == 9.0


def test_contact_mask_finds_synthetic_contact(reference, center_contact):
    frame, _ = center_contact
    signed = frame.astype(np.float32) - reference.astype(np.float32)
    mask = P.contact_mask(signed, threshold=10.0, min_area=40, region=P.active_region(reference))
    stats = P.contact_stats(mask, signed)
    assert stats.touch
    assert stats.area_px > 100
    # centroid is close to the image center
    h, w = reference.shape[:2]
    assert abs(stats.centroid[0] - w / 2) < 10
    assert abs(stats.centroid[1] - h / 2) < 10


def test_contact_mask_empty_for_no_contact(reference):
    signed = np.zeros_like(reference, dtype=np.float32)
    mask = P.contact_mask(signed, threshold=10.0, min_area=40)
    assert not mask.any()


def test_centroid_tracks_offset_contact(reference, offset_contact):
    frame, _ = offset_contact
    detector_signed = frame.astype(np.float32) - reference.astype(np.float32)
    mask = P.contact_mask(detector_signed, threshold=10.0, min_area=40)
    stats = P.contact_stats(mask, detector_signed)
    h, w = reference.shape[:2]
    assert abs(stats.centroid[0] - 0.7 * w) < 15
    assert abs(stats.centroid[1] - 0.3 * h) < 15


def test_active_region_covers_most_of_synthetic_reference(reference):
    region = P.active_region(reference)
    assert (region > 0).mean() > 0.5
    bbox = P.region_bbox(region)
    assert bbox is not None and bbox[2] > 0.7 * reference.shape[1]


def test_poisson_integrates_a_gaussian_height_field():
    h, w = 64, 80
    yy, xx = np.mgrid[0:h, 0:w]
    height = 20.0 * np.exp(-(((xx - w / 2) ** 2) + ((yy - h / 2) ** 2)) / 200.0)
    gy, gx = np.gradient(height)
    recovered = P.poisson_dct_neumann(gx, gy)
    r0 = recovered - recovered.mean()
    h0 = height - height.mean()
    corr = np.corrcoef(r0.ravel(), h0.ravel())[0, 1]
    assert corr > 0.9


def test_depth_proxy_returns_consistent_shapes(reference, center_contact):
    frame, _ = center_contact
    depth, normals, height = P.depth_proxy(frame, reference)
    assert depth.shape == reference.shape[:2]
    assert normals.shape == (reference.shape[0], reference.shape[1], 3)
    assert height.shape == reference.shape[:2]
    nz = normals[..., 2]
    assert np.allclose(np.linalg.norm(normals, axis=2), 1.0, atol=1e-3)
    assert (nz > 0).all()


def test_grid_cell_mapping():
    assert P.grid_cell((0.1, 0.1)) == "top-left"
    assert P.grid_cell((0.5, 0.5)) == "middle-center"
    assert P.grid_cell((0.9, 0.9)) == "bottom-right"
    assert P.grid_cell(None) is None


def test_tactile_processor_bundle(reference, center_contact):
    frame, _ = center_contact
    processor = P.TactileProcessor(reference, threshold=10.0, min_area=40)
    result = processor.process(frame)
    assert result.touch
    assert result.normalized_centroid is not None
    assert result.depth is not None
    assert result.grid_cell == "middle-center"
