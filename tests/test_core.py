import cv2
import numpy as np
import pytest

from lili_shap.core import deletion_curve, kernel_shap, mean_mask, removal_mask, sample_coalitions, summarize


def test_seeded_coalitions_and_endpoint_constrained_additive_recovery():
    z = sample_coalitions(16)
    assert z.shape == (128, 16)
    assert len(np.unique(z, axis=0)) == 128
    assert np.array_equal(z, sample_coalitions(16))
    assert np.count_nonzero(z.sum(1) == 1) == 16
    assert np.count_nonzero(z.sum(1) == 15) == 16
    expected = np.linspace(-0.2, 0.4, 16)
    phi, fit = kernel_shap(z, 0.12 + z @ expected)
    np.testing.assert_allclose(phi, expected, atol=1e-12)
    assert abs(fit['efficiency_residual']) < 1e-12


def test_complete_game_recovers_interaction_shap():
    z = np.array([[int(bit) for bit in f'{i:03b}'] for i in range(8)])
    phi, _ = kernel_shap(z, 0.7*z[:, 0]*z[:, 1] + 0.2*z[:, 2])
    np.testing.assert_allclose(phi, [0.35, 0.35, 0.2], atol=1e-12)


def test_mask_polarity_square_expansion_and_global_mean():
    segments = np.zeros((9, 9), dtype=int)
    segments[4, 4] = 1
    mask = removal_mask(segments, [1, 0])
    expanded = removal_mask(segments, [1, 0], 3)
    assert np.count_nonzero(mask) == 1
    assert np.count_nonzero(expanded) == 49
    image = np.arange(243, dtype=np.uint8).reshape(9, 9, 3)
    result = mean_mask(image, mask)
    np.testing.assert_array_equal(result[4, 4], np.rint(image.mean((0, 1))).astype(np.uint8))
    np.testing.assert_array_equal(result[mask == 0], image[mask == 0])


def test_deletion_uses_telea_positive_order_and_area(monkeypatch):
    segments = np.zeros((10, 10), dtype=int)
    segments[:, :2] = 1
    segments[:, 2:5] = 2
    image = np.full((10, 10, 3), 100, dtype=np.uint8)
    calls = []
    def fake_inpaint(original, mask, radius, flag):
        calls.append((mask.copy(), radius, flag))
        out = original.copy()
        out[mask > 0] = 0
        return out
    monkeypatch.setattr(cv2, 'inpaint', fake_inpaint)
    curve = deletion_curve(image, segments, np.array([-1., 0.2, 0.5]),
                           lambda images: np.array([x.mean()/100 for x in images]))
    assert curve['order'] == [2, 1]
    np.testing.assert_allclose(curve['fractions'], [0, 0.3, 0.5])
    assert all(radius == 3 and flag == cv2.INPAINT_TELEA for _,radius,flag in calls)
    assert curve['auc'] == pytest.approx(0.3*0.85 + 0.2*0.6 + 0.5*0.5)


def test_no_positive_features_and_zero_difference_statistics():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    curve = deletion_curve(image, np.zeros((4, 4), dtype=int), [-1], lambda images: [0.4])
    assert curve['order'] == []
    assert curve['auc'] == pytest.approx(0.4)
    rows = [{'mean_shap_deletion_auc': .4, 'lama_shap_deletion_auc': .4,
             'lama_expanded_shap_deletion_auc': .4} for _ in range(3)]
    result = summarize(rows)
    assert result['wilcoxon']['mean_minus_lama']['pvalue'] == 1
