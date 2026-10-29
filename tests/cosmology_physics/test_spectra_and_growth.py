import numpy as np
import pytest

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.growth import growth_factor_scale_independent, growth_rate_f
from nufield.cosmology.physics.linear_regime import first_k_crossing, linear_mode_mask
from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum, top_hat_window
from nufield.cosmology.physics.transfer import effective_log_slope


def make_spectrum():
    k = np.geomspace(1e-3, 10, 300)
    p = 1e4 * k**0.96 / (1 + (k / 0.18) ** 2) ** 2
    return MatterPowerSpectrum(k, p, 0.0, True, "test")


def test_spectrum_units_and_sigma_r_are_valid():
    s = make_spectrum()
    assert np.all(s.dimensionless_power() > 0)
    assert s.sigma_r(8) > 0
    assert s.sigma8(h=0.7) > 0
    kh, ph = s.to_h_units(0.7)
    assert np.allclose(kh * 0.7, s.k_mpc_inv)
    assert np.allclose(ph / 0.7**3, s.pk_mpc3)


def test_top_hat_window_series_and_large_x():
    assert top_hat_window(0.0) == pytest.approx(1.0)
    assert top_hat_window(1e-6) == pytest.approx(1.0, abs=1e-10)
    assert np.isfinite(top_hat_window(np.array([0.0, 1.0, 100.0]))).all()


def test_spectrum_rejects_invalid_support_and_extrapolation():
    with pytest.raises(ValueError):
        MatterPowerSpectrum(np.array([1, 1, 2]), np.array([1, 2, 3]), 0)
    s = make_spectrum()
    with pytest.raises(ValueError, match="outside"):
        s.evaluate(1e-4)


def test_massive_neutrino_growth_requires_explicit_approximation():
    bg = BackgroundCosmology(PhysicalCosmology())
    with pytest.raises(ValueError, match="scale-independent growth omits"):
        growth_factor_scale_independent(bg, [0.0, 1.0])
    d = growth_factor_scale_independent(
        bg, [0.0, 0.5, 1.0], allow_massive_neutrino_approximation=True
    )
    f = growth_rate_f(bg, [0.0, 0.5, 1.0], allow_massive_neutrino_approximation=True)
    assert d[0] == pytest.approx(1.0)
    assert d[0] > d[1] > d[2] > 0
    assert np.all(f > 0)


def test_zero_mass_growth_is_normalized_and_monotonic():
    bg = BackgroundCosmology(PhysicalCosmology(sum_mnu_ev=0.0))
    d = growth_factor_scale_independent(bg, np.array([0, 1, 2]))
    assert d[0] == pytest.approx(1.0)
    assert d[0] > d[1] > d[2]


def test_linear_regime_masks_and_spectral_slope():
    s = make_spectrum()
    mask = linear_mode_mask(s, delta2_max=10)
    assert mask.shape == s.k_mpc_inv.shape
    assert mask.dtype == bool
    crossing = first_k_crossing(s, 0.1)
    assert crossing is None or crossing > 0
    slope = effective_log_slope(s.k_mpc_inv, s.pk_mpc3)
    assert slope.shape == s.pk_mpc3.shape
    assert np.all(np.isfinite(slope))
