import numpy as np

from nufield.units import k_hmpc_to_inv_mpc, power_mpc3_to_mpc_over_h_cubed


def test_unit_conversions_are_explicit():
    assert np.allclose(k_hmpc_to_inv_mpc(np.array([1.0, 2.0]), 0.7), [0.7, 1.4])
    assert np.allclose(power_mpc3_to_mpc_over_h_cubed(np.array([10.0]), 0.5), [1.25])
