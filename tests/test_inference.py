import numpy as np

from nufield.inference.kernel_posterior import KernelPosterior


def test_kernel_posterior_shape_and_finiteness():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 6))
    theta = rng.uniform(size=(30, 3))
    model = KernelPosterior(bandwidth=1.2, top_k=12).fit(x[:25], theta[:25])
    result = model.summarize(x[26], n_samples=100, rng=rng)
    assert result.samples.shape == (100, 3)
    assert result.point_estimate.shape == (3,)
    assert np.isfinite(result.samples).all()
    assert result.effective_sample_size >= 1
