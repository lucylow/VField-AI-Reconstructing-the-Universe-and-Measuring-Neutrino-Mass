from nufield.randomness import rng_for_id


def test_rng_is_deterministic_per_id():
    a = rng_for_id(10, "sim-1").normal(size=8)
    b = rng_for_id(10, "sim-1").normal(size=8)
    c = rng_for_id(10, "sim-2").normal(size=8)
    assert (a == b).all()
    assert not (a == c).all()
