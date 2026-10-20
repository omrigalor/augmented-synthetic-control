import numpy as np
from model import fit, in_space_placebos


def test_recovers_known_post_treatment_effect():
    t = np.arange(20)
    donors = np.vstack([t, 0.5 * t + 2, np.sin(t / 3)])
    treated = 0.6 * donors[0] + 0.4 * donors[1]
    treated[12:] += 5
    result = fit(treated, donors, 12)
    assert np.isclose(result.weights.sum(), 1)
    assert np.all(result.weights >= 0)
    assert abs(result.effect[12:].mean() - 5) < 0.1


def test_placebo_p_value_is_bounded():
    panel = np.vstack([np.arange(16) + i for i in range(5)]).astype(float)
    panel[0, 10:] += 3
    _, p = in_space_placebos(panel, 0, 10)
    assert 0 <= p <= 1
