"""The long-run comparison's yardstick: one-ulp perturbations and relative distances."""

import numpy as np
from pjrt_long_run import distance, perturbed


def test_perturbation_moves_every_element_by_one_ulp():
    params = {"w": np.array([1.0, -2.5, 0.0, 3e-38], np.float32), "b": np.float32(7)}
    for direction in (1, -1, 0):
        moved = perturbed(params, direction, seed=3)
        for name, value in params.items():
            before, after = np.asarray(value), np.asarray(moved[name])
            assert after.dtype == np.float32
            steps = np.abs(after.view(np.int32) - before.view(np.int32))
            # Zero's neighbours are the smallest subnormals of either sign.
            assert np.all((steps == 1) | ((before == 0) & (np.abs(after) == np.float32(1e-45))))
    up, down = perturbed(params, 1, 0)["w"], perturbed(params, -1, 0)["w"]
    assert np.all(up > params["w"]) and np.all(down < params["w"])


def test_distance_is_relative_over_all_leaves():
    reference = {"a": np.array([3.0, 0.0]), "b": np.array([4.0])}
    assert distance(reference, reference) == 0
    assert distance({"a": np.array([3.0, 0.0]), "b": np.array([5.0])}, reference) == 0.2
