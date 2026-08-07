"""
test_lookup_interpolation.py
============================
Portability gate for the SLR interpolation kernel.

The damage cube is ``float32`` while the SLR grid is ``float64``.  The linear
kernel used to delegate to SciPy's legacy ``interp1d``, which meant the result
depended on where SciPy promoted the mixed dtypes.  That promotion is an
implementation detail of a deprecated API and is not stable across SciPy
builds or NumPy promotion regimes (NEP 50).  Because interpolated damages
routinely land within one ``float32`` unit in the last place of a rounding
boundary, a promotion difference flipped stored damages by one ulp and broke
the golden bit-parity regression on some platforms while passing on others.

``_linear_at_slr`` pins every intermediate dtype so the kernel depends only on
IEEE-754 add/subtract/multiply/divide, which are exactly rounded and therefore
identical on every platform.  These tests pin that contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from floodadapt_abm._core.lookup_utils import _linear_at_slr, interpolate_cube_at_slr


# A tiny cube with exactly representable values: (1 agent, 3 SLR, 2 events).
CUBE = np.array([[[10.0, 3.0], [20.0, 7.0], [50.0, 9.0]]], dtype=np.float32)
GRID = np.array([0.0, 1.0, 3.0], dtype=np.float64)


def test_grid_points_are_reproduced_exactly() -> None:
    """Interpolating at a grid point returns that grid point's values."""
    for k, slr in enumerate(GRID):
        got = _linear_at_slr(CUBE, GRID, float(slr))
        assert got.tolist() == CUBE[:, k, :].astype(np.float64).tolist()


@pytest.mark.parametrize(
    "slr, expected",
    [
        (0.5, [[15.0, 5.0]]),      # midpoint of the first interval
        (2.0, [[35.0, 8.0]]),      # midpoint of the second interval
        (-1.0, [[0.0, -1.0]]),     # extrapolated below the grid
        (4.0, [[65.0, 10.0]]),     # extrapolated above the grid
    ],
)
def test_pinned_interpolation_values(slr: float, expected: list) -> None:
    """Exact expected values, including extrapolation off both ends."""
    assert _linear_at_slr(CUBE, GRID, slr).tolist() == expected


def test_output_is_float64_regardless_of_cube_dtype() -> None:
    """The affine step runs in float64 whatever the cube dtype is."""
    assert _linear_at_slr(CUBE, GRID, 0.5).dtype == np.float64
    assert _linear_at_slr(CUBE.astype(np.float64), GRID, 0.5).dtype == np.float64


def test_descending_grid_is_sorted_internally() -> None:
    """A reversed SLR axis yields the same answer as an ascending one."""
    got = _linear_at_slr(CUBE[:, ::-1, :], GRID[::-1], 0.5)
    assert got.tolist() == _linear_at_slr(CUBE, GRID, 0.5).tolist()


def test_single_grid_point_is_rejected() -> None:
    """Linear interpolation needs a bracketing interval."""
    with pytest.raises(ValueError, match="at least 2 SLR grid points"):
        _linear_at_slr(CUBE[:, :1, :], GRID[:1], 0.0)


def test_result_is_independent_of_grid_dtype() -> None:
    """A float32 SLR grid must not change the pinned float64 arithmetic."""
    got32 = _linear_at_slr(CUBE, GRID.astype(np.float32), 0.5)
    assert got32.tolist() == _linear_at_slr(CUBE, GRID, 0.5).tolist()


def test_repeated_calls_are_bit_identical() -> None:
    """No accumulation or caching effect between calls."""
    first = _linear_at_slr(CUBE, GRID, 0.37)
    for _ in range(5):
        assert np.array_equal(_linear_at_slr(CUBE, GRID, 0.37), first)


def test_cube_wrapper_clamps_to_the_damage_ceiling() -> None:
    """``interpolate_cube_at_slr`` clips the interpolated cube to [0, max_pot_dmg]."""
    out = interpolate_cube_at_slr(
        CUBE, GRID, 4.0, method="linear", max_pot_dmg=np.array([40.0])
    )
    assert out.tolist() == [[40.0, 10.0]]          # 65 clipped to 40, 10 kept

    out_neg = interpolate_cube_at_slr(CUBE, GRID, -1.0, method="linear")
    assert out_neg.tolist() == [[0.0, 0.0]]        # -1 clipped up to 0


def test_cube_wrapper_is_float32_without_a_ceiling() -> None:
    """
    The interpolated cube is stored at float32.

    With a ``max_pot_dmg`` ceiling the final ``np.clip`` promotes to the
    ceiling's dtype; that widening is lossless and is the historical
    behaviour, so it is pinned here rather than "fixed".
    """
    assert interpolate_cube_at_slr(CUBE, GRID, 0.5, method="linear").dtype == np.float32
    with_ceiling = interpolate_cube_at_slr(
        CUBE, GRID, 0.5, method="linear", max_pot_dmg=np.array([1e9])
    )
    assert with_ceiling.dtype == np.float64
    assert with_ceiling.tolist() == [[15.0, 5.0]]


def test_linear_path_does_not_need_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The linear kernel must not import SciPy.

    This is the structural half of the portability guarantee: if the linear
    branch ever delegates to SciPy again, the platform-dependent promotion
    comes back with it.
    """
    import builtins

    real_import = builtins.__import__

    def guard(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("scipy"):
            raise AssertionError(f"linear interpolation imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    interpolate_cube_at_slr(CUBE, GRID, 0.5, method="linear")
