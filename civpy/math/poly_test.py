import pytest
import numpy as np
from .poly import *


def test_polyfit():
    p = np.array([25, -89.743, 109, -93])
    x = np.linspace(-100, 100, 100)
    y = polyval(p, x)

    a = polyfit(x, y, 3)

    assert pytest.approx(a, 0.01) == p


def test_polyval():
    p = np.array([10, -298, 34.7])
    x = np.linspace(-100, 100, 100)
    y = 10 - 298 * x + 34.7 * x**2

    a = polyval(p, x)

    assert pytest.approx(a, 0.01) == y


def test_polyderiv():
    p = np.array([10, -298, 34.7])
    x = np.linspace(-100, 100, 100)
    yp = -298 + 2 * 34.7 * x
    ypp = 2 * 34.7

    a = polyderiv(p, x)
    assert pytest.approx(a, 0.01) == yp

    a = polyderiv(p, x, 2)
    assert pytest.approx(a, 0.01) == ypp
