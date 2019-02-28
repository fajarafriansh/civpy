import pytest
import numpy as np
from .catenary import *


def test_belly_length():
    a = belly_length(1000, 500)
    assert pytest.approx(a) == 250

    a = belly_length(1000, 500, 10)
    b = 250 - 1000 * np.arcsinh(5/(1000*np.sinh(0.25)))
    assert pytest.approx(a, 0.01) == b


def test_sag():
    a = sag([0, 0.5, 1], 1000, 500)
    b = np.array([0, 1000 * np.cosh(0.25) - 1000, 0])

    assert pytest.approx(a, 0.01) == b

    a = sag([0, 0.25, 0.5, 0.75, 1], 1000, 500, 10)
    b = sag([1, 0.75, 0.5, 0.25, 0], 1000, 500, -10)

    assert pytest.approx(a, 0.01) == b


def test_position():
    a = position([0, 0.5, 1], 1000, 500).ravel()
    b = np.array([
        [0, 0, 0],
        [250, 0, -1000 * np.cosh(0.25) + 1000],
        [500, 0, 0]
    ]).ravel()

    assert pytest.approx(a, 0.01) == b

    a = position(0.5, 1000, 500)
    b = np.array([250, 0, -1000 * np.cosh(0.25) + 1000])

    assert pytest.approx(a, 0.01) == b


def test_stressed_length():
    a = stressed_length(1000, 500, 10)
    b = belly_length(1000, 500, 10)
    b = 1000 * (np.sinh(b / 1000) + np.sinh((500 - b) / 1000))

    assert pytest.approx(a, 0.01) == b


def test_average_tension_1():
    a = average_tension_1(1000, 2, 500)

    b = np.linspace(0, 250, 1000)
    b = 1000 * 2 * np.mean(np.cosh(b / 1000))

    assert pytest.approx(a, 0.01) == b


def test_max_tension():
    a = max_tension(1000, 8, 500, 10)

    assert pytest.approx(a, 0.01) == 8292.92


def test_catenary_const_1():
    a = catenary_const_1(1000, 8)
    b = 1000 / 8

    assert pytest.approx(a, 0.01) == b


def test_catenary_const_2():
    a = stressed_length(1000, 500, 10)
    a = catenary_const_2(a, 500, 10)

    assert pytest.approx(a, 0.01) == 1000


def test_catenary_const_3():
    a = position(0.5, 1000, 500, 10, np.pi/4)
    a = catenary_const_3(a[0], a[1], a[2], 500, 10)
    b = np.array([1000, np.pi/4])

    assert pytest.approx(a, 0.01) == b


def test_catenary_const_4():
    a = max_tension(1000, 8, 500, 10)
    a = catenary_const_4(a, 8, 500, 10)

    assert pytest.approx(a, 0.01) == 1000


def test_plot_catenary_2d():
    plot_catenary_2d(1000, 500)


def test_plot_catenary_3d():
    plot_catenary_3d(1000, 500)
