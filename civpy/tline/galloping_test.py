import pytest
import numpy as np
from .galloping import *


def test_loop_factor():
    # Test single loop
    a = np.array([loop_factor(x, 1) for x in (0, 0.5, 1)])
    b = np.array([0, 1, 0])

    assert pytest.approx(a, 0.01) == b

    # Test double loop
    a = np.array([loop_factor(x, 2) for x in (0, 0.25, 0.5, 0.25, 1)])
    b = np.array([0, 1, 0, 1, 0])

    assert pytest.approx(a, 0.01) == b


def test_single_loop_davison():
    a = single_loop_davison(7.9*0.3048, 9.5*np.pi/180)
    r = (a.x**2 + a.y**2)**0.5
    b = 0.5 * a.height - r

    assert a.x > 0
    assert a.y > 0
    assert pytest.approx(b, 0.02) == 2*0.3048
    assert pytest.approx(a.width, 0.02) == 4.3*0.3048
    assert pytest.approx(a.height, 0.02) == 10.9*0.3048
    assert pytest.approx(a.rotation, 0.02) == -0.5*9.5*np.pi/180

    a = single_loop_davison(7.9*0.3048, -9.5*np.pi/180)

    assert a.x < 0
    assert a.y > 0
    assert pytest.approx(a.rotation, 0.02) == 0.5*9.5*np.pi/180


def test_single_loop_cigre():
    # Single wire
    a = single_loop_cigre(5.72*0.3048, 0.0281432, 1)
    b = 0.5 * a.height - a.y

    assert a.x == 0
    assert pytest.approx(b, 0.02) == 5.1*0.3048
    assert pytest.approx(a.height, 0.02) == 16.9*0.3048
    assert pytest.approx(a.width, 0.02) == 6.8*0.3048

    # Bundled wire
    a = single_loop_cigre(20.65*0.3048, 0.0281432, 2)
    b = 0.5 * a.height - a.y

    assert a.x == 0
    assert pytest.approx(b, 0.02) == 6*0.3048
    assert pytest.approx(a.height, 0.02) == 20*0.3048
    assert pytest.approx(a.width, 0.02) == 8*0.3048


def test_single_loop_aeso():
    # Single wire
    a1, a2 = single_loop_aeso(5.72*0.3048, 0.0281432, 1)
    b = 0.5 * a1.height - a1.y

    assert a1.x == 0
    assert pytest.approx(b, 0.02) == 4.2*0.3048
    assert pytest.approx(a1.height, 0.02) == 16.9*0.3048
    assert pytest.approx(a1.width, 0.02) == 3.4*0.3048
    assert a1.rotation == 5*np.pi/180

    b = 0.5 * a2.height - a2.y

    assert a2.x == 0
    assert pytest.approx(b, 0.02) == 4.2*0.3048
    assert pytest.approx(a2.height, 0.02) == 16.9*0.3048
    assert pytest.approx(a2.width, 0.02) == 3.4*0.3048
    assert a2.rotation == -5*np.pi/180

    # Bundled wire
    a1, a2 = single_loop_aeso(20.65*0.3048, 0.0281432, 2)
    b = 0.5 * a1.height - a1.y

    assert a1.x == 0
    assert pytest.approx(b, 0.02) == 5*0.3048
    assert pytest.approx(a1.height, 0.02) == 20*0.3048
    assert pytest.approx(a1.width, 0.02) == 4*0.3048
    assert a1.rotation == 5*np.pi/180

    b = 0.5 * a2.height - a2.y

    assert a2.x == 0
    assert pytest.approx(b, 0.02) == 5*0.3048
    assert pytest.approx(a2.height, 0.02) == 20*0.3048
    assert pytest.approx(a2.width, 0.02) == 4*0.3048
    assert a2.rotation == -5*np.pi/180


def test_double_loop_toye():
    a = double_loop_toye(24.99*0.3048, 9.5*np.pi/180, 1000*0.3048)
    r = (a.x**2 + a.y**2)**0.5
    b = 0.5 * a.height - r

    assert a.x > 0
    assert a.y > 0
    assert pytest.approx(b, 0.02) == 2*0.3048
    assert pytest.approx(a.width, 0.02) == 5.9*0.3048
    assert pytest.approx(a.height, 0.02) == 9.8*0.3048
    assert pytest.approx(a.rotation, 0.02) == -0.5*9.5*np.pi/180

    a = double_loop_toye(24.99*0.3048, -9.5*np.pi/180, 1000*0.3048)

    assert a.x < 0
    assert a.y > 0
    assert pytest.approx(a.rotation, 0.02) == 0.5*9.5*np.pi/180
