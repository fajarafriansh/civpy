import pytest
import numpy as np
from .ellipse import *


def test_repr():
    e = Ellipse(1, 2, 3, 4)
    repr(e)


def test_point():
    e = Ellipse(1, 2, 3, 4)

    a = e.point(0)
    b = np.array([2.5, 2])
    assert pytest.approx(a) == b

    a = e.point(np.pi/2)
    b = np.array([1, 4])
    assert pytest.approx(a) == b


def test_contains():
    e = Ellipse(0, 0, 4, 8, np.pi/4)
    p = e.points()

    a = np.array([e.contains(x) for x in 1.01 * p], dtype='bool')
    b = np.zeros(len(a), dtype='bool')
    assert (a == b).all()

    a = np.array([e.contains(x) for x in 0.99 * p], dtype='bool')
    b = np.ones(len(a), dtype='bool')
    assert (a == b).all()


def test_perimeter():
    e = Ellipse(0, 0, 5, 10)

    a = e.perimeter()
    b = e.polygon().length
    assert pytest.approx(a, 0.01) == b


def test_polygon():
    e = Ellipse(0, 0, 5, 10)
    e.polygon()


def test_area():
    e = Ellipse(0, 0, 5, 10)

    a = e.area()
    assert pytest.approx(a, 0.01) == 39.27


def test_lap_area():
    e1 = Ellipse(0, 0, 5, 10)
    e2 = Ellipse(0, 0, 5, 10)

    a = e1.lap_area(e2)
    assert pytest.approx(a, 0.01) == 39.27


def test_lap_fractions():
    e1 = Ellipse(0, 0, 5, 10)
    e2 = Ellipse(0, 0, 5, 10)

    a, b = e1.lap_fractions(e2)
    assert pytest.approx(a, 0.01) == 1
    assert pytest.approx(b, 0.01) == 1


def test_distance():
    e1 = Ellipse(0, 0, 10, 20)
    e2 = Ellipse(20, 0, 20, 10, np.pi/2)

    p1 = e1.polygon()
    p2 = e2.polygon()

    a, p = e1.distance(e2)
    b = p1.distance(p2)

    p[np.abs(p) < 1e-5] = 0
    p = p.ravel()
    q = np.array([[5, 0], [15, 0]]).ravel()

    assert pytest.approx(a, 0.01) == b
    assert pytest.approx(p, 0.01) == q
