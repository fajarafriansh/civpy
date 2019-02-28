"""
Copyright (c) 2019, Matt Pewsey
"""

import numpy as np
from math import cos, sin, log
from ..math import Ellipse

__all__ = [
    'loop_factor',
    'single_loop_davison',
    'single_loop_cigre',
    'single_loop_aeso',
    'double_loop_toye',
]


def loop_factor(span_fract, num_loops):
    """
    Returns the galloping ellipse scale factor based on the span length
    and number of loops.

    Parameters
    ----------
    span_fract : float
        The fraction of the span length.
    num_loops : int
        The number of galloping loops.
    """
    if span_fract > 1 or span_fract < 0:
        raise ValueError('Span fraction {!r} beyond span bounds.'.format(span_fract))

    n = 1 / num_loops
    b = n * (span_fract // n)
    a = b + n
    return 4 * num_loops**2 * (span_fract - b) * (a - span_fract)


def single_loop_davison(sag, swing, span_fract=0.5, **kwargs):
    """
    Returns an ellipse for single loop Davison galloping.

    Parameters
    ----------
    sag : float
        The mid span sag in the wire.
    swing : float
        The blowout angle of the wire in radians.
    """
    rot = -0.5 * swing
    major = 1.25 * sag + 0.3048
    minor = 0.4 * major
    b = 0.25 * sag

    # Scale based on location in span
    f = loop_factor(span_fract, 1)
    minor *= f
    major *= f
    b *= f

    ang = np.pi / 2 + rot
    c = 0.5 * major - b
    x = c * cos(ang)
    y = c * sin(ang)

    return Ellipse(x=x, y=y, width=minor, height=major, rotation=rot)


def single_loop_cigre(sag, diameter, num_wires, span_fract=0.5, **kwargs):
    """
    Returns an ellipse for single loop Cigre 322 galloping.

    Parameters
    ----------
    sag : float
        The mid span sag in the wire.
    diameter : float
        The diameter of the wire.
    num_wires : int
        The number of wires in the bundle.
    """
    f = 8*sag / (50*diameter)

    if num_wires > 1:
        major = 170 * diameter * log(0.1 * f)
    else:
        major = 80 * diameter * log(f)

    if major <= 0:
        major = 1e-15

    minor = 0.4 * major
    b = 0.3 * major

    # Scale based on location in span
    f = loop_factor(span_fract, 1)
    minor *= f
    major *= f
    b *= f

    y = 0.5 * major - b

    return Ellipse(x=0, y=y, width=minor, height=major)


def single_loop_aeso(sag, diameter, num_wires, span_fract=0.5, **kwargs):
    """
    Returns ellipses for single loop AESO galloping.

    Parameters
    ----------
    sag : float
        The mid span sag in the wire.
    diameter : float
        The diameter of the wire.
    num_wires : int
        The number of wires in the bundle.
    """
    rot = 5 * np.pi / 180
    f = 8*sag / (50*diameter)

    if num_wires > 1:
        major = 170 * diameter * log(0.1 * f)
    else:
        major = 80 * diameter * log(f)

    if major <= 0:
        major = 1e-15

    minor = 0.2 * major
    b = 0.25 * major

    # Scale based on location in span
    f = loop_factor(span_fract, 1)
    minor *= f
    major *= f
    b *= f

    y = 0.5 * major - b

    e1 = Ellipse(x=0, y=y, width=minor, height=major, rotation=rot)
    e2 = Ellipse(x=0, y=y, width=minor, height=major, rotation=-rot)

    return e1, e2


def double_loop_toye(sag, swing, span, span_fract=0.25, **kwargs):
    """
    Returns an ellipse for double loop Toye galloping.

    Parameters
    ----------
    sag : float
        The mid span sag in the wire.
    swing : float
        The blowout angle of the wire in radians.
    span : float
        The span length.
    """
    rot = -0.5 * swing
    a = ((0.5 * span)**2 + sag**2)**0.5
    f = (3*a/8) * (span + 8*sag**2/(3*span) - 2*a)

    if f <= 0:
        f = 1e-15

    major = f**0.5 + 0.3048
    minor = 1.104 * (major - 0.3048)**0.5
    b = 0.2 * major

    # Scale based on location in span
    f = loop_factor(span_fract, 2)
    minor *= f
    major *= f
    b *= f

    ang = np.pi / 2 + rot
    c = 0.5 * major - b
    x = c * cos(ang)
    y = c * sin(ang)

    return Ellipse(x=x, y=y, width=minor, height=major, rotation=rot)
