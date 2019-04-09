import numpy as np


__all__ = [
    'polyfit',
    'polyval',
    'polyderiv',
]


def polyfit(x, y, deg=1):
    return np.polyfit(x, y, deg)[::-1]


def polyval(p, x):
    return np.polyval(p[::-1], x)


def polyderiv(p, x, n=1):
    p = np.asarray(p)
    x = np.asarray(x)

    if p.shape[0] <= n:
        # Derivative vanishes
        if len(x.shape) == 0:
            return 0

        return np.zeros(x.shape[0])

    i = np.arange(n, p.shape[0])
    c = i.astype('float')

    for j in range(1, n):
        c *= i - j

    c *= p[n:]
    i -= n

    if len(x.shape) == 0:
        return np.sum(c * x**i)

    return np.array([np.sum(c * v**i) for v in x])
