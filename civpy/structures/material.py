"""
Copyright (c) 2019, Matt Pewsey
"""

import propy

__all__ = ['Material']


class Material(object):
    """
    A class representing an engineered material.

    Parameters
    ----------
    name : str
        The name of the material.
    elasticity : float
        The modulus of elasticity.
    rigidity : float
        The modulus of rigidity.
    """
    # Custom properties
    name = propy.str_property('name')

    def __init__(self, name, elasticity, rigidity=0):
        self.name = name
        self.elasticity = elasticity
        self.rigidity = rigidity

    def __repr__(self):
        s = ('name', 'elasticity', 'rigidity')
        s = ('{}={!r}'.format(k, getattr(self, k)) for k in s)
        return '{}({})'.format(type(self).__name__, ', '.join(s))
