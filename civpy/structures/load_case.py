"""
Copyright (c) 2019, Matt Pewsey
"""

import propy

__all__ = ['LoadCase']


class LoadCase(object):
    """
    A class representing a structural load case.

    Parameters
    ----------
    name : str
        The name of the load case.
    node_loads : list
        A list of :class:`.NodeLoad` to apply with the load case.
    elem_loads : list
        A list of :class:`.ElementLoad` to apply with the load case.
    weather : :class:`.WeatherCase`
        The applied weather case.
    wind_azimuth : float
        The wind azimuth.
    wire_condition : {'initial', 'creep'}
        The applied wire condition.
    """
    # Custom properties
    name = propy.str_property('name')

    def __init__(self, name, node_loads=[], elem_loads=[], weather=None,
                 wind_azimuth=None, wire_condition=None):
        self.name = name
        self.node_loads = node_loads
        self.elem_loads = elem_loads
        self.weather = weather
        self.wind_azimuth = wind_azimuth
        self.wire_condition = wire_condition

    def __repr__(self):
        s = ('name', 'node_loads', 'elem_loads')
        s = ('{}={!r}'.format(k, getattr(self, k)) for k in s)
        return '{}({})'.format(type(self).__name__, ', '.join(s))

    def set_nodes(self, ndict):
        """
        Sets the node references for all node loads assigned to the load case.

        Parameters
        ----------
        ndict : dict
            A dictionary mapping node names to node objects.
        """
        for n in self.node_loads:
            n.set_node(ndict)

    def set_elements(self, edict):
        """
        Sets the element references for all element loads assigned to the load
        case.

        Parameters
        ----------
        edict : dict
            A dictionary mapping element names to element objects.
        """
        for e in self.elem_loads:
            e.set_element(edict)
