import numpy as np
from .config_db import DB_CONNECTION

__all__ = ['query_wire']


def query_wire(name):
    name = name.upper()
    statement = "SELECT * FROM wires WHERE UPPER(name)='{}';".format(name)
    cursor = DB_CONNECTION.execute(statement)

    header = [x[0] for x in cursor.description]
    row = cursor.fetchone()

    odict = {k: x for k, x in zip(header, row) if x is not None}

    keys = (
        'outer_initial_coeffs',
        'outer_creep_coeffs',
        'core_initial_coeffs',
        'core_creep_coeffs',
        'temperature_data',
        'resistance_data',
    )

    for k in keys:
        if k in odict:
            v = str(odict[k]).split(' ')
            odict[k] = np.array(v, dtype='float')

    return odict
