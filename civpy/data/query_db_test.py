from .query_db import *


def test_query_wire():
    odict = query_wire('Drake ACSR')
    assert odict['name'] == 'Drake ACSR'
