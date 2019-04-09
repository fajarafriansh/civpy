import re
import pandas as pd
from config_db import DB_CONNECTION

__all__ = []


OPTIONS = dict(
    if_exists='replace',
    index=False,
)


def gsheet_csv_url(url):
    """
    Returns the url for the Google Sheet csv export.

    Parameters
    ----------
    url : str
        The editor url string as found when viewing the sheet in a browser.
    """
    def get_sheet():
        for i, x in enumerate(s):
            if x == 'gid':
                return s[i+1]

        raise ValueError('Sheet ID not found in url {}'.format(url))

    s = re.split('/|#|=|&', url)

    key = s[5]
    sheet = get_sheet()

    return 'https://docs.google.com/spreadsheets/d/{}/export?gid={}&format=csv'.format(key, sheet)


def make_unique_columns(df):
    """
    Renames the data frame columns to be unique when exported to an SQL
    database.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The data frame.
    """
    names = {}
    lnames = set()

    for x in list(df):
        n = x

        # Get a unique name
        while n.lower() in lnames:
            n = n + '_'

        lnames.add(n.lower())
        names[x] = n

    df.rename(columns=names, inplace=True)

    return df


def write_wires():
    """
    Writes the 'wires' table to the database.
    """
    url = 'https://docs.google.com/spreadsheets/d/15lkBkeVt8kA3VrHsu0aEMPnZ5K6lawNtqW4Xrra6oZs/edit#gid=1260312582'
    table = 'wires'

    url = gsheet_csv_url(url)

    df = pd.read_csv(url)
    make_unique_columns(df)

    df.to_sql(table, DB_CONNECTION, **OPTIONS)


def write_database():
    """
    Writes all data to the database.
    """
    write_wires()


if __name__ == '__main__':
    write_database()
