import json
import os
from typing import Union, Iterable
from datetime import datetime
import numpy as np
import pandas as pd

import sqlalchemy as sa
from sqlalchemy.engine import Connection


def read_sql_file(name: str) -> str:
    # navigate to parent dir
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(parent_dir):
        raise ValueError("Parent directory not found")

    # navigate to data dir
    dir_data = os.path.join(parent_dir, "data")
    if not os.path.isdir(dir_data):
        raise ValueError("Data directory not found")

    # navigate to secrets dir
    dir_sql = os.path.join(dir_data, "sql")
    if not os.path.isdir(dir_sql):
        raise ValueError("Sql directory not found")

    # navigate to file
    if not name.endswith(".sql"):
        name += ".sql"
    fn_sql = os.path.join(dir_sql, name)

    if not os.path.isfile(fn_sql):
        raise ValueError("Query file not found")

    # load file
    with open(fn_sql, "r") as f:
        return f.read()


def get_db_connection(fn_connection: str) -> Connection:
    """
    Get a connection to the database.
    This function assumes RedShift databases and uses postgresql_psycopg2 driver.

    :param fn_connection:
        Path to db_connect.json notepad file;
            A JSON file which consist the credentials.
            The required keys are: "host", "port", "dbname", "user", "password"

    :return:
        The connection object
    """

    # # verify the path exists
    # assert os.path.exists(fn_connection)

    # load credentials json file
    sql_config = json.load(open(fn_connection))

    # extract credentials
    host, port, dbname, user, password = [sql_config[k] for k in ["host", "port", "dbname", "user", "password"]]

    # connect Redshift
    conn = sa.create_engine(sa.engine.url.URL.create(drivername="postgresql+psycopg2",
                                                     username=user,
                                                     password=password,
                                                     host=host,
                                                     port=port,
                                                     database=dbname,
                                                     )
                            ).connect()

    return conn


def generate_where_part(vals: Iterable, col_name: str) -> str:
    """ For when you have a list of values and want to generate a WHERE part of an SQL query"""

    # values which will be used to filter column, e.g., "WHERE col IN (values)"
    vals = [str(v) for v in vals]
    if not vals:
        # empty value list, should not return anything
        ret = "FALSE"
    else:
        ret = ",".join(vals)
        ret = f"{col_name} IN ({ret})"

    return ret


def read_table(table_name: str,
               cur: Connection,
               where: Union[None, str] = None,
               limit: int = None
               ) -> pd.DataFrame:
    """ Read a table by its name, optionally with a WHERE clause and a LIMIT """

    # generate query
    query = f"SELECT * FROM {table_name}"
    if where:
        query += f" WHERE {where}"
    if limit:
        query += f" LIMIT {limit}"

    # execute query
    ret = pd.read_sql(query, cur)

    return ret


def get_tables_and_views(cur: Connection,
                         schema: str = None
                         ) -> pd.DataFrame:
    """ Get a dataframe of tables and views in a schema """

    # generate query
    sql = f"""
        SELECT t.table_name, t.table_type
        FROM information_schema.tables t
    """
    if schema:
        sql += f" WHERE table_schema = '{schema}';"

    # execute query
    df_tables_and_views = pd.read_sql(sql, cur)

    return df_tables_and_views


def drop_table_if_exists(cur: Connection,
                         schema: str,
                         table_name: str
                         ):
    # generate query
    sql = f"DROP TABLE IF EXISTS {schema}.{table_name}"

    print(f"Dropping {schema}.{table_name} - started at: \033[1m{datetime.now()}\033[0m")

    # execute query
    cur.execute(sql)

    print(f"Dropping {schema}.{table_name} - finished at: \033[1m{datetime.now()}\033[0m")


def sanity_check_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform several diagnostic heuristics on a dataframe.

    Return the results as a dataframe
    """

    # extract numeric_cols from df
    numeric_cols = set(df.select_dtypes(include=np.number).columns)

    # extract all cols from df
    cols = df.columns

    # initialize dictionary to store diagnostic heuristics results of df
    res = dict()

    for col in cols:
        # retrieve col's data
        vals = df[col]
        # initialize dictionary to store diagnostic heuristics results of col
        curr = dict()
        # store number of values in col
        curr["n"] = f"{len(vals):8,d}"
        # store number of unique values in col
        curr["n_unique"] = f"{vals.nunique(dropna=False):8,d}"
        # store cardinality of col
        curr["cardinality_fraction"] = np.round(vals.nunique() / len(vals), 3)
        # store number of empty values (nulls/0's) in col
        if col in numeric_cols:
            empty = vals.isna().sum() + (vals == 0).sum()
        else:
            empty = (vals.fillna("").astype(str) == "").sum()
        assert empty <= len(vals)
        # store fraction of empty values (nulls/0's) in col
        empty_fraction = empty / len(vals)
        empty_fraction = f"{empty_fraction:5.3f}{' ***' if empty_fraction > 0.95 else ''}"
        curr["empty_fraction"] = empty_fraction
        # store current col results
        res[col] = curr

    # store diagnostic heuristics results in df
    ret = pd.DataFrame(res)

    return ret


def sample_an_sql_table(table_name: str,
                        n_limit: int,
                        cur: Connection
                        ) -> pd.DataFrame:
    """ Extract a random sample of a table with an APPROXIMATE size of `n_limit` """

    # extract number of rows in table
    n_rows = pd.read_sql(f"SELECT COUNT(*) AS n FROM {table_name} ", cur).n.iloc[0]
    # limit calculation
    frac = n_limit / n_rows
    # extract df limit rows
    df_sample = pd.read_sql(f"SELECT * FROM {table_name} WHERE RAND() < {frac} LIMIT {n_limit}", cur)

    return df_sample


def get_n_rows(cur: Connection,
               schema: str,
               table_name: str
               ):
    """ Get the number of rows in a table as a well-formatted string """

    # extract number of rows in table
    res = cur.execute(f"SELECT COUNT(*) from {schema}.{table_name}")
    res = next(res)[0]
    # table decryption string
    what = f"{schema}.{table_name:30s}"
    ret = f"{what}: {res:12,d} rows"

    return ret


def actual_create_table(engine: Connection,
                        conn: Connection,
                        cur: Connection,
                        schema: str,
                        table_name: str,
                        create_table_from_sql_or_df: str,
                        sql: str,
                        df: pd.DataFrame,
                        skip_if_exists: bool = False,
                        drop_if_exists: bool = True,
                        add_create_table_clause: bool = True,
                        ):
    # skip table if exists
    if skip_if_exists and table_exists(conn, table_name, schema):
        print(f"Table {schema}.{table_name} already exists, skipping.")
        return

    # drop table if exists
    if drop_if_exists and table_exists(conn, table_name, schema):
        drop_table_if_exists(cur=cur, schema=schema, table_name=table_name)

    print(f"Creating {schema}.{table_name} - started at: \033[1m{datetime.now()}\033[0m")

    # create sql query
    lines = []
    for l in sql.splitlines():
        l = l.strip()
        if l.startswith("--"):
            continue
        lines.append(l)
    sql = " ".join(lines)

    # create table from sql query
    if create_table_from_sql_or_df == "sql":
        # add clause
        if add_create_table_clause:
            sql = f"""
                    CREATE TABLE {schema}.{table_name} AS (
                        {sql}
                    )
                    """
        # create table
        cur.execute(sql)

    # create table from df
    elif create_table_from_sql_or_df == "df":
        df.to_sql(name=table_name,
                  con=engine,
                  schema=schema,
                  index=False,
                  if_exists='replace',
                  chunksize=1000
                  )

    else:
        raise ValueError("Parameter create_table_from_sql_or_df was not defined correctly")

    print(f"Creating {schema}.{table_name} - finished at: \033[1m{datetime.now()}\033[0m")


def table_exists(cur: Connection,
                 table_name: str,
                 schema_name: str = None
                 ) -> bool:
    # get tables exists in schema
    df_tables = get_tables_and_views(cur, schema_name)

    return table_name in df_tables.table_name.values


def get_table_summary(cur: Connection,
                      schema: str,
                      table_name: str
                      ) -> str:
    """Get a nicely formatted summary of a table as a string"""

    #  get the number of rows in a table as a well-formatted string
    ret = [get_n_rows(cur, schema, table_name)]

    # extract a random sample of a table with an APPROXIMATE size of `n_limit`
    df = sample_an_sql_table(table_name=f"{schema}.{table_name}",
                             n_limit=10,
                             cur=cur
                             )

    # perform several diagnostic heuristics on a dataframe
    df_sanity = sanity_check_df(df)

    # store summary of a table in df
    df = pd.concat(
        [
            df.tail(),
            pd.DataFrame([["..."] * df.shape[1]], columns=df.columns, index=["..."]),
            df_sanity,
        ]
    )
    ret.append(df.to_markdown())
    ret = "\n".join(ret)

    return ret
