from typing import Optional
from typing import Optional, Dict
from sqlalchemy.dialects.postgresql import JSONB
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def get_db_data(
    query: str,
    db_engine: Engine,
    *,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Execute a SQL query using SQLAlchemy Engine and return the result as a pandas DataFrame.

    Args:
        query: SQL query string
        db_engine: SQLAlchemy Engine instance
        params: Optional parameters for parameterized queries

    Returns:
        pd.DataFrame: Query result
    """
    try:
        with db_engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=params)
    except Exception as e:
        raise RuntimeError(
            "Database query failed. Check connection or VPN status."
        ) from e

    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    return df

import uuid
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def upsert_df(
    df: pd.DataFrame,
    table_name: str,
    engine: Engine,
    *,
    schema: str = "public",
    chunksize: int = 1000,
    dtype: Optional[Dict[str, object]] = None,
) -> bool:
    """
    UPSERT (insert or update) a pandas DataFrame into a PostgreSQL table using SQLAlchemy.

    - Uses DataFrame index as the conflict key.
    - Creates table if it does not exist.
    - Executes in a transaction for consistency.

    Args:
        df: DataFrame to upsert. Must have a named index.
        table_name: Target table name.
        engine: SQLAlchemy Engine.
        schema: Target schema (default "public").
        chunksize: Chunk size for to_sql writes.

    Returns:
        True if successful.
    """
    if df.index.names == [None] or any(n is None for n in df.index.names):
        df = df.copy()
        df.index.name = df.index.name or "idx"

    idx_cols = list(df.index.names)
    data_cols = list(df.columns)
    all_cols = idx_cols + data_cols

    idx_sql = ", ".join(f'"{c}"' for c in idx_cols)
    all_sql = ", ".join(f'"{c}"' for c in all_cols)
    update_sql = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in data_cols)

    constraint_name = f"uq_upsert_{table_name}_" + "_".join(idx_cols)
    temp_table = f"tmp_{table_name}_{uuid.uuid4().hex[:6]}"

    with engine.begin() as conn:
        # 1) Check if table exists
        exists = conn.execute(
            text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = :schema AND table_name = :name
                )
            """),
            {"schema": schema, "name": table_name},
        ).scalar_one()

        if not exists:
            # Create table if missing
            df.to_sql(
                table_name, conn, schema=schema, index=True,
                if_exists="fail", chunksize=chunksize,
                dtype=dtype,
            )
            conn.execute(
                text(f'ALTER TABLE "{schema}"."{table_name}" '
                     f'ADD CONSTRAINT {constraint_name} UNIQUE ({idx_sql})')
            )
            return True

        # 2) Create temp table
        df.to_sql(
            temp_table, conn, schema=schema, index=True,
            if_exists="replace", chunksize=chunksize,
            dtype=dtype,
        )

        # 3) Ensure unique constraint exists on target table
        conn.execute(
            text(f'ALTER TABLE "{schema}"."{table_name}" '
                 f'DROP CONSTRAINT IF EXISTS {constraint_name}')
        )
        conn.execute(
            text(f'ALTER TABLE "{schema}"."{table_name}" '
                 f'ADD CONSTRAINT {constraint_name} UNIQUE ({idx_sql})')
        )

        # 4) Perform UPSERT
        conn.execute(
            text(
                f'INSERT INTO "{schema}"."{table_name}" ({all_sql}) '
                f'SELECT {all_sql} FROM "{schema}"."{temp_table}" '
                f'ON CONFLICT ({idx_sql}) DO UPDATE SET {update_sql}'
            )
        )

        # 5) Drop temp table
        conn.execute(text(f'DROP TABLE "{schema}"."{temp_table}"'))

    return True
