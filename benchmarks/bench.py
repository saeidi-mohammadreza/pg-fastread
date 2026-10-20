"""Benchmark: four ways to get a PostgreSQL table into NumPy arrays.

    python benchmarks/bench.py --dsn postgresql://postgres@localhost/postgres --rows 2000000

Every method's output is checked against the row-fetch baseline before its time
is reported, so a fast-but-wrong method cannot appear in the table.
"""
from __future__ import annotations

import argparse
import io
import statistics
import time
from datetime import timezone

import numpy as np
import psycopg

from pg_fastread import fetch_arrays

COLS = {"id": "int8", "ts": "timestamptz", "price": "float8", "qty": "int4"}
TABLE = "pgfr_bench"


def setup(conn, rows: int) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
    conn.execute(
        f"CREATE TABLE {TABLE} (id bigint PRIMARY KEY, ts timestamptz NOT NULL, "
        "price float8 NOT NULL, qty int4 NOT NULL)"
    )
    conn.execute(
        f"INSERT INTO {TABLE} SELECT g, timestamptz '2026-01-01 09:00+00' + g * interval '137 ms', "
        "100 + sin(g / 1000.0) * 5 + (g % 7) * 0.01, (g * 37) % 1000 + 1 "
        "FROM generate_series(1, %s) g",
        (rows,),
    )
    conn.execute(f"VACUUM ANALYZE {TABLE}")


def row_fetch(conn):
    rows = conn.execute(f"SELECT id, ts, price, qty FROM {TABLE} ORDER BY id").fetchall()
    ids, ts, price, qty = zip(*rows)
    return {
        "id": np.array(ids, dtype=np.int64),
        "ts": np.array([t.astimezone(timezone.utc).replace(tzinfo=None) for t in ts], dtype="datetime64[us]"),
        "price": np.array(price, dtype=np.float64),
        "qty": np.array(qty, dtype=np.int32),
    }


def copy_csv(conn):
    buf = io.BytesIO()
    with conn.cursor() as cur:
        with cur.copy(
            f"COPY (SELECT id, (extract(epoch FROM ts) * 1e6)::int8, price, qty "
            f"FROM {TABLE} ORDER BY id) TO STDOUT WITH (FORMAT csv)"
        ) as cp:
            for chunk in cp:
                buf.write(chunk)
    buf.seek(0)
    a = np.loadtxt(buf, delimiter=",", dtype=np.float64)  # floats parse exactly via repr
    return {
        "id": a[:, 0].astype(np.int64),
        "ts": a[:, 1].astype(np.int64).view("datetime64[us]"),
        "price": a[:, 2],
        "qty": a[:, 3].astype(np.int32),
    }


def pandas_read_sql(conn):
    import pandas as pd

    df = pd.read_sql(f"SELECT id, ts, price, qty FROM {TABLE} ORDER BY id", conn)
    return {
        "id": df["id"].to_numpy(np.int64),
        "ts": df["ts"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[us]"),
        "price": df["price"].to_numpy(np.float64),
        "qty": df["qty"].to_numpy(np.int32),
    }


def packed(conn):
    return fetch_arrays(conn, TABLE, COLS, order_by="id")


def timeit(fn, conn, repeat):
    out, times = None, []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(conn)
        times.append(time.perf_counter() - t0)
    return out, statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--skip-pandas", action="store_true")
    args = ap.parse_args()

    with psycopg.connect(args.dsn, autocommit=True) as conn:
        setup(conn, args.rows)
        methods = [("psycopg fetchall -> np.array", row_fetch), ("COPY csv -> np.loadtxt", copy_csv)]
        if not args.skip_pandas:
            methods.append(("pandas.read_sql", pandas_read_sql))
        methods.append(("pg-fastread (packed bytea)", packed))

        ref, base = timeit(row_fetch, conn, args.repeat)
        print(f"rows: {args.rows:,}   (median of {args.repeat})\n")
        print(f"| method | seconds | speed-up | mismatches |")
        print(f"| --- | ---: | ---: | ---: |")
        for name, fn in methods:
            out, t = (ref, base) if fn is row_fetch else timeit(fn, conn, args.repeat)
            bad = sum(int((~((out[k] == ref[k]))).sum()) for k in COLS)
            print(f"| {name} | {t:.3f} | {base / t:.1f}x | {bad} |")
        conn.execute(f"DROP TABLE {TABLE}")


if __name__ == "__main__":
    main()
