"""End-to-end tests against a real PostgreSQL.

Set PG_FASTREAD_DSN (e.g. ``postgresql://postgres@localhost:5432/postgres``);
the module is skipped when it is unset or psycopg is not installed.
"""
import os
from datetime import timezone

import numpy as np
import pytest

psycopg = pytest.importorskip("psycopg")
DSN = os.environ.get("PG_FASTREAD_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="PG_FASTREAD_DSN not set")

from pg_fastread import fetch_arrays, fetch_arrays_chunked  # noqa: E402

COLS = {"id": "int8", "ts": "timestamptz", "price": "float8", "qty": "int4", "side": "bool"}
N = 50_000


@pytest.fixture(scope="module")
def conn():
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute("DROP TABLE IF EXISTS pgfr_ticks")
        c.execute(
            "CREATE TABLE pgfr_ticks (id bigint PRIMARY KEY, ts timestamptz NOT NULL, "
            "price float8, qty int4 NOT NULL, side bool NOT NULL)"
        )
        c.execute(
            "INSERT INTO pgfr_ticks SELECT g, timestamptz '2026-01-01 09:00+00' + g * interval '137 ms', "
            "100 + sin(g / 1000.0) * 5, (g * 37) %% 1000 + 1, g %% 2 = 0 FROM generate_series(1, %s) g",
            (N,),
        )
        c.execute("UPDATE pgfr_ticks SET price = NULL WHERE id = 777")
        yield c
        c.execute("DROP TABLE pgfr_ticks")


def reference(conn, nulls_as_nan=True):
    rows = conn.execute("SELECT id, ts, price, qty, side FROM pgfr_ticks ORDER BY id").fetchall()
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    ts = np.array([np.datetime64(r[1].astimezone(timezone.utc).replace(tzinfo=None), "us") for r in rows])
    price = np.array([np.nan if r[2] is None else r[2] for r in rows])
    qty = np.array([r[3] for r in rows], dtype=np.int32)
    side = np.array([r[4] for r in rows], dtype=bool)
    return dict(id=ids, ts=ts, price=price, qty=qty, side=side)


def test_bit_identical_to_row_fetch(conn):
    fast = fetch_arrays(conn, "pgfr_ticks", COLS, order_by="id", allow_nulls=True)
    ref = reference(conn)
    for k in COLS:
        assert np.array_equal(fast[k], ref[k], equal_nan=(k == "price")), k
    assert np.isnan(fast["price"][776])


def test_nulls_are_detected_not_silently_dropped(conn):
    with pytest.raises(ValueError, match="NULL"):
        fetch_arrays(conn, "pgfr_ticks", COLS, order_by="id")


def test_where_with_parameters(conn):
    out = fetch_arrays(conn, "pgfr_ticks", {"id": "int8", "qty": "int4"},
                       where="id <= %s AND side", params=(1000,), order_by="id")
    assert out["id"].tolist() == list(range(2, 1001, 2))


def test_chunked_equals_single_read(conn):
    one = fetch_arrays(conn, "pgfr_ticks", {"id": "int8", "qty": "int4"}, order_by="id")
    edges = [1, 12_345, 30_000, N + 1]
    many = fetch_arrays_chunked(conn, "pgfr_ticks", {"id": "int8", "qty": "int4"}, key="id",
                                bounds=zip(edges[:-1], edges[1:]), order_by="id")
    assert np.array_equal(one["id"], many["id"]) and np.array_equal(one["qty"], many["qty"])


def test_empty_result(conn):
    out = fetch_arrays(conn, "pgfr_ticks", COLS, where="id < 0")
    assert all(len(v) == 0 for v in out.values())


def test_infinite_timestamps_come_back_as_nat(conn):
    conn.execute("CREATE TEMP TABLE pgfr_inf (ts timestamptz NOT NULL)")
    try:
        conn.execute("INSERT INTO pgfr_inf VALUES ('-infinity'), ('2026-01-01 00:00+00'), ('infinity')")
        out = fetch_arrays(conn, "pgfr_inf", {"ts": "timestamptz"}, order_by="ts")
        assert np.isnat(out["ts"][0]) and np.isnat(out["ts"][2])
        assert out["ts"][1] == np.datetime64("2026-01-01T00:00:00", "us")
    finally:
        conn.execute("DROP TABLE pgfr_inf")
