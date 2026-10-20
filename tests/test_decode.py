"""Pure unit tests: no database needed. Records are packed with ``struct`` in
network byte order, exactly as PostgreSQL's *send functions emit them."""
import struct

import numpy as np
import pytest

from pg_fastread import Column, build_query, decode, record_dtype

PG_EPOCH_US = 946_684_800_000_000


def pack(rows, fmt):
    return b"".join(struct.pack(">" + fmt, *r) for r in rows)


def test_record_dtype_is_packed_big_endian():
    dt = record_dtype({"a": "int8", "b": "float8", "c": "int4", "d": "bool", "e": "int2"})
    assert dt.itemsize == 8 + 8 + 4 + 1 + 2  # no alignment padding
    assert dt["a"].byteorder == ">"


def test_decode_numeric_types_roundtrip():
    rows = [(1, 1.5, -7, 1, 3, 0.25), (2**40, -2.0, 2**31 - 1, 0, -32768, -1.0)]
    buf = pack(rows, "qdi?hf")
    cols = {"a": "int8", "b": "float8", "c": "int4", "d": "bool", "e": "int2", "f": "float4"}
    out = decode(buf, cols)
    assert out["a"].tolist() == [1, 2**40]
    assert out["b"].tolist() == [1.5, -2.0]
    assert out["c"].tolist() == [-7, 2**31 - 1]
    assert out["d"].tolist() == [True, False]
    assert out["e"].tolist() == [3, -32768]
    assert out["f"].tolist() == [0.25, -1.0]
    # results are native-endian, contiguous and writable copies
    for a in out.values():
        assert a.dtype.isnative and a.flags["C_CONTIGUOUS"] and a.flags["WRITEABLE"]
    assert out["a"].dtype == np.int64 and out["f"].dtype == np.float32


def test_decode_timestamps_from_pg_epoch():
    # 2026-01-01T00:00:00Z as microseconds since 2000-01-01
    unix_us = int(np.datetime64("2026-01-01T00:00:00", "us").astype(np.int64))
    buf = pack([(unix_us - PG_EPOCH_US,), (0,)], "q")
    out = decode(buf, {"ts": "timestamptz"})
    assert out["ts"][0] == np.datetime64("2026-01-01T00:00:00", "us")
    assert out["ts"][1] == np.datetime64("2000-01-01T00:00:00", "us")


def test_negative_infinity_timestamp_becomes_nat():
    buf = pack([(np.iinfo(np.int64).min,)], "q")
    out = decode(buf, {"ts": "timestamp"})
    assert np.isnat(out["ts"][0])


def test_empty_buffer_gives_empty_arrays():
    out = decode(b"", {"a": "int8", "b": "float8"})
    assert out["a"].shape == (0,) and out["b"].dtype == np.float64


def test_truncated_buffer_is_rejected():
    with pytest.raises(ValueError):
        decode(b"\x00" * 7, {"a": "int8"})


def test_type_aliases_and_unknown_types():
    assert Column("x", "BIGINT").pg_type == "int8"
    assert Column("x", "double precision").pg_type == "float8"
    with pytest.raises(ValueError):
        Column("x", "numeric")
    with pytest.raises(ValueError):
        Column("bad name", "int4")


def test_build_query_shape():
    sql = build_query("public.ticks", {"ts": "timestamptz", "px": "float8"},
                      where="sym = %s", order_by="ts")
    assert 'FROM "public"."ticks"' in sql
    assert "timestamptz_send((\"ts\")::timestamptz) || float8send((\"px\")::float8)" in sql
    assert "ORDER BY ts" in sql and sql.endswith("WHERE sym = %s")
    assert "count(*) AS n" in sql


def test_build_query_null_sentinels():
    sql = build_query("t", {"px": "float8", "q": "int4"}, allow_nulls=True)
    assert "COALESCE((\"px\")::float8, 'NaN'::float8)" in sql
    assert "'-2147483648'::int4" in sql


def test_custom_expression_column():
    sql = build_query("t", [Column("mid", "float8", expr="(bid + ask) / 2")])
    assert "float8send(((bid + ask) / 2)::float8)" in sql


def test_identifier_quoting_escapes_quotes():
    sql = build_query('we"ird', {"a": "int4"})
    assert 'FROM "we""ird"' in sql
