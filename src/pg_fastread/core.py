"""Read PostgreSQL rows into NumPy arrays through one packed ``bytea`` value.

The idea: instead of shipping N rows that the driver turns into N Python tuples,
ask the server to pack every row into fixed-width binary records
(``int8send(a) || float8send(b) || ...``), concatenate them with ``string_agg``
and return a *single* ``bytea``. The client then reinterprets that buffer with
``np.frombuffer`` -- no per-row Python objects are ever created.

The ``*send`` functions emit network byte order (big-endian), so the decode is a
structured big-endian dtype followed by one vectorised byte swap per column.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "PG_TYPES",
    "Column",
    "build_query",
    "record_dtype",
    "decode",
    "fetch_arrays",
    "fetch_arrays_chunked",
]

# PostgreSQL epoch (2000-01-01) expressed in microseconds since the Unix epoch.
_PG_EPOCH_US = 946_684_800_000_000


@dataclass(frozen=True)
class _PgType:
    send_fn: str          # server-side function producing the binary form
    wire: str             # numpy dtype of the wire (big-endian) representation
    native: str           # numpy dtype returned to the caller
    null_literal: str     # SQL literal substituted for NULL when nulls are allowed


PG_TYPES: dict[str, _PgType] = {
    "int2": _PgType("int2send", ">i2", "int16", "'-32768'::int2"),
    "int4": _PgType("int4send", ">i4", "int32", "'-2147483648'::int4"),
    "int8": _PgType("int8send", ">i8", "int64", "'-9223372036854775808'::int8"),
    "float4": _PgType("float4send", ">f4", "float32", "'NaN'::float4"),
    "float8": _PgType("float8send", ">f8", "float64", "'NaN'::float8"),
    "bool": _PgType("boolsend", "u1", "bool", "false"),
    # timestamp / timestamptz are int8 microseconds since 2000-01-01 on the wire.
    "timestamp": _PgType("timestamp_send", ">i8", "datetime64[us]", "'-infinity'::timestamp"),
    "timestamptz": _PgType("timestamptz_send", ">i8", "datetime64[us]", "'-infinity'::timestamptz"),
}

_ALIASES = {
    "smallint": "int2",
    "integer": "int4",
    "int": "int4",
    "bigint": "int8",
    "real": "float4",
    "double precision": "float8",
    "double": "float8",
    "boolean": "bool",
}

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_ident(name: str) -> str:
    """Quote an identifier; dotted names (schema.table) are quoted per part."""
    parts = name.split(".")
    for p in parts:
        if not p:
            raise ValueError(f"empty identifier part in {name!r}")
    return ".".join('"' + p.replace('"', '""') + '"' for p in parts)


@dataclass(frozen=True)
class Column:
    """One output column: a SQL expression, its PostgreSQL type and an output name."""

    name: str
    pg_type: str
    expr: str | None = None  # defaults to the quoted column name

    def __post_init__(self) -> None:
        t = _ALIASES.get(self.pg_type.lower(), self.pg_type.lower())
        if t not in PG_TYPES:
            raise ValueError(
                f"unsupported type {self.pg_type!r} for column {self.name!r}; "
                f"supported: {sorted(PG_TYPES)}"
            )
        object.__setattr__(self, "pg_type", t)
        if not _IDENT.match(self.name):
            raise ValueError(f"output name {self.name!r} must be a plain identifier")

    @property
    def sql_expr(self) -> str:
        return self.expr if self.expr is not None else _quote_ident(self.name)


def _as_columns(columns: Mapping[str, str] | Sequence[Column]) -> list[Column]:
    if isinstance(columns, Mapping):
        cols = [Column(n, t) for n, t in columns.items()]
    else:
        cols = list(columns)
    if not cols:
        raise ValueError("at least one column is required")
    return cols


def record_dtype(columns: Mapping[str, str] | Sequence[Column]) -> np.dtype:
    """Packed (unaligned) big-endian record dtype matching ``build_query`` output."""
    cols = _as_columns(columns)
    return np.dtype([(c.name, PG_TYPES[c.pg_type].wire) for c in cols])


def build_query(
    table: str,
    columns: Mapping[str, str] | Sequence[Column],
    *,
    where: str | None = None,
    order_by: str | None = None,
    allow_nulls: bool = False,
) -> str:
    """Return SQL that yields one row, one ``bytea`` column with every record packed.

    ``where`` and ``order_by`` are inserted verbatim: pass trusted SQL and use
    driver parameters (``%s``) for values.

    With ``allow_nulls=False`` (default) a NULL anywhere makes ``string_agg`` drop
    the whole row silently, so the query also returns the row count and
    :func:`fetch_arrays` checks it. With ``allow_nulls=True`` NULLs are replaced by
    a per-type sentinel (NaN for floats, the minimum value for integers,
    ``-infinity`` for timestamps, false for bools).
    """
    cols = _as_columns(columns)
    pieces = []
    for c in cols:
        t = PG_TYPES[c.pg_type]
        expr = c.sql_expr
        if allow_nulls:
            expr = f"COALESCE(({expr})::{c.pg_type}, {t.null_literal})"
        else:
            expr = f"({expr})::{c.pg_type}"
        pieces.append(f"{t.send_fn}({expr})")
    packed = " || ".join(pieces)
    agg_order = f" ORDER BY {order_by}" if order_by else ""
    sql = (
        f"SELECT COALESCE(string_agg({packed}, ''::bytea{agg_order}), ''::bytea) AS buf, "
        f"count(*) AS n FROM {_quote_ident(table)}"
    )
    if where:
        sql += f" WHERE {where}"
    return sql


def decode(buf: bytes | bytearray | memoryview, columns: Mapping[str, str] | Sequence[Column]) -> dict[str, np.ndarray]:
    """Turn the packed buffer into a dict of native-endian, contiguous NumPy arrays."""
    cols = _as_columns(columns)
    dt = record_dtype(cols)
    mv = memoryview(buf)
    if mv.nbytes % dt.itemsize:
        raise ValueError(
            f"buffer of {mv.nbytes} bytes is not a multiple of the record size {dt.itemsize}"
        )
    rec = np.frombuffer(mv, dtype=dt)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        t = PG_TYPES[c.pg_type]
        raw = rec[c.name]
        if c.pg_type in ("timestamp", "timestamptz"):
            us = raw.astype(np.int64)
            # -infinity and infinity arrive as int64 min and max. Both become NaT;
            # shifting them to the Unix epoch would overflow.
            big = np.iinfo(np.int64)
            inf = (us == big.min) | (us == big.max)
            us = np.where(inf, big.min, us + _PG_EPOCH_US)
            out[c.name] = us.view("datetime64[us]")
        elif c.pg_type == "bool":
            out[c.name] = raw.astype(bool)
        else:
            out[c.name] = raw.astype(t.native)  # byte swap + contiguous copy
    return out


def fetch_arrays(
    conn,
    table: str,
    columns: Mapping[str, str] | Sequence[Column],
    *,
    where: str | None = None,
    params: Sequence | Mapping | None = None,
    order_by: str | None = None,
    allow_nulls: bool = False,
) -> dict[str, np.ndarray]:
    """Run the packed query on a psycopg 3 connection and decode the result."""
    cols = _as_columns(columns)
    sql = build_query(table, cols, where=where, order_by=order_by, allow_nulls=allow_nulls)
    with conn.cursor(binary=True) as cur:
        cur.execute(sql, params)
        buf, n = cur.fetchone()
    arrays = decode(buf, cols)
    got = len(next(iter(arrays.values())))
    if got != n:
        raise ValueError(
            f"{n - got} of {n} rows contained NULLs and were dropped; "
            "pass allow_nulls=True or filter them out"
        )
    return arrays


def fetch_arrays_chunked(
    conn,
    table: str,
    columns: Mapping[str, str] | Sequence[Column],
    *,
    key: str,
    bounds: Iterable[tuple[object, object]],
    order_by: str | None = None,
    allow_nulls: bool = False,
) -> dict[str, np.ndarray]:
    """Read in key ranges ``[lo, hi)`` and concatenate.

    A single ``bytea`` is limited to 1 GB, so very large reads are split by a key
    (for example a timestamp or an id). Ranges are read in the order given.
    """
    cols = _as_columns(columns)
    k = _quote_ident(key)
    parts = [
        fetch_arrays(
            conn, table, cols,
            where=f"{k} >= %s AND {k} < %s", params=(lo, hi),
            order_by=order_by, allow_nulls=allow_nulls,
        )
        for lo, hi in bounds
    ]
    if not parts:
        return {c.name: np.empty(0, dtype=PG_TYPES[c.pg_type].native) for c in cols}
    return {c.name: np.concatenate([p[c.name] for p in parts]) for c in cols}
