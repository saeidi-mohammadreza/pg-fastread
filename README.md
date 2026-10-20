# pg-fastread

[![CI](https://github.com/saeidi-mohammadreza/pg-fastread/actions/workflows/ci.yml/badge.svg)](https://github.com/saeidi-mohammadreza/pg-fastread/actions/workflows/ci.yml)

**Read millions of PostgreSQL rows into NumPy arrays without creating a single Python object per row.**

The usual path, `cursor.fetchall()` → list of tuples → `np.array`, spends most of its time building and then throwing away millions of Python objects. `pg-fastread` asks the server to pack every row into fixed-width binary records and return them as **one `bytea` value**; the client reinterprets that buffer with `np.frombuffer` and a single vectorised byte swap per column.

```sql
SELECT string_agg(int8send(id) || timestamptz_send(ts) || float8send(price) || int4send(qty),
                  ''::bytea ORDER BY id),
       count(*)
FROM ticks WHERE ...
```

```python
import psycopg
from pg_fastread import fetch_arrays

with psycopg.connect("postgresql://localhost/market") as conn:
    a = fetch_arrays(conn, "ticks",
                     {"id": "int8", "ts": "timestamptz", "price": "float8", "qty": "int4"},
                     where="symbol = %s AND ts >= %s", params=("ABC", "2026-01-01"),
                     order_by="ts")

a["price"]   # float64 ndarray, native byte order, contiguous
a["ts"]      # datetime64[us]
```

## Exact, not approximately right

Speed-ups that change the numbers are bugs. Every value is the server's own binary representation, reinterpreted, so results are **bit-identical** to a row-by-row fetch, including NaN payloads and microsecond timestamps. The integration tests assert this against a real PostgreSQL on every push, and the benchmark refuses to print a time for any method whose output differs from the baseline.

NULLs get special care: `string_agg` silently drops a row if any packed field is NULL. The query therefore also returns `count(*)`, and `fetch_arrays` raises if the two disagree. Pass `allow_nulls=True` to substitute a per-type sentinel instead (NaN for floats, the type's minimum for integers, NaT for timestamps).

## Benchmark

```bash
pip install -e ".[dev]"
python benchmarks/bench.py --dsn postgresql://postgres@localhost/postgres --rows 2000000
```

Compares `psycopg` `fetchall()` → `np.array`, `COPY … csv` → `np.loadtxt`, `pandas.read_sql` and `pg-fastread`, reports the median of three runs and the number of mismatching values per method. CI runs it on every push; see the latest *Benchmark* step in the Actions tab.

Result from CI (GitHub-hosted runner, PostgreSQL 16 in a service container, 300,000 rows of `int8, timestamptz, float8, int4`, Python 3.10):

| method | seconds | speed-up | mismatches |
| --- | ---: | ---: | ---: |
| psycopg `fetchall` → `np.array` | 1.481 | 1.0× | 0 |
| `COPY … csv` → `np.loadtxt` | 0.625 | 2.4× | 0 |
| `pandas.read_sql` | 1.893 | 0.8× | 0 |
| **pg-fastread** (packed `bytea`) | **0.110** | **13.5×** | **0** |

## API

| function | does |
| --- | --- |
| `fetch_arrays(conn, table, columns, where=, params=, order_by=, allow_nulls=)` | one packed query → `dict[str, np.ndarray]` |
| `fetch_arrays_chunked(conn, table, columns, key=, bounds=)` | same, split into `[lo, hi)` key ranges (a `bytea` is capped at 1 GB) |
| `build_query(...)` / `decode(buf, columns)` | the two halves, for use with any driver |
| `Column(name, pg_type, expr=)` | a computed column, e.g. `Column("mid", "float8", expr="(bid + ask) / 2")` |

Supported types: `int2`, `int4`, `int8`, `float4`, `float8`, `bool`, `timestamp`, `timestamptz` (common aliases such as `bigint` and `double precision` work).

## When to use it, and when not

* **Good fit:** wide scans of numeric time series (ticks, candles, sensor data) into NumPy, Numba or Polars; backtests and feature pipelines that re-read the same history many times.
* **Not needed:** a few thousand rows, or text-heavy rows. `COPY … (FORMAT binary)` is the other fast path and streams without the 1 GB limit, but it needs a client-side parser for the PGCOPY format; this library keeps the decode to one `np.frombuffer` call.
* The server does a little more work (packing). The win comes from moving per-row work out of Python, so it grows with row count.

## Tests

```bash
pytest tests/test_decode.py                       # no database needed
PG_FASTREAD_DSN=postgresql://postgres@localhost/postgres pytest   # + end-to-end
```

MIT licence. Author: Mohammadreza Saeidi, who used the same technique to make tick reads in a production backtester 31–37× faster with zero mismatches; this is a clean-room, generic implementation.
