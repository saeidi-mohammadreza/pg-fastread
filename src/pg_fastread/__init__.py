"""pg-fastread: read PostgreSQL rows into NumPy arrays through one packed bytea."""
from .core import (
    PG_TYPES,
    Column,
    build_query,
    decode,
    fetch_arrays,
    fetch_arrays_chunked,
    record_dtype,
)

__all__ = [
    "PG_TYPES",
    "Column",
    "build_query",
    "decode",
    "fetch_arrays",
    "fetch_arrays_chunked",
    "record_dtype",
]
__version__ = "0.1.0"
