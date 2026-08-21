# harp-benchmarks

**Internal** parsing-speed benchmarks for the Harp register/payload API, run over every register defined in [`register_models.py`](src/harp/benchmarks/register_models.py), the device.yml coverage model, which is also imported by the acceptance tests under `tests/`.

> This package is **not published to PyPI**, since its `pyproject.toml` sets `classifiers` to `Private :: Do Not Upload`. It is a workspace member consumed only in dev/internal builds via the repo-root `pyproject.toml` `dev` dependency-group. Its console scripts ship with this package, so they never leak into the published `harp` distribution.

## Layout

| Path | Purpose |
| --- | --- |
| `src/harp/benchmarks/register_models.py` | Reference models for every device.yml register, with fixtures shared with the acceptance tests. |
| `src/harp/benchmarks/_registers.py` | Registry: each register plus a representative sample value, and artifact paths. |
| `src/harp/benchmarks/generate.py` | Writes `./benchmark/data/<Name>_<addr>.bin`, and exposes a cache-aware `ensure_corpus`. |
| `src/harp/benchmarks/benchmark.py` | Ensures corpora exist, then times `parse_bulk`, `parse_to_dataframe`, `payload_as_columns`; writes `./benchmark/report.md`. |

All generated artifacts, both corpora and report, are written under **`./benchmark`** in the current working directory, git-ignored and fully regenerable.

## Usage

Console scripts are declared in the `pyproject.toml` of this package:

```bash
# One command: generate if needed, honoring the cache, then benchmark and write the report.
uv run harp-benchmark
uv run harp-benchmark --runs 20
uv run harp-benchmark --entries 100000 --force            # rebuild smaller corpora
uv run harp-benchmark --only Version ComplexConfiguration

# Generate corpora explicitly. Optional, since harp-benchmark does this automatically.
uv run harp-benchmark-generate
uv run harp-benchmark-generate --entries 100000
```

Equivalent module invocations: `uv run python -m harp.benchmarks.benchmark` / `uv run python -m harp.benchmarks.generate`.

## What is measured

- **`parse_bulk`**, the core zero-copy strided-view parse into a `Batch` payload. This is **lazy**: it builds strided views only and runs **no** converters.
- **`parse_to_dataframe`**, the full path to a pandas `DataFrame`, with `copy=False`.
- **`payload_as_columns`**, decode only, with `parse_bulk` views built once up front and then only `payload.payload_as_columns()` timed. This is where the `converter.decode_batch` of each field actually runs, with no file read and no pandas construction.

`parse_bulk` and `parse_to_dataframe` are each timed in two modes:

- **pre-read**, file read once up front, so only deserialization is timed. This isolates library speed.
- **re-read**, file re-read from disk on every run, the real-world "load a dump" path, which includes disk.

The report also decomposes `parse_to_dataframe` into `parse_bulk` plus `payload_as_columns` plus pandas overhead.

`harp-benchmarks` is released as open source under the [MIT license](https://github.com/harp-tech/python/blob/main/LICENSE). Bug reports and contributions are welcome at [the GitHub repository](https://github.com/harp-tech/python).
