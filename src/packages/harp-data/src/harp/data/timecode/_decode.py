import warnings
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

_BITS_PER_BYTE = 10  # 8N1: start bit, eight data bits, stop bit
_PACKET_BYTES = 4  # whole second as a little-endian U32
_RESYNC_WINDOW = 0.3  # max start-bit offset, in bits, to re-synchronize a frame on
_THRESHOLD_SUBSAMPLE = 1_000_000  # samples used to detect the threshold


def decode_timecode_from_transitions(
    timestamps: ArrayLike,
    states: ArrayLike,
    *,
    baud_rate: float = 1000.0,
) -> pd.DataFrame:
    """Decode a Harp Time Code from line transitions.

    See the [Harp Time Code specification](https://harp-tech.org/protocol/TimeCode.html).
    ``timestamps`` are the local times, in seconds, of each transition and ``states`` the
    level the line took (0 low, 1 high). Returns the whole Harp second of each packet
    (``"Time"``) indexed by the local time of its first falling edge (``"LocalTime"``).

    Packets with bad framing are dropped; every other packet is returned as decoded, so
    a clock reset or a corrupt value shows up as an anchor that does not line up.
    """
    times, levels = _as_transitions(timestamps, states)
    local, harp = _decode(times, levels, baud_rate=baud_rate)
    return pd.DataFrame({"Time": harp}, index=pd.Index(local, name="LocalTime"))


def decode_timecode_from_samples(
    samples: ArrayLike,
    sample_rate: float,
    *,
    threshold: float | None = None,
    baud_rate: float = 1000.0,
) -> pd.DataFrame:
    """Decode a Harp Time Code from a sampled line.

    See the [Harp Time Code specification](https://harp-tech.org/protocol/TimeCode.html).
    Sample ``i`` is taken at local time ``i / sample_rate``. Samples are binarized as
    ``samples >= threshold``; by default the threshold is detected from the data as the
    midpoint between the high and low levels. Returns the whole Harp second of each packet
    (``"Time"``) indexed by the sample of its first falling edge (``"Sample"``), filtered
    as in :func:`decode_timecode_from_transitions`.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    if sample_rate < 2 * baud_rate:
        raise ValueError(
            f"A {baud_rate} bps signal cannot be decoded from samples taken at "
            f"{sample_rate} Hz; at least 2 samples per bit are required (5 recommended)."
        )

    level = _as_levels(samples, threshold)
    edges = np.flatnonzero(np.diff(level)) + 1
    times = edges / sample_rate
    local, harp = _decode(times, level[edges], baud_rate=baud_rate)
    sample = np.rint(local * sample_rate).astype(np.int64)
    return pd.DataFrame({"Time": harp}, index=pd.Index(sample, name="Sample"))


def _as_transitions(
    timestamps: ArrayLike, states: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Validate transitions into sorted times and boolean levels."""
    times = np.asarray(timestamps, dtype=np.float64)
    levels = np.asarray(states)
    if times.ndim != 1 or levels.ndim != 1:
        raise ValueError("timestamps and states must be one-dimensional.")
    if times.size != levels.size:
        raise ValueError(
            f"timestamps and states must have the same length, got {times.size} and {levels.size}."
        )
    if np.any(np.diff(times) < 0):
        raise ValueError("timestamps must be sorted in ascending order.")
    return times, levels.astype(np.bool_)


def _as_levels(samples: ArrayLike, threshold: float | None) -> NDArray[np.bool_]:
    """Binarize a sampled waveform into digital line levels."""
    values = np.asarray(samples)
    if values.ndim != 1:
        raise ValueError("samples must be one-dimensional.")
    if threshold is None:
        if values.dtype == np.bool_:
            return values
        threshold = _detect_threshold(values)
    return values >= threshold


def _detect_threshold(values: NDArray[Any]) -> float:
    """Midpoint between the low and high levels of the line."""
    if values.size == 0:
        return 0.0
    if values.size > _THRESHOLD_SUBSAMPLE:
        # a fixed-seed random subsample is plenty, and a strided one could alias with the bits
        values = values[np.random.default_rng(0).integers(0, values.size, _THRESHOLD_SUBSAMPLE)]
    # extreme percentiles rather than min/max, so that rare glitches are ignored
    return float(np.percentile(values, [0.01, 99.99]).mean())


def _level_at(
    times: NDArray[np.float64], levels: NDArray[np.bool_], at: NDArray[np.floating[Any]]
) -> NDArray[np.bool_]:
    """Line level at each time in ``at``."""
    index = np.searchsorted(times, at, side="right") - 1
    # transitions alternate, so before the first one the line held the opposite level
    return np.where(index < 0, ~levels[0], levels[np.maximum(index, 0)])


def _resync(falling: NDArray[np.float64], expected: float, bit: float) -> float:
    """Snap a predicted start-bit time to a nearby falling edge."""
    best, window = expected, _RESYNC_WINDOW * bit
    index = int(np.searchsorted(falling, expected))
    for candidate in (index - 1, index):
        if 0 <= candidate < falling.size:
            distance = abs(falling[candidate] - expected)
            if distance <= window:
                best, window = float(falling[candidate]), distance
    return best


def _decode_packet(
    times: NDArray[np.float64],
    levels: NDArray[np.bool_],
    falling: NDArray[np.float64],
    origin: float,
    bit: float,
) -> tuple[int, float] | None:
    """Decode the packet starting at ``origin`` into (seconds, end time), or ``None``."""
    payload = bytearray(_PACKET_BYTES)
    offsets = (np.arange(_BITS_PER_BYTE) + 0.5) * bit
    start = origin
    for frame in range(_PACKET_BYTES):
        if frame:
            # predict from the previous start bit, not from the packet, so that a small
            # baud rate error stays a fraction of a bit instead of accumulating
            start = _resync(falling, start + _BITS_PER_BYTE * bit, bit)
        bits = _level_at(times, levels, start + offsets)
        if bits[0] or not bits[-1]:  # start bit must be low, stop bit high
            return None
        payload[frame] = np.packbits(bits[1:-1], bitorder="little")[0]
    return int.from_bytes(payload, "little"), start + _BITS_PER_BYTE * bit


def _decode(
    times: NDArray[np.float64],
    levels: NDArray[np.bool_],
    *,
    baud_rate: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Decode every packet into (first falling edge, Harp second) arrays."""
    if baud_rate <= 0:
        raise ValueError(f"baud_rate must be positive, got {baud_rate}.")
    local, harp = _decode_packets(times, levels, 1.0 / baud_rate)
    if local.size == 0 and np.any(~levels):
        warnings.warn(_explain_no_packets(times, levels, baud_rate), stacklevel=3)
    return local, harp


def _decode_packets(
    times: NDArray[np.float64], levels: NDArray[np.bool_], bit: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    falling = times[~levels]  # a packet can only start on a falling edge
    local: list[float] = []
    harp: list[float] = []
    decoded_until = -np.inf
    for origin in falling:
        if origin < decoded_until:
            continue  # inside a packet we already decoded
        packet = _decode_packet(times, levels, falling, float(origin), bit)
        if packet is None:
            continue
        seconds, decoded_until = packet
        local.append(float(origin))
        harp.append(float(seconds))
    return np.array(local, dtype=np.float64), np.array(harp, dtype=np.float64)


def _explain_no_packets(
    times: NDArray[np.float64], levels: NDArray[np.bool_], baud_rate: float
) -> str:
    """Guess why a line with falling edges yielded no packet."""
    message = f"No Harp Time Code packet decoded from {np.count_nonzero(~levels)} falling edges."
    if _decode_packets(times, ~levels, 1.0 / baud_rate)[0].size:
        return f"{message} The line looks inverted (idle low); invert it before decoding."
    pulses = np.diff(times)
    pulses = pulses[pulses > 0]
    if pulses.size == 0:
        return message
    # most pulses inside a packet are one bit long, so the short end of the
    # distribution estimates the bit period actually on the line
    apparent = 1.0 / float(np.percentile(pulses, 10))
    if abs(apparent / baud_rate - 1) <= 0.2:
        return f"{message} Its pulses match {baud_rate:g} bps, but no packet frames cleanly."
    rate = f"{apparent:,.0f}" if apparent >= 10 else f"{apparent:.2g}"
    message += f" Its pulses look like ~{rate} bps, not the {baud_rate:g} bps of baud_rate."
    if apparent < baud_rate / 100:
        message += " If the timestamps are sample numbers, divide them by the sample rate."
    return message
