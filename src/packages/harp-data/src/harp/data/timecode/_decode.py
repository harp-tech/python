"""Decode a recorded Harp Time Code back into Harp seconds.

The [Harp Time Code](https://harp-tech.org/protocol/TimeCode.html) is a serial signal
carrying the current Harp time in whole seconds, emitted on a digital output at a low baud
rate (typically 1 kbps) so that non-Harp acquisition systems can record it alongside data
timestamped on their own clock. The functions here decode such a recording back into Harp
seconds, paired with the local time each of them was received at.

The wire format is 8N1 serial: the line idles high and every byte is sent as one low start
bit, eight data bits (least significant first) and one high stop bit. Each packet is four
back-to-back bytes holding the whole Harp second as a little-endian ``uint32``, with no
header or checksum, and its first falling edge marks the start of that second.

Two ways in, depending on what the recording system gives you:

- a uniformly sampled waveform (digital or analog) → :func:`decode_timecode_from_samples`
- a list of line transitions ``(time, 0/1)`` → :func:`decode_timecode_from_transitions`

Each returns a table of anchors keyed on the axis that system reports its own timestamps
on (a sample number or local seconds) against the whole Harp second that starts there.
How those timestamps are then interpolated onto the Harp axis is left to the caller.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

#: Baud rate used by existing Harp Time Code emitters, in bits per second.
DEFAULT_BAUD_RATE = 1000.0

#: Bits per byte on the wire: one start bit, eight data bits, one stop bit.
_BITS_PER_BYTE = 10

#: Bytes in a Harp Time Code packet, holding the whole second as a ``U32``.
_PACKET_BYTES = 4

#: How far, as a fraction of a bit period, a start bit may sit from its predicted
#: position and still be used to re-synchronize the frame that follows it.
_RESYNC_WINDOW = 0.3


def decode_timecode_from_transitions(
    timestamps: ArrayLike,
    states: ArrayLike,
    *,
    baud_rate: float = DEFAULT_BAUD_RATE,
    max_drift: float | None = 0.1,
) -> pd.DataFrame:
    """Decode Harp seconds from a list of time code line transitions.

    ``timestamps`` are the local times, in seconds, of each transition of the time code
    line and ``states`` the level the line took at each of them (0 low, 1 high); this is
    the shape most event-based acquisition systems report. The returned DataFrame maps the
    local time of each decoded packet's first falling edge (the index, named
    ``"LocalTime"``) to the whole Harp second that starts there (``"Time"``, the name the
    readers also give the Harp time axis).

    ``baud_rate`` must match the emitter; bits are read at the center of each bit period
    and every frame re-synchronizes on its own start bit, so a mismatch of a few percent
    is tolerated. Packets whose start/stop bits do not check out are dropped silently, so
    a partially recorded or glitched packet costs one anchor, not the decoding of the ones
    around it. ``max_drift`` additionally drops packets whose seconds do not advance
    consistently with the local clock, allowing for at most that fractional rate mismatch
    between the two (pass ``None`` to keep everything).
    """
    times, levels = _as_transitions(timestamps, states)
    local, harp = _decode(times, levels, baud_rate=baud_rate)
    local, harp = _drop_inconsistent(local, harp, max_drift)
    return _anchor_frame(pd.Index(local, name="LocalTime", dtype=np.float64), harp)


def decode_timecode_from_samples(
    samples: ArrayLike,
    sample_rate: float,
    *,
    threshold: float | None = None,
    baud_rate: float = DEFAULT_BAUD_RATE,
    max_drift: float | None = 0.1,
) -> pd.DataFrame:
    """Decode Harp seconds from a uniformly sampled recording of the time code line.

    ``samples`` is the recorded waveform and ``sample_rate`` its sampling frequency in Hz;
    sample ``i`` is taken to occur at local time ``i / sample_rate``. Boolean and integer
    input is treated as the digital line state (nonzero is high); pass ``threshold`` to
    binarize an analog recording as ``samples >= threshold``.

    The returned DataFrame maps the first sample of each decoded packet's start bit (the
    index, named ``"Sample"``) to the whole Harp second that starts there (``"Time"``).
    See :func:`decode_timecode_from_transitions` for ``baud_rate`` and ``max_drift``.
    About five samples per bit are needed for reliable decoding, so a 1 kbps signal wants
    a sample rate of at least ~5 kHz.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    if baud_rate <= 0:
        raise ValueError(f"baud_rate must be positive, got {baud_rate}.")
    if sample_rate < 2 * baud_rate:
        raise ValueError(
            f"A {baud_rate} bps signal cannot be decoded from samples taken at "
            f"{sample_rate} Hz; at least 2 samples per bit are required (5 recommended)."
        )

    level = _as_levels(samples, threshold)
    edges = np.flatnonzero(np.diff(level)) + 1
    times = edges / sample_rate
    local, harp = _decode(times, level[edges], baud_rate=baud_rate)
    local, harp = _drop_inconsistent(local, harp, max_drift)
    sample = np.rint(local * sample_rate).astype(np.int64)
    return _anchor_frame(pd.Index(sample, name="Sample"), harp)


def _as_transitions(
    timestamps: ArrayLike, states: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Validate a transition list into sorted times and boolean levels."""
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
    if threshold is not None:
        return values >= threshold
    if values.dtype == np.bool_ or np.issubdtype(values.dtype, np.integer):
        return values != 0
    raise ValueError(
        f"Pass a threshold to binarize samples of dtype {values.dtype}; only boolean "
        "and integer samples are taken as digital line states."
    )


def _level_at(
    times: NDArray[np.float64], levels: NDArray[np.bool_], at: NDArray[np.floating[Any]]
) -> NDArray[np.bool_]:
    """The line level at each time in ``at``, from the transitions ``(times, levels)``."""
    index = np.searchsorted(times, at, side="right") - 1
    # transitions alternate, so before the first one the line held the opposite level
    return np.where(index < 0, ~levels[0], levels[np.maximum(index, 0)])


def _resync(falling: NDArray[np.float64], expected: float, bit: float) -> float:
    """Snap a predicted start-bit time to a nearby falling edge, as a UART would."""
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
    """Decode one time code packet whose first start bit falls at ``origin``.

    Returns the whole second it carries and the time its last stop bit ends, or ``None``
    if any of its frames is not delimited by a low start bit and a high stop bit.
    """
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
    """Decode every packet in a transition list into (first falling edge, Harp second)."""
    if baud_rate <= 0:
        raise ValueError(f"baud_rate must be positive, got {baud_rate}.")

    bit = 1.0 / baud_rate
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


def _drop_inconsistent(
    local: NDArray[np.float64], harp: NDArray[np.float64], max_drift: float | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Drop anchors whose Harp seconds disagree with the elapsed local time."""
    if max_drift is None or local.size < 2:
        return local, harp
    if max_drift < 0:
        raise ValueError(f"max_drift must be non-negative, got {max_drift}.")

    def agrees(earlier: int, later: int) -> bool:
        """Whether two anchors are one plausible stretch of the same clock apart."""
        elapsed_harp = harp[later] - harp[earlier]
        elapsed_local = local[later] - local[earlier]
        return bool(
            elapsed_harp >= 1
            and abs(elapsed_harp - elapsed_local) <= max_drift * abs(elapsed_local)
        )

    # grow the longest chain of mutually consistent anchors out of the first agreeing
    # pair, so that one corrupt value costs one anchor instead of every anchor after it
    keep = np.zeros(local.size, dtype=np.bool_)
    seed = next((i for i in range(local.size - 1) if agrees(i, i + 1)), None)
    if seed is not None:
        keep[seed] = keep[seed + 1] = True
        last = seed + 1
        for candidate in range(seed + 2, local.size):
            if agrees(last, candidate):
                keep[candidate] = True
                last = candidate
        first = seed
        for candidate in range(seed - 1, -1, -1):
            if agrees(candidate, first):
                keep[candidate] = True
                first = candidate

    dropped = int(np.count_nonzero(~keep))
    if dropped:
        warnings.warn(
            f"Dropped {dropped} time code packet(s) whose Harp time is inconsistent with "
            "the local clock. Pass max_drift=None to keep them.",
            stacklevel=3,
        )
    return local[keep], harp[keep]


def _anchor_frame(anchors: pd.Index, harp: NDArray[np.float64]) -> pd.DataFrame:
    """Assemble the decoded packets into an ``anchors`` → Harp time table.

    ``anchors`` locate each packet on the local device's own axis, whichever one it
    reports timestamps on. "Time" is the Harp time axis throughout ``harp.data``; here
    it is the column, because the index is the axis being translated from.
    """
    return pd.DataFrame({"Time": harp}, index=anchors)
