"""Decode the Harp Synchronization Clock from a downsampled digital signal.

Harp devices share time over a dedicated serial bus running at 100 kbps (see the
[Synchronization Clock protocol](https://harp-tech.org/protocol/SynchronizationClock.html)).
Some emitters also mirror that signal on a digital output at a much lower baud rate —
typically 1 kbps — so that it can be recorded by non-Harp acquisition systems whose
sample rates are far below 100 kHz. The functions here decode such a recording back into
Harp seconds, paired with the local time each of them was received at.

The wire format is plain RS-232 without parity: the line idles high and every byte is
sent as one low start bit, eight data bits (least significant first) and one high stop
bit. The payload is the current Harp time in whole seconds (``uint32``, little-endian),
optionally preceded by the ``0xAA 0xAF`` header of the full protocol packet — both
framings are recognized.

Two ways in, depending on what the recording system gives you:

- a uniformly sampled waveform (digital or analog) → :func:`decode_clock_from_samples`
- a list of line transitions ``(time, 0/1)`` → :func:`decode_clock_from_transitions`

Each returns a table of anchors keyed on the axis that system reports its own timestamps
on — a sample number or local seconds — against the whole Harp second received at it. How
those timestamps are then interpolated onto the Harp axis is left to the caller.
"""

import warnings
from typing import Any, Literal, Union, get_args

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

#: Baud rate used by existing downsampled clock emitters, in bits per second.
DEFAULT_BAUD_RATE = 1000.0

#: Bits per byte on the wire: one start bit, eight data bits, one stop bit.
_BITS_PER_BYTE = 10

#: Bytes carrying the whole-second payload of a clock packet.
_PAYLOAD_BYTES = 4

#: Leading bytes of a full Harp Synchronization Clock packet, when present.
_HEADER = (0xAA, 0xAF)

#: How far, as a fraction of a bit period, a start bit may sit from its predicted
#: position and still be used to re-synchronize the frame that follows it.
_RESYNC_WINDOW = 0.3

#: Where in the packet the whole second is taken to lapse. ``"last_bit"`` anchors on
#: the end of the last transmitted bit, mirroring the Harp Synchronization Clock where
#: the last byte carries the synchronization event; ``"first_edge"`` anchors on the
#: first falling edge, which is what current emitters align to the second boundary.
ClockAnchor = Literal["last_bit", "first_edge"]

_ANCHORS = get_args(ClockAnchor)


def decode_clock_from_transitions(
    timestamps: ArrayLike,
    states: ArrayLike,
    *,
    baud_rate: float = DEFAULT_BAUD_RATE,
    anchor: ClockAnchor = "last_bit",
    max_drift: Union[float, None] = 0.1,
) -> pd.DataFrame:
    """Decode Harp seconds from a list of clock-line transitions.

    ``timestamps`` are the local times, in seconds, of each transition of the clock
    line and ``states`` the level the line took at each of them (0 low, 1 high); this
    is the shape most event-based acquisition systems report. The returned DataFrame
    maps the local time each decoded packet is anchored on (the index, named
    ``"LocalTime"``, see ``anchor``) to the whole Harp second it carries (``"Time"``,
    the name the readers also give the Harp time axis).

    ``baud_rate`` must match the emitter; bits are read at the center of each bit
    period and every frame re-synchronizes on its own start bit, so a mismatch of a
    few percent is tolerated. Packets whose start/stop bits do not check out are
    dropped silently — a partially recorded or glitched packet costs one anchor, not
    the decoding of the ones around it. ``max_drift`` additionally drops packets whose
    seconds do not advance consistently with the local clock, allowing for at most
    that fractional rate mismatch between the two (pass ``None`` to keep everything).
    """
    times, levels = _as_transitions(timestamps, states)
    local, harp = _decode(times, levels, baud_rate=baud_rate, anchor=anchor)
    local, harp = _drop_inconsistent(local, harp, max_drift)
    return _clock_frame(pd.Index(local, name="LocalTime", dtype=np.float64), harp)


def decode_clock_from_samples(
    samples: ArrayLike,
    sample_rate: float,
    *,
    threshold: Union[float, None] = None,
    baud_rate: float = DEFAULT_BAUD_RATE,
    anchor: ClockAnchor = "last_bit",
    max_drift: Union[float, None] = 0.1,
) -> pd.DataFrame:
    """Decode Harp seconds from a uniformly sampled recording of the clock line.

    ``samples`` is the recorded waveform and ``sample_rate`` its sampling frequency in
    Hz; sample ``i`` is taken to occur at local time ``i / sample_rate``. Boolean and
    integer input is treated as the digital line state (nonzero is high); pass
    ``threshold`` to binarize an analog recording as ``samples >= threshold``.

    The returned DataFrame maps the sample each decoded packet is anchored on (the
    index, named ``"Sample"``, rounded to the nearest sample) to the whole Harp second
    it carries (``"Time"``). See :func:`decode_clock_from_transitions` for ``anchor``
    and ``max_drift``. About five samples per bit are needed for reliable decoding, so
    a 1 kbps signal wants a sample rate of at least ~5 kHz.
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
    local, harp = _decode(times, level[edges], baud_rate=baud_rate, anchor=anchor)
    local, harp = _drop_inconsistent(local, harp, max_drift)
    sample = np.rint(local * sample_rate).astype(np.int64)
    return _clock_frame(pd.Index(sample, name="Sample"), harp)


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


def _as_levels(samples: ArrayLike, threshold: Union[float, None]) -> NDArray[np.bool_]:
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


def _decode_bytes(
    times: NDArray[np.float64],
    levels: NDArray[np.bool_],
    falling: NDArray[np.float64],
    origin: float,
    bit: float,
    count: int,
) -> Union[tuple[NDArray[np.uint8], NDArray[np.float64]], None]:
    """Read ``count`` back-to-back RS-232 frames starting at ``origin``.

    Returns the decoded bytes and the start-bit time of each frame, or ``None`` if any
    of the frames is not properly delimited by a low start bit and a high stop bit.
    """
    values = np.empty(count, dtype=np.uint8)
    origins = np.empty(count, dtype=np.float64)
    offsets = (np.arange(_BITS_PER_BYTE) + 0.5) * bit
    start = origin
    for frame in range(count):
        if frame:
            # predict from the previous start bit, not from the packet, so that a small
            # baud rate error stays a fraction of a bit instead of accumulating
            start = _resync(falling, start + _BITS_PER_BYTE * bit, bit)
        bits = _level_at(times, levels, start + offsets)
        if bits[0] or not bits[-1]:  # start bit must be low, stop bit high
            return None
        origins[frame] = start
        values[frame] = np.packbits(bits[1:-1], bitorder="little")[0]
    return values, origins


def _decode_packet(
    times: NDArray[np.float64],
    levels: NDArray[np.bool_],
    falling: NDArray[np.float64],
    origin: float,
    bit: float,
) -> Union[tuple[int, NDArray[np.float64]], None]:
    """Decode one clock packet starting at ``origin``, with or without header."""
    headed = _decode_bytes(times, levels, falling, origin, bit, len(_HEADER) + _PAYLOAD_BYTES)
    if headed is not None and tuple(headed[0][: len(_HEADER)]) == _HEADER:
        payload, origins = headed[0][len(_HEADER) :], headed[1]
    else:
        bare = _decode_bytes(times, levels, falling, origin, bit, _PAYLOAD_BYTES)
        if bare is None:
            return None
        payload, origins = bare
    return int.from_bytes(payload.tobytes(), "little"), origins


def _decode(
    times: NDArray[np.float64],
    levels: NDArray[np.bool_],
    *,
    baud_rate: float,
    anchor: ClockAnchor,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Decode every clock packet in a transition list into (local time, Harp time)."""
    if anchor not in _ANCHORS:
        raise ValueError(f"anchor must be one of {_ANCHORS}, got {anchor!r}.")
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
        seconds, origins = packet
        end = origins[-1] + _BITS_PER_BYTE * bit
        decoded_until = end
        local.append(end if anchor == "last_bit" else float(origin))
        harp.append(float(seconds))
    return np.array(local, dtype=np.float64), np.array(harp, dtype=np.float64)


def _drop_inconsistent(
    local: NDArray[np.float64], harp: NDArray[np.float64], max_drift: Union[float, None]
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
            f"Dropped {dropped} clock packet(s) whose Harp time is inconsistent with the "
            "local clock. Pass max_drift=None to keep them.",
            stacklevel=3,
        )
    return local[keep], harp[keep]


def _clock_frame(anchors: pd.Index, harp: NDArray[np.float64]) -> pd.DataFrame:
    """Assemble the decoded packets into an ``anchors`` → Harp time table.

    ``anchors`` locate each packet on the local device's own axis — whichever one it
    reports timestamps on. "Time" is the Harp time axis throughout ``harp.data``; here
    it is the column, because the index is the axis being translated from.
    """
    return pd.DataFrame({"Time": harp}, index=anchors)
