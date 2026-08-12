"""Aligning a non-Harp device's local timestamps to Harp time with the Harp Time Code."""

from ._decode import (
    DEFAULT_BAUD_RATE,
    decode_timecode_from_samples,
    decode_timecode_from_transitions,
)

__all__ = [
    "decode_timecode_from_samples",
    "decode_timecode_from_transitions",
    "DEFAULT_BAUD_RATE",
]
