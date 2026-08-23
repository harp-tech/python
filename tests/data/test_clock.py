from contextlib import nullcontext

import numpy as np
import pytest
from harp.data.synchronization import decode_clock_from_samples, decode_clock_from_transitions

BAUD_RATE = 1000.0
HEADER = (0xAA, 0xAF)


def _packet_bits(value, *, header=False):
    """The RS-232 bit stream of one clock packet: start, 8 data bits (LSB first), stop."""
    payload = int(value).to_bytes(4, "little")
    data = bytes(HEADER) + payload if header else payload
    bits = []
    for byte in data:
        bits.append(0)
        bits.extend((byte >> position) & 1 for position in range(8))
        bits.append(1)
    return bits


def transitions(
    values,
    *,
    baud_rate=BAUD_RATE,
    emitted_baud_rate=None,
    period=1.0,
    first=0.5,
    header=False,
    jitter=0.0,
    seed=0,
):
    """An idle-high clock line carrying one packet per ``period``.

    Returns the transition times, the level taken at each of them, and the local time
    of the first falling edge of every packet. ``emitted_baud_rate`` sets the rate the
    signal is actually generated at, which may differ from the nominal ``baud_rate``.
    """
    bit = 1.0 / (emitted_baud_rate or baud_rate)
    rng = np.random.default_rng(seed)
    times, states, starts = [], [], []
    level = 1
    for packet, value in enumerate(values):
        start = first + packet * period
        starts.append(start)
        for position, expected in enumerate(_packet_bits(value, header=header)):
            if expected != level:
                offset = rng.uniform(-jitter, jitter) * bit if jitter else 0.0
                times.append(start + position * bit + offset)
                states.append(expected)
                level = expected
    return np.array(times), np.array(states), np.array(starts)


def render(times, states, sample_rate, *, duration=None):
    """Sample an idle-high transition list onto a uniform grid of digital states."""
    duration = duration if duration is not None else times[-1] + 0.5
    grid = np.arange(int(round(duration * sample_rate))) / sample_rate
    index = np.searchsorted(times, grid, side="right") - 1
    return np.where(index < 0, ~states[0].astype(bool), states[np.maximum(index, 0)]).astype(bool)


def packet_duration(*, baud_rate=BAUD_RATE, header=False):
    return len(_packet_bits(0, header=header)) / baud_rate


# ---------------------------------------------------------------- decoding


@pytest.mark.parametrize("anchor", ["last_bit", "first_edge"])
def test_decodes_a_single_packet(anchor):
    times, states, starts = transitions([3806874])
    clock = decode_clock_from_transitions(times, states, anchor=anchor)

    assert clock.index.name == "LocalTime"
    assert clock.columns.tolist() == ["Time"]
    assert clock["Time"].tolist() == [3806874]
    expected = starts[0] + (packet_duration() if anchor == "last_bit" else 0.0)
    assert clock.index.to_numpy() == pytest.approx([expected])


def test_decodes_consecutive_seconds():
    values = [3806874, 3806875, 3806876, 3806877]
    times, states, starts = transitions(values)

    clock = decode_clock_from_transitions(times, states)

    assert clock["Time"].tolist() == values
    assert clock.index.to_numpy() == pytest.approx(starts + packet_duration())


def test_header_framed_packets_are_detected():
    values = [1000, 1001, 1002]
    times, states, starts = transitions(values, header=True)

    clock = decode_clock_from_transitions(times, states)

    assert clock["Time"].tolist() == values
    assert clock.index.to_numpy() == pytest.approx(starts + packet_duration(header=True))


@pytest.mark.parametrize("value", [0, 1, 0xFF, 0xAFAA, 0xFF00FF00, 0xFFFFFFFE, 3806874])
def test_round_trips_every_byte_pattern(value):
    """Bytes of all zeros or all ones leave long runs without a transition."""
    times, states, _ = transitions([value, value + 1])

    clock = decode_clock_from_transitions(times, states)

    assert clock["Time"].tolist() == [value, value + 1]


@pytest.mark.parametrize("sample_rate", [5_000.0, 30_000.0])
def test_sampled_recording_matches_the_transition_list(sample_rate):
    values = [3806874, 3806875, 3806876]
    times, states, starts = transitions(values)
    samples = render(times, states, sample_rate)

    clock = decode_clock_from_samples(samples, sample_rate)

    assert clock["Time"].tolist() == values
    assert clock.index.name == "Sample"
    assert clock.index.dtype == np.int64
    assert clock.index.tolist() == [
        round(anchor * sample_rate) for anchor in starts + packet_duration()
    ]


def test_sample_index_points_at_the_anchor_sample():
    sample_rate = 10_000.0
    times, states, starts = transitions([42, 43], first=0.5)
    samples = render(times, states, sample_rate)

    clock = decode_clock_from_samples(samples, sample_rate, anchor="first_edge")

    assert clock.columns.tolist() == ["Time"]
    # the anchor sample is the first sample of the packet's start bit
    for sample, start in zip(clock.index, starts):
        assert sample == round(start * sample_rate)
        assert not samples[sample]
        assert samples[sample - 1]


def test_analog_recording_is_binarized_by_threshold():
    sample_rate = 10_000.0
    values = [7, 8]
    times, states, _ = transitions(values)
    analog = np.where(render(times, states, sample_rate), 3.3, 0.1) + 0.02

    clock = decode_clock_from_samples(analog, sample_rate, threshold=1.5)

    assert clock["Time"].tolist() == values


def test_float_samples_without_threshold_raises():
    with pytest.raises(ValueError, match="threshold"):
        decode_clock_from_samples(np.zeros(100, dtype=np.float64), 10_000.0)


def test_sample_rate_below_two_samples_per_bit_raises():
    with pytest.raises(ValueError, match="cannot be decoded"):
        decode_clock_from_samples(np.zeros(100, dtype=bool), 1500.0, baud_rate=BAUD_RATE)


def test_recording_without_any_transition_yields_no_anchors():
    clock = decode_clock_from_samples(np.ones(10_000, dtype=bool), 10_000.0)

    assert clock.empty
    assert clock.columns.tolist() == ["Time"]
    assert clock.index.name == "Sample"


def test_empty_transition_list_yields_no_anchors():
    clock = decode_clock_from_transitions([], [])

    assert clock.empty
    assert clock.columns.tolist() == ["Time"]
    assert clock["Time"].dtype == np.float64


def test_wrong_baud_rate_yields_no_anchors():
    times, states, _ = transitions([100, 101, 102])

    clock = decode_clock_from_transitions(times, states, baud_rate=9600.0)

    assert clock.empty
    assert clock.index.name == "LocalTime"


# ---------------------------------------------------------------- robustness


def test_glitches_between_packets_are_ignored():
    values = [500, 501]
    times, states, starts = transitions(values)
    # a 100 us spike in the idle line, well away from any packet
    times = np.concatenate([times, [1.4, 1.4001]])
    states = np.concatenate([states, [0, 1]])
    order = np.argsort(times)

    clock = decode_clock_from_transitions(times[order], states[order])

    assert clock["Time"].tolist() == values
    assert clock.index.to_numpy() == pytest.approx(starts + packet_duration())


def test_packet_truncated_by_the_end_of_the_recording_is_dropped():
    values = [500, 501, 502]
    times, states, starts = transitions(values)
    keep = times < starts[-1] + 0.5 * packet_duration()

    clock = decode_clock_from_transitions(times[keep], states[keep])

    assert clock["Time"].tolist() == values[:-1]


def test_recording_starting_mid_packet_keeps_the_later_packets():
    values = [500, 501, 502]
    times, states, starts = transitions(values)
    keep = times > starts[0] + 0.5 * packet_duration()

    clock = decode_clock_from_transitions(times[keep], states[keep])

    assert clock["Time"].tolist() == values[1:]


def test_tolerates_jitter_on_every_edge():
    values = [3806874, 3806875, 3806876]
    times, states, starts = transitions(values, jitter=0.2)

    clock = decode_clock_from_transitions(times, states)

    assert clock["Time"].tolist() == values
    assert clock.index.to_numpy() == pytest.approx(starts + packet_duration(), abs=0.3 / BAUD_RATE)


def test_tolerates_a_baud_rate_error_that_would_slip_a_whole_bit():
    """Per-frame re-synchronization keeps a 3% rate error from walking off the bits."""
    values = [3806874, 3806875]
    times, states, _ = transitions(values, emitted_baud_rate=BAUD_RATE * 1.03)

    clock = decode_clock_from_transitions(times, states, baud_rate=BAUD_RATE)

    assert clock["Time"].tolist() == values


def test_corrupt_packet_is_dropped_with_a_warning():
    values = [3806874, 3806875, 12345, 3806877]
    times, states, starts = transitions(values)

    with pytest.warns(UserWarning, match="inconsistent with the local clock"):
        clock = decode_clock_from_transitions(times, states)

    assert clock["Time"].tolist() == [3806874, 3806875, 3806877]
    assert clock.index.to_numpy() == pytest.approx(np.delete(starts, 2) + packet_duration())


def test_corrupt_first_packet_is_dropped_with_a_warning():
    values = [12345, 3806875, 3806876, 3806877]
    times, states, _ = transitions(values)

    with pytest.warns(UserWarning):
        clock = decode_clock_from_transitions(times, states)

    assert clock["Time"].tolist() == values[1:]


def test_max_drift_none_keeps_every_decoded_packet():
    values = [3806874, 3806875, 12345, 3806877]
    times, states, _ = transitions(values)

    clock = decode_clock_from_transitions(times, states, max_drift=None)

    assert clock["Time"].tolist() == values


def test_missing_packets_do_not_break_the_chain():
    values = [3806874, 3806875, 3806876, 3806877]
    times, states, starts = transitions(values)
    dropped = (times < starts[1]) | (times >= starts[2])

    clock = decode_clock_from_transitions(times[dropped], states[dropped])

    assert clock["Time"].tolist() == [3806874, 3806876, 3806877]


def test_invalid_transition_lists_raise():
    times, states, _ = transitions([1, 2])
    with pytest.raises(ValueError, match="same length"):
        decode_clock_from_transitions(times, states[:-1])
    with pytest.raises(ValueError, match="sorted"):
        decode_clock_from_transitions(times[::-1], states)
    with pytest.raises(ValueError, match="one-dimensional"):
        decode_clock_from_transitions(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="anchor"):
        decode_clock_from_transitions(times, states, anchor="middle")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="baud_rate"):
        decode_clock_from_transitions(times, states, baud_rate=0.0)


def test_a_long_drifting_sampled_recording_decodes_end_to_end():
    """A minute of a 30 kHz recording: local clock drift, jitter and one lost packet."""
    rate = 1.001  # Harp seconds per local second
    sample_rate = 30_000.0
    values = [3806874 + second for second in range(60)]
    times, states, starts = transitions(
        values, period=1.0 / rate, baud_rate=BAUD_RATE * rate, jitter=0.15
    )
    lost = (times >= starts[30]) & (times < starts[30] + 1.5 * packet_duration())
    samples = render(times[~lost], states[~lost], sample_rate)

    clock = decode_clock_from_samples(
        samples, sample_rate, baud_rate=BAUD_RATE * rate, anchor="first_edge"
    )

    assert clock["Time"].tolist() == [value for value in values if value != values[30]]
    # anchors land on the emitted start bits, within the jitter and one sample period
    assert clock.index.to_numpy() / sample_rate == pytest.approx(np.delete(starts, 30), abs=5e-4)


# ---------------------------------------------------------------- recorded signals

# Transitions acquired by an Open Ephys system from a Behavior board emitting the
# clock at 1 kbps, taken from https://github.com/harp-tech/harp-python/pull/38.
# fmt: off
RECORDED = [
    {
        # two valid packets, followed by one cut short by the end of the recording
        "timestamps": np.array([
            0.        , 0.96106667, 0.96306667, 0.96406667, 0.96506667,
            0.96706667, 0.96906667, 0.97106667, 0.97306667, 0.97506667,
            0.97606667, 0.97706667, 0.98006667, 0.98106667, 0.98306667,
            0.98406667, 0.98506667, 0.98806667, 0.99006667, 0.99106667,
            1.00006667, 1.96116667, 1.96216667, 1.96416667, 1.96516667,
            1.96716667, 1.96916667, 1.97116667, 1.97316667, 1.97516667,
            1.97616667, 1.97716667, 1.98016667, 1.98116667, 1.98316667,
            1.98416667, 1.98516667, 1.98816667, 1.99016667, 1.99116667,
            2.00016667, 2.96126667, 2.96426667, 2.96726667, 2.96926667,
            2.97126667, 2.97326667, 2.97526667, 2.97626667, 2.97726667,
            2.98026667, 2.98126667, 2.98326667, 2.98426667, 2.98526667,
            2.98826667, 2.99026667, 2.99126667]),
        "expected_start_times": np.array([0.96106667, 1.96116667]),
        "expected_harp_times": [3806874, 3806875],
        # the cut-short packet fails its framing check, so nothing reaches the drift check
        "expected_warning": False,
    },
    {
        # four valid packets and one corrupt one
        "timestamps": np.array([
            0.14036667, 1.10146667, 1.10246667, 1.10346667, 1.10446667,
            1.11146667, 1.11246667, 1.11646667, 1.11746667, 1.11846667,
            1.11946667, 1.12146667, 1.12246667, 1.12546667, 1.12746667,
            1.12846667, 1.13046667, 1.13146667, 1.14046667, 2.10156667,
            2.10356667, 2.11156667, 2.11256667, 2.11656667, 2.11756667,
            2.11856667, 2.11956667, 2.12156667, 2.12256667, 2.12556667,
            2.12756667, 2.12856667, 2.13056667, 2.13156667, 2.14056667,
            3.10163333, 3.10263333, 3.11163333, 3.11263333, 3.11663333,
            3.11763333, 3.11863333, 3.11963333, 3.12163333, 3.12263333,
            3.12563333, 3.12763333, 3.12863333, 3.13063333, 3.13163333,
            3.14063333, 4.10173333, 4.11073333, 4.11173333, 4.11673333,
            4.11873333, 4.11973333, 4.12173333, 4.12273333, 4.12573333,
            4.12773333, 4.12873333, 4.13073333, 4.13173333, 4.14073333,
            5.1018    , 5.1028    , 5.1038    , 5.1108    , 5.1118    ,
            5.1168    , 5.1188    , 5.1198    , 5.1218    , 5.1228    ,
            5.1258    , 5.1278    , 5.1288    , 5.1308    , 5.1318    ,
            5.1368    , 5.1388    , 5.1398    , 5.14183333, 5.1428    ,
            5.14583333, 5.14783333, 5.14883333, 5.15083333, 5.15183333,
            5.16083333, 6.1059    ]),
        "expected_start_times": np.array([1.10146667, 2.10156667, 3.10163333, 4.10173333]),
        "expected_harp_times": [2600957, 2600958, 2600959, 2600960],
        # the corrupt packet decodes cleanly but its seconds do not add up
        "expected_warning": True,
    },
]
# fmt: on


def _decode_recorded(recorded, **kwargs):
    times = recorded["timestamps"]
    states = np.resize([1, 0], times.size)  # transitions alternate, starting from high
    expected = (
        pytest.warns(UserWarning, match="inconsistent with the local clock")
        if recorded["expected_warning"]
        else nullcontext()
    )
    with expected:
        return decode_clock_from_transitions(times, states, **kwargs)


@pytest.mark.parametrize("recorded", RECORDED)
def test_decodes_a_recorded_clock_signal(recorded):
    clock = _decode_recorded(recorded, anchor="first_edge")

    assert clock["Time"].tolist() == recorded["expected_harp_times"]
    assert clock.index.to_numpy() == pytest.approx(recorded["expected_start_times"])


@pytest.mark.parametrize("recorded", RECORDED)
def test_anchors_a_recorded_signal_on_its_last_bit(recorded):
    clock = _decode_recorded(recorded)

    assert clock["Time"].tolist() == recorded["expected_harp_times"]
    assert clock.index.to_numpy() == pytest.approx(
        recorded["expected_start_times"] + packet_duration()
    )
