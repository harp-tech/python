"""Acceptance tests: the device.yml coverage model in
``harp.benchmarks.register_models`` round-trips, and the new API behaviors
(offsets/gaps, masked overlap, single-member unwrap, strict enums) hold.
"""

import numpy as np
from harp.data import payload_to_dataframe
from harp.protocol import HarpMessage, HarpVersion
from harp.benchmarks.register_models import (
    AnalogData,
    AnalogDataPayload,
    BitmaskSplitter,
    BitmaskSplitterPayload,
    ComplexConfiguration,
    ComplexConfigurationPayload,
    Counter0,
    CustomMemberConverter,
    CustomMemberConverterPayload,
    CustomPayload,
    CustomPayloadPayload,
    DigitalInputs,
    EncoderMode,
    EncoderModeMask,
    PortDIOSet,
    PortDIOSetPayload,
    PortDigitalIOS,
    PulseDO0,
    PulseDOPort0,
    PwmPort,
    StartPulse,
    StartPulsePayload,
    StartPulseTrain,
    StartPulseTrainPayload,
    Version,
    VersionPayload,
)


def _roundtrip(register, value):
    return register.parse(HarpMessage.parse(register.format(value)))


# ---------------------------------------------------------------------------
# Every register round-trips (format -> parse)
# ---------------------------------------------------------------------------


def test_scalar_registers_roundtrip():
    assert int(_roundtrip(DigitalInputs, np.uint8(0b1010))) == 0b1010
    assert int(_roundtrip(Counter0, np.int32(-100000))) == -100000
    assert int(_roundtrip(PulseDOPort0, np.uint16(5))) == 5
    assert int(_roundtrip(PulseDO0, np.uint16(9))) == 9


def test_analog_data_roundtrip():
    ad = AnalogDataPayload(
        analog0=np.float32(1.0),
        analog1=np.float32(2.0),
        analog2=np.float32(3.0),
        accelerometer=np.array([4, 5, 6], dtype=np.float32),
    )
    p = _roundtrip(AnalogData, ad)
    assert float(p.analog0) == 1.0 and float(p.analog2) == 3.0
    np.testing.assert_array_equal(p.accelerometer, [4, 5, 6])
    assert AnalogDataPayload.payload_dtype.itemsize == 24  # 6 floats


def test_version_roundtrip():
    ver = VersionPayload(
        protocol_version=HarpVersion(2, 0, 0),
        firmware_version=HarpVersion(1, 2, 3),
        hardware_version=HarpVersion(1, 0, 0),
        core_id="abc",
        interface_hash=np.arange(20, dtype=np.uint8),
    )
    p = _roundtrip(Version, ver)
    assert p.protocol_version == HarpVersion(2, 0, 0)
    assert p.core_id == "abc"
    np.testing.assert_array_equal(p.interface_hash, np.arange(20))
    assert VersionPayload.payload_dtype.itemsize == 32


def test_custom_member_converter_roundtrip():
    p = _roundtrip(
        CustomMemberConverter, CustomMemberConverterPayload(header=np.uint8(7), data=-1234)
    )
    assert int(p.header) == 7 and int(p.data) == -1234


def test_encoder_mode_roundtrip():
    # Single whole-register groupMask -> parse() unwraps to the bare enum.
    p = _roundtrip(EncoderMode, EncoderModeMask.DISPLACEMENT)
    assert p == EncoderModeMask.DISPLACEMENT
    assert isinstance(p, EncoderModeMask)


# ---------------------------------------------------------------------------
# Offsets + gaps (ComplexConfiguration)
# ---------------------------------------------------------------------------


def test_complex_configuration_gap_and_offsets():
    cc = ComplexConfigurationPayload(
        pwm_port=PwmPort.PWM2,
        duty_cycle=np.float32(0.5),
        frequency=np.float32(1000.0),
        events_enabled=True,
        delta=np.uint32(42),
    )
    # itemsize from the register length (17), not the member extent.
    assert ComplexConfigurationPayload.payload_dtype.itemsize == 17
    # bytes 1..3 are an uncovered gap, preserved on encode.
    assert cc.payload_array.tobytes()[1:4] == b"\x00\x00\x00"
    # explicit byte offsets (base element = uint8, so element units == bytes).
    fields = ComplexConfigurationPayload.payload_dtype.fields
    assert fields["duty_cycle"][1] == 4
    assert fields["delta"][1] == 13

    p = _roundtrip(ComplexConfiguration, cc)
    assert p.pwm_port == PwmPort.PWM2
    assert float(p.duty_cycle) == 0.5
    assert p.events_enabled is True
    assert int(p.delta) == 42


# ---------------------------------------------------------------------------
# Masked overlap on one element (StartPulse / StartPulseTrain / BitmaskSplitter)
# ---------------------------------------------------------------------------


def test_start_pulse_overlapping_masks():
    # Two views of one U16 element share storage (one numpy field, itemsize 2).
    assert StartPulsePayload.payload_dtype.itemsize == 2
    assert len(StartPulsePayload.payload_dtype.names) == 1
    p = _roundtrip(
        StartPulse, StartPulsePayload(digital_output=PwmPort.PWM1, pulse_width=np.uint16(300))
    )
    assert p.digital_output == PwmPort.PWM1
    assert int(p.pulse_width) == 300


def test_start_pulse_train_two_words_and_default():
    p = _roundtrip(
        StartPulseTrain,
        StartPulseTrainPayload(
            digital_output=PwmPort.PWM1,
            pulse_width=np.uint16(300),
            frequency=np.uint8(200),
            pulse_count=np.uint8(50),
        ),
    )
    assert p.digital_output == PwmPort.PWM1 and int(p.pulse_width) == 300
    assert int(p.frequency) == 200 and int(p.pulse_count) == 50
    assert StartPulseTrainPayload.payload_dtype.itemsize == 4  # two U16 words
    # defaultValue: frequency defaults to 1 when not provided.
    assert int(StartPulseTrainPayload(pulse_count=np.uint8(3)).frequency) == 1


def test_bitmask_splitter_masked_ints():
    p = _roundtrip(BitmaskSplitter, BitmaskSplitterPayload(low=0xA, high=0x5))
    assert int(p.low) == 0xA and int(p.high) == 0x5
    assert p.payload_array.tobytes() == bytes([0x5A])  # high packs into the top nibble


def test_port_dio_set_bitmask():
    # Single whole-register bitMask -> parse() unwraps to the bare IntFlag.
    p = _roundtrip(PortDIOSet, PortDigitalIOS.DIO0 | PortDigitalIOS.DIO3)
    assert p == PortDigitalIOS.DIO0 | PortDigitalIOS.DIO3
    assert PortDigitalIOS.DIO1 not in p
    assert PortDIOSetPayload.payload_dtype.itemsize == 1


# ---------------------------------------------------------------------------
# Register-level interfaceType: single full-span member unwraps on parse
# ---------------------------------------------------------------------------


def test_custom_payload_single_member_unwrap():
    # Root payload: single __value__ view, unwrapped on parse.
    assert CustomPayloadPayload._single_member == "__value__"
    assert CustomPayloadPayload._root is True
    # U32[3] HarpVersion -> 12-byte buffer (3 x u32), same converter class.
    assert CustomPayloadPayload.payload_dtype.itemsize == 12
    parsed = _roundtrip(CustomPayload, CustomPayloadPayload(HarpVersion(3, 1, 4)))
    assert isinstance(parsed, HarpVersion)
    assert parsed == HarpVersion(3, 1, 4)


# ---------------------------------------------------------------------------
# Undefined masked enum codes are preserved as their raw int (permissive, like C#)
# ---------------------------------------------------------------------------


def test_unknown_enum_code_preserves_raw():
    # StartPulse.digital_output is a 2-bit field; code 0b11 has no PwmPort member.
    raw = np.array(0b11 << 10, dtype=np.uint16).tobytes()
    payload = StartPulsePayload.payload_from_buffer(raw)
    value = payload.digital_output  # permissive: the raw code is kept, not raised
    assert value == 0b11
    assert not isinstance(value, PwmPort)


# ---------------------------------------------------------------------------
# to_dataframe over masked + offset payloads
# ---------------------------------------------------------------------------


def test_complex_configuration_to_dataframe():
    cc = ComplexConfigurationPayload(
        pwm_port=PwmPort.PWM2,
        duty_cycle=np.float32(0.5),
        frequency=np.float32(1.0),
        events_enabled=True,
        delta=np.uint32(42),
    )
    batch = ComplexConfigurationPayload.payload_from_buffer(cc.payload_array.tobytes() * 2)
    df = payload_to_dataframe(batch)
    assert len(df) == 2
    assert list(df["pwm_port"]) == ["PWM2", "PWM2"]
    np.testing.assert_array_equal(df["delta"], [42, 42])
