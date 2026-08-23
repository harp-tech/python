"""Tests for _register.py and round-trips between register format/parse."""

from typing import ClassVar

import numpy as np
import pytest
from harp.data import parse_to_dataframe, payload_to_dataframe, to_buffer, to_file
from harp.protocol._message import HarpMessage, HarpParseError
from harp.protocol._message_type import MessageType
from harp.protocol._payload import (
    PayloadBase,
    PayloadFloat,
    PayloadS8,
    PayloadS16,
    PayloadS32,
    PayloadS64,
    PayloadU8,
    PayloadU16,
    PayloadU32,
    PayloadU64,
    Field,
    _IdentityConverter,
)
from harp.protocol._payload_type import PayloadType
from harp.protocol._register import (
    RegisterBase,
    RegisterFloat,
    RegisterS8,
    RegisterS16,
    RegisterS16Array,
    RegisterS32,
    RegisterS64,
    RegisterU8,
    RegisterU16,
    RegisterU16Array,
    RegisterU32,
    RegisterU32Array,
    RegisterU64,
)
# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class TimestampSecond(RegisterU32):
    address: ClassVar[int] = 8


class DigitalOutputSet(RegisterU16):
    address: ClassVar[int] = 32


class AnalogDataPayload(PayloadBase):
    analog_input0 = Field(converter=_IdentityConverter("<i2"), offset=0)
    encoder = Field(converter=_IdentityConverter("<i2"), offset=2)
    analog_input1 = Field(converter=_IdentityConverter("<i2"), offset=4)


class AnalogData(RegisterBase[AnalogDataPayload]):
    address: ClassVar[int] = 33
    payload_type: ClassVar[PayloadType] = PayloadType.S16
    payload_class: ClassVar = AnalogDataPayload


def _parse_frame(frame: bytes) -> HarpMessage:
    return HarpMessage.parse(frame)


@pytest.mark.parametrize(
    "reg_cls, address, payload_type, value, dtype",
    [
        (RegisterU8, 0x10, PayloadType.U8, 42, np.dtype("u1")),
        (RegisterU16, 0x11, PayloadType.U16, 1000, np.dtype("<u2")),
        (RegisterU32, 0x12, PayloadType.U32, 99999, np.dtype("<u4")),
        (RegisterU64, 0x13, PayloadType.U64, 2**32, np.dtype("<u8")),
        (RegisterS8, 0x14, PayloadType.S8, -5, np.dtype("i1")),
        (RegisterS16, 0x15, PayloadType.S16, -300, np.dtype("<i2")),
        (RegisterS32, 0x16, PayloadType.S32, -100000, np.dtype("<i4")),
        (RegisterS64, 0x17, PayloadType.S64, -(2**33), np.dtype("<i8")),
        (RegisterFloat, 0x18, PayloadType.Float, 3.14, np.dtype("<f4")),
    ],
)
def test_scalar_register_format_write(reg_cls, address, payload_type, value, dtype):
    # format(value) produces a parseable Write frame with the correct payload.
    reg = reg_cls(address)
    frame = reg.format(value)
    msg = _parse_frame(frame)
    assert msg.message_type == MessageType.Write
    assert msg.address == address
    assert msg.payload_type == payload_type
    expected = np.asarray(value, dtype=dtype).tobytes()
    assert msg.payload_bytes == expected


@pytest.mark.parametrize(
    "reg_cls, address, payload_type",
    [
        (RegisterU8, 0x10, PayloadType.U8),
        (RegisterU16, 0x11, PayloadType.U16),
        (RegisterU32, 0x12, PayloadType.U32),
        (RegisterU64, 0x13, PayloadType.U64),
        (RegisterS8, 0x14, PayloadType.S8),
        (RegisterS16, 0x15, PayloadType.S16),
        (RegisterS32, 0x16, PayloadType.S32),
        (RegisterS64, 0x17, PayloadType.S64),
        (RegisterFloat, 0x18, PayloadType.Float),
    ],
)
def test_scalar_register_format_read(reg_cls, address, payload_type):
    # format() (no value) produces a Read frame with empty payload.
    reg = reg_cls(address)
    frame = reg.format()
    msg = _parse_frame(frame)
    assert msg.message_type == MessageType.Read
    assert msg.address == address
    assert msg.payload_bytes == b""


@pytest.mark.parametrize("value", [0, 1, 2**32 - 1])
def test_named_register_roundtrip(value):
    # TimestampSecond write frame parses back to the same value.
    frame = TimestampSecond.format(value)
    msg = _parse_frame(frame)
    parsed = TimestampSecond.parse(msg)
    # Anonymous-payload registers unwrap to a numpy scalar.
    assert isinstance(parsed, np.uint32)
    assert parsed == value
    assert parsed.ndim == 0


def test_named_register_format_read_frame():
    frame = TimestampSecond.format()
    msg = _parse_frame(frame)
    assert msg.message_type == MessageType.Read
    assert msg.address == 8


def test_factory_creates_subclass():
    reg = RegisterU32(0x08)
    assert issubclass(reg, RegisterU32)
    assert reg.address == 0x08


def test_factory_format_and_parse():
    reg = RegisterU32(0x08)
    frame = reg.format(100)
    msg = _parse_frame(frame)
    parsed = reg.parse(msg)
    assert isinstance(parsed, np.uint32)
    assert parsed == 100
    assert parsed.ndim == 0


def test_factory_different_addresses_are_independent():
    r1 = RegisterU32(0x08)
    r2 = RegisterU32(0x09)
    assert r1.address != r2.address
    assert r1 is not r2


def test_declared_register_raises_type_error():
    # A declared address cannot be reassigned.
    declared = RegisterU32(0x08)
    with pytest.raises(TypeError, match="already declares address"):
        declared(0x09)


def test_declared_array_register_raises_type_error():
    declared = RegisterU32Array(0x28, length=3)
    with pytest.raises(TypeError, match="already declares address"):
        declared(0x29, length=3)


def test_register_repr_shows_name_and_address():
    assert repr(RegisterU32(0x08)) == "<RegisterU32_0x08 @8>"
    # A base declares no address, so it keeps the default class repr.
    assert repr(RegisterU32).startswith("<class ")


@pytest.mark.parametrize(
    "reg_cls, payload_cls, value",
    [
        (RegisterU8, PayloadU8, 255),
        (RegisterU16, PayloadU16, 1000),
        (RegisterU32, PayloadU32, 123456),
        (RegisterS16, PayloadS16, -42),
        (RegisterFloat, PayloadFloat, 1.5),
    ],
)
def test_format_with_payload_instance(reg_cls, payload_cls, value):
    # Passing a PayloadXxx instance to format() uses the bytes of the instance directly.
    reg = reg_cls(0x08)
    payload = payload_cls(value)
    frame = reg.format(payload)
    msg = _parse_frame(frame)
    assert msg.message_type == MessageType.Write
    assert msg.payload_bytes == payload.payload_array.tobytes()


def test_format_accepts_sequence_for_array_register():
    # The payload dtype is a sub-array, so converting a sequence against it directly
    # broadcasts each element into the full shape and doubles the payload.
    reg = RegisterU16Array(0x20, length=2)
    expected = np.array([1, 2], dtype=np.uint16).tobytes()
    for value in ([1, 2], (1, 2), np.array([1, 2], dtype=np.uint16)):
        msg = _parse_frame(reg.format(value))
        assert msg.payload_bytes == expected
        assert list(reg.parse(msg)) == [1, 2]


def test_format_rejects_wrong_length_sequence():
    reg = RegisterU16Array(0x20, length=2)
    with pytest.raises(ValueError, match="expects 2 elements but got 3"):
        reg.format([1, 2, 3])


def test_parse_names_register_on_short_payload():
    # A read request carries no payload, and numpy would otherwise report only
    # "buffer is smaller than requested size", naming neither side.
    reg = RegisterU16(0x20)
    request = _parse_frame(reg.format(message_type=MessageType.Read))
    with pytest.raises(HarpParseError, match="reads 2 payload bytes"):
        reg.parse(request)


def test_format_with_payload_instance_via_register():
    # format() accepts a typed PayloadU32 and encodes it correctly.
    payload = PayloadU32(42)
    frame = TimestampSecond.format(payload)
    msg = _parse_frame(frame)
    parsed = TimestampSecond.parse(msg)
    assert parsed == 42


def test_empty_buffer_keeps_columns():
    # A sub-array renders one column per element, and the count comes from the dtype,
    # so a buffer carrying no frames still renders all of them.
    reg = RegisterU32Array(0x28, length=3)
    records = np.arange(6, dtype=np.uint32).reshape(2, 3)
    populated = parse_to_dataframe(reg, bytes(reg.format_bulk(records)), time_index=False)
    empty = parse_to_dataframe(reg, b"", time_index=False)
    assert list(empty.columns) == list(populated.columns)
    assert empty.dtypes.equals(populated.dtypes)
    assert len(empty) == 0


def test_structured_register_format_single_sample():
    sample = np.array([(100, 512, -200)], dtype=AnalogDataPayload.payload_dtype)
    frame = AnalogData.format(sample)
    msg = _parse_frame(frame)
    parsed = AnalogData.parse(msg)
    assert isinstance(parsed, AnalogDataPayload)
    # parse() yields a 0-D record; @property accessors return numpy scalars.
    assert int(parsed.analog_input0) == 100
    assert int(parsed.encoder) == 512
    assert int(parsed.analog_input1) == -200


def test_structured_register_to_dataframe():
    raw = np.array(
        [(1, 2, 3), (4, 5, 6)],
        dtype=AnalogDataPayload.payload_dtype,
    ).tobytes()
    # Bulk decode goes through ._PayloadBatchType; from_buffer handles the redirect.
    bulk = AnalogDataPayload.payload_from_buffer(raw)
    df = payload_to_dataframe(bulk)
    assert list(df.columns) == ["analog_input0", "encoder", "analog_input1"]
    assert len(df) == 2
    assert df["analog_input0"].tolist() == [1, 4]


def test_array_register_factory_creates_subclass():
    reg = RegisterU32Array(0x08, length=3)
    assert issubclass(reg, RegisterU32Array)
    assert reg.address == 0x08
    assert reg.length == 3


def test_array_register_factory_different_lengths_independent():
    # Just to make sure there are no weird interactions when using the metaclass-base factory
    r1 = RegisterU32Array(0x08, length=2)
    r2 = RegisterU32Array(0x08, length=4)
    assert r1.length != r2.length
    assert r1.payload_class is not r2.payload_class


def test_array_register_format_write():
    reg = RegisterU32Array(0x08, length=3)
    values = np.array([10, 20, 30], dtype=np.dtype("<u4"))
    frame = reg.format(values)
    msg = _parse_frame(frame)
    assert msg.message_type == MessageType.Write
    assert msg.payload_bytes == values.tobytes()


def test_array_register_parse_roundtrip():
    reg = RegisterU32Array(0x08, length=3)
    values = np.array([10, 20, 30], dtype=np.dtype("<u4"))
    frame = reg.format(values)
    msg = _parse_frame(frame)
    parsed = reg.parse(msg)
    # parse() yields a 1-D length-N ndarray directly.
    np.testing.assert_array_equal(parsed, values)
    assert parsed.shape == (3,)


def test_s16_array_roundtrip():
    reg = RegisterS16Array(0x20, length=4)
    values = np.array([-1, 0, 1, 32767], dtype=np.dtype("<i2"))
    frame = reg.format(values)
    msg = _parse_frame(frame)
    parsed = reg.parse(msg)
    np.testing.assert_array_equal(parsed, values)
    assert parsed.shape == (4,)


def test_unnamed_register_auto_payload_class():
    # A bare RegisterU8 subclass with only address set gets an auto-generated payload class.

    class MyReg(RegisterU8):
        address: ClassVar[int] = 0x50

    # payload_class should exist and parse correctly
    raw = np.array([7], dtype=np.dtype("u1")).tobytes()
    parsed = MyReg.parse(raw)
    assert parsed == 7


def test_explicit_payload_class_not_overwritten():
    # Explicit payload_class on AnalogData is not replaced by auto-generation.
    assert AnalogData.payload_class is AnalogDataPayload


def test_format_read_override_message_type():
    frame = TimestampSecond.format(message_type=MessageType.Write)
    msg = _parse_frame(frame)
    assert msg.message_type == MessageType.Write
    assert msg.payload_bytes == b""


def test_format_write_override_message_type():
    frame = TimestampSecond.format(42, message_type=MessageType.Event)
    msg = _parse_frame(frame)
    assert msg.message_type == MessageType.Event


def test_format_read_with_timestamp():
    ts = 12.5
    frame = TimestampSecond.format(timestamp=ts)
    msg = _parse_frame(frame)
    assert msg.message_type == MessageType.Read
    assert msg.has_timestamp
    assert msg.timestamp == pytest.approx(ts, abs=1e-4)


def test_format_write_with_timestamp():
    ts = 100.0
    frame = TimestampSecond.format(42, timestamp=ts)
    msg = _parse_frame(frame)
    assert msg.message_type == MessageType.Write
    assert msg.has_timestamp
    assert msg.timestamp == pytest.approx(ts, abs=1e-4)
    parsed = TimestampSecond.parse(msg)
    assert parsed == 42


# ---------------------------------------------------------------------------
# 9. AnonymousPayload / register parse return contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload_cls, raw_value, np_dtype",
    [
        (PayloadU8, 255, np.dtype("u1")),
        (PayloadU16, 1000, np.dtype("<u2")),
        (PayloadU32, 99999, np.dtype("<u4")),
        (PayloadU64, 2**32, np.dtype("<u8")),
        (PayloadS8, -5, np.dtype("i1")),
        (PayloadS16, -300, np.dtype("<i2")),
        (PayloadS32, -100000, np.dtype("<i4")),
        (PayloadS64, -(2**33), np.dtype("<i8")),
        (PayloadFloat, 1.5, np.dtype("<f4")),
    ],
)
def test_anonymous_payload_roundtrip(payload_cls, raw_value, np_dtype):
    # Anonymous payload constructor + payload_bytes roundtrips through bytes.
    payload = payload_cls(raw_value)
    assert payload.payload_array.dtype == np_dtype
    assert payload.payload_array.tobytes() == np.asarray(raw_value, dtype=np_dtype).tobytes()


def test_structured_payload_descriptors_single():
    buf = np.array([(100, 512, -200)], dtype=AnalogDataPayload.payload_dtype).tobytes()
    parsed = AnalogDataPayload.payload_from_buffer(buf)
    # 1-D batch (frombuffer always returns at least 1-D); descriptors return ndarrays.
    np.testing.assert_array_equal(parsed.analog_input0, [100])
    np.testing.assert_array_equal(parsed.encoder, [512])
    np.testing.assert_array_equal(parsed.analog_input1, [-200])


def test_structured_payload_descriptors_multi():
    records = [(100, 512, -200), (110, 513, -210), (120, 514, -220)]
    buf = np.array(records, dtype=AnalogDataPayload.payload_dtype).tobytes()
    parsed = AnalogDataPayload.payload_from_buffer(buf)
    assert len(parsed) == 3
    np.testing.assert_array_equal(parsed.analog_input0, [100, 110, 120])
    np.testing.assert_array_equal(parsed.encoder, [512, 513, 514])
    np.testing.assert_array_equal(parsed.analog_input1, [-200, -210, -220])


def test_anonymous_payload_converter_roundtrip():
    # A __value__ Field codec encodes and decodes the single slot, modeling a register
    # that carries one value but needs a domain codec, such as DeviceName through
    # StringConverter.
    from harp.protocol._payload import AnonymousPayload, Field
    from harp.protocol._payload_converters import StringConverter

    class PayloadDeviceName(AnonymousPayload[np.uint8]):
        __value__: str = Field(StringConverter(25))

    class DeviceName(RegisterBase):
        address: ClassVar[int] = 12
        payload_type: ClassVar[PayloadType] = PayloadType.U8
        payload_class = PayloadDeviceName

    # dtype derives from the converter (one structured slot); raw bytes are the
    # encoded, null-padded value.
    assert PayloadDeviceName.payload_dtype.names == ("__value__",)
    assert PayloadDeviceName.payload_dtype.itemsize == 25
    payload = PayloadDeviceName("Behavior")
    assert payload.payload_array.tobytes() == b"Behavior".ljust(25, b"\x00")

    # Register round-trip decodes back to the high-level str, whether format()
    # is given a payload instance or the bare value (symmetric with parse()).
    for arg in (payload, "Behavior"):
        parsed = DeviceName.parse(_parse_frame(DeviceName.format(arg)))
        assert parsed == "Behavior"

    # to_dataframe decodes both a single record and a batch.
    assert payload_to_dataframe(PayloadDeviceName("Behavior"))["value"].tolist() == ["Behavior"]
    two = (
        PayloadDeviceName("Foo").payload_array.tobytes()
        + PayloadDeviceName("Bar").payload_array.tobytes()
    )
    batch = PayloadDeviceName.payload_from_buffer(two)
    assert payload_to_dataframe(batch)["value"].tolist() == ["Foo", "Bar"]


def test_anonymous_payload_scalar_converter_roundtrip():
    # A scalar (non-sub-array) __value__ codec also round-trips through unwrap.
    import enum

    from harp.protocol._payload import AnonymousPayload, Field
    from harp.protocol._payload_converters import EnumConverter

    class Color(enum.IntEnum):
        RED = 0
        GREEN = 1
        BLUE = 2

    class PayloadColor(AnonymousPayload[np.uint8]):
        __value__: Color = Field(EnumConverter(Color))

    assert PayloadColor.payload_dtype.itemsize == 1
    raw = PayloadColor(Color.BLUE).payload_array.tobytes()
    record = np.frombuffer(raw, dtype=PayloadColor.payload_dtype, count=1)[0]
    assert PayloadColor._unwrap(record) == Color.BLUE


def test_array_register_parse_returns_ndarray():
    # parse() on an array register returns the 1-D ndarray directly (no .value).
    reg = RegisterU32Array(0x08, length=3)
    values = np.array([10, 20, 30], dtype=np.dtype("<u4"))
    frame = reg.format(values)
    msg = _parse_frame(frame)
    parsed = reg.parse(msg)
    assert isinstance(parsed, np.ndarray)
    assert parsed.shape == (3,)
    np.testing.assert_array_equal(parsed, values)


# ---------------------------------------------------------------------------
# 10. parse vs read_frames / ._PayloadBatchType contract
# ---------------------------------------------------------------------------


def test_parse_returns_numpy_scalar():
    # parse() on a scalar register returns a 0-D numpy scalar of the correct dtype.
    frame = TimestampSecond.format(42)
    msg = _parse_frame(frame)
    parsed = TimestampSecond.parse(msg)
    assert isinstance(parsed, np.uint32)
    assert parsed.ndim == 0
    assert parsed == 42


def test_parse_does_not_overrun_buffer():
    # parse() reads exactly one record even if the buffer is larger.
    # Two-record buffer in raw form, with no Harp header, exercising the raw-bytes path.
    raw = np.array([42, 99], dtype=np.dtype("<u4")).tobytes()
    parsed = TimestampSecond.parse(raw)
    assert parsed == 42
    assert parsed.ndim == 0


def test_batch_payload_routes_to_batch_twin():
    # from_buffer wraps a multi-element buffer in the auto-derived Batch twin. For array
    # registers the payload dtype is a sub-array dtype, so a buffer holding N rows of
    # length L decodes to an ndarray of shape N by L.
    reg = RegisterU32Array(0x08, length=3)
    rows = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.dtype("<u4"))
    batch = reg.payload_class.payload_from_buffer(rows.tobytes())
    assert type(batch) is reg.payload_class._PayloadBatchType
    assert isinstance(batch, reg.payload_class)
    assert batch._arr.shape == (2, 3)


def test_struct_payload_field_descriptors_codegen_style():
    # A struct payload declared via _Field descriptors decodes both ndim modes.

    class GeneratedAnalogPayload(PayloadBase):
        a = Field(converter=_IdentityConverter("<i2"), offset=0)
        b = Field(converter=_IdentityConverter("<i2"), offset=2)
        c = Field(converter=_IdentityConverter("<i2"), offset=4)

    p = GeneratedAnalogPayload._from_array(
        np.array((1, 2, 3), dtype=GeneratedAnalogPayload.payload_dtype)
    )
    # 0-D _arr gives a numpy scalar per field.
    assert int(p.a) == 1
    assert int(p.b) == 2
    assert int(p.c) == 3

    # 1-D _arr gives ndarray columns. The same descriptor handles both.
    batch_arr = np.array([(1, 2, 3), (4, 5, 6)], dtype=GeneratedAnalogPayload.payload_dtype)
    batch = GeneratedAnalogPayload.payload_from_buffer(batch_arr.tobytes())
    np.testing.assert_array_equal(batch.a, [1, 4])
    np.testing.assert_array_equal(batch.b, [2, 5])
    assert batch._arr.ndim == 1


def test_repr_fields_auto_derived_from_dtype():
    # A struct payload that doesn't set _repr_fields gets them from _dtype.names.

    class P(PayloadBase):
        alpha = Field(converter=_IdentityConverter("<i2"), offset=0)
        beta = Field(converter=_IdentityConverter("<u1"), offset=2)

    assert P._repr_fields == ("alpha", "beta")


def test_repr_fields_auto_derived_mixed_bitfield_and_field():
    # A payload mixing a masked sub-field with plain Fields keeps all of them in
    # declaration order, so the masked slot does not shadow the plain fields.

    class P(PayloadBase):
        flags = Field(converter=_IdentityConverter("u1"), mask=0xFF, offset=0)
        scale = Field(converter=_IdentityConverter("<f4"), offset=4)
        count = Field(converter=_IdentityConverter("<u4"), offset=8)

    assert P._repr_fields == ("flags", "scale", "count")


def test_repr_fields_auto_derived_shared_slot_bitfields():
    # Several masked sub-fields packed into one dtype slot each appear in _repr_fields.

    class P(PayloadBase):
        low = Field(converter=_IdentityConverter("u1"), mask=0x0F, offset=0)
        high = Field(converter=_IdentityConverter("u1"), mask=0xF0, offset=0)

    assert P._repr_fields == ("low", "high")


# ---------------------------------------------------------------------------
# format_bulk (inverse of parse_bulk) + harp.data.to_buffer / to_file
# ---------------------------------------------------------------------------


def test_format_bulk_single_matches_format():
    reg = RegisterU16(0x20)
    one = reg.format(np.uint16(42), message_type=MessageType.Event, timestamp=1.0)
    bulk = reg.format_bulk(
        np.array([42], dtype="<u2"), timestamps=[1.0], message_type=MessageType.Event
    )
    assert bytes(bulk) == one


def test_format_bulk_parse_bulk_roundtrip():
    reg = RegisterU16(0x20)
    values = np.array([1, 2, 3], dtype="<u2")
    buf = reg.format_bulk(values, timestamps=[1.0, 2.0, 3.0])
    df = parse_to_dataframe(reg, bytes(buf), time_index=False)
    assert df["value"].tolist() == [1, 2, 3]


def test_format_bulk_is_exact_inverse_of_parse_bulk():
    reg = RegisterS16Array(0x2C, length=3)
    original = reg.format_bulk(np.array([[1, 0, 2], [3, 4, 5]], dtype="<i2"), timestamps=[1.0, 2.0])
    _data, ts, msg, payload = reg.parse_bulk(bytes(original))
    rebuilt = reg.format_bulk(payload, timestamps=np.asarray(ts), message_type=np.asarray(msg))
    assert bytes(rebuilt) == bytes(original)


def test_format_bulk_accepts_sequences_for_array_register():
    # An array register formats a sequence element-wise, as format does, so every
    # spelling of the same values produces the same frames.
    reg = RegisterU16Array(0x20, length=2)
    records = np.array([(1, 2), (3, 4)], dtype=reg.payload_class.payload_dtype)
    expected = bytes(reg.format_bulk(records))
    for values in (
        np.array([[1, 2], [3, 4]], dtype=np.uint16),
        np.array([[1, 2], [3, 4]]),
        [[1, 2], [3, 4]],
        ((1, 2), (3, 4)),
    ):
        assert bytes(reg.format_bulk(values)) == expected


def test_format_bulk_single_frame_matches_format_for_array_register():
    reg = RegisterU16Array(0x20, length=2)
    bulk = reg.format_bulk([1, 2], message_type=MessageType.Write)
    assert bytes(bulk) == reg.format([1, 2], message_type=MessageType.Write)


def test_format_bulk_rejects_shape_without_declared_payload():
    # Four elements for a two-element register could be one frame or two, so the
    # ambiguous flat form raises rather than guessing.
    reg = RegisterU16Array(0x20, length=2)
    with pytest.raises(ValueError, match="do not end in the declared payload shape"):
        reg.format_bulk(np.array([1, 2, 3, 4], dtype=np.uint16))
    with pytest.raises(ValueError, match="do not end in the declared payload shape"):
        reg.format_bulk([[1, 2, 3], [4, 5, 6]])


def test_parse_bulk_names_register_on_partial_frame():
    # numpy would otherwise report an out-of-bounds index against the strided view,
    # naming neither the register nor how many bytes a frame needs.
    reg = RegisterU16Array(0x20, length=2)
    buf = bytes(reg.format_bulk([[1, 2], [3, 4]]))
    stride = buf[1] + 2
    with pytest.raises(HarpParseError, match="at least 6 bytes"):
        reg.parse_bulk(buf[:5])
    with pytest.raises(HarpParseError, match=f"frames of {stride} bytes"):
        reg.parse_bulk(buf[: stride - 1])


def test_parse_bulk_empty_buffer_does_not_raise():
    reg = RegisterU16Array(0x20, length=2)
    _data, timestamps, msgtype, payload = reg.parse_bulk(b"")
    assert len(np.asarray(payload.payload_array)) == 0
    assert timestamps is None and msgtype is None


def test_to_buffer_and_to_file_roundtrip(tmp_path):
    reg = RegisterU16(0x20)
    values = np.array([10, 20], dtype="<u2")
    buf = to_buffer(reg, values, timestamps=[1.0, 2.0])
    assert parse_to_dataframe(reg, bytes(buf))["value"].tolist() == [10, 20]

    path = tmp_path / "reg.bin"
    to_file(reg, values, path, timestamps=[1.0, 2.0])
    assert parse_to_dataframe(reg, path)["value"].tolist() == [10, 20]
