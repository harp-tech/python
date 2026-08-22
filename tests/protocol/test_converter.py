"""Tests for the unified _Field + _Converter machinery in _payload.py.

Covers:
* IdentityConverter via auto-generated _Field (parity with previous _Field).
* StringConverter, sub-array uint8 to str and back.
* EnumConverter (full-byte enum decoding).
* Declarations-build-dtype direction (no _dtype on the subclass).
* Reserved-name collision check.
"""

import enum

import numpy as np
import pytest
from harp.data import payload_to_dataframe
from harp.protocol._payload import (
    PayloadBase,
    BitMask,
    Field,
    GroupMask,
    _IdentityConverter,
)
from harp.protocol._payload_converters import StringConverter as _StringConverter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Color(enum.IntEnum):
    Red = 0
    Green = 1
    Blue = 2


class _Flag(enum.IntFlag):
    A = 0x01


# ---------------------------------------------------------------------------
# IdentityConverter, pass-through field
# ---------------------------------------------------------------------------


class _NumericPayload(PayloadBase):
    a = Field(converter=_IdentityConverter("<i2"), offset=0)
    b = Field(converter=_IdentityConverter("<u4"), offset=2)


def test_identity_converter_scalar_view():
    rec = np.array((-7, 99), dtype=_NumericPayload.payload_dtype)
    p = _NumericPayload._from_array(rec)
    assert int(p.a) == -7
    assert int(p.b) == 99


def test_identity_converter_batch_view():
    arr = np.array([(-7, 99), (1, 2)], dtype=_NumericPayload.payload_dtype)
    p = _NumericPayload.payload_from_buffer(arr.tobytes())
    np.testing.assert_array_equal(p.a, [-7, 1])
    np.testing.assert_array_equal(p.b, [99, 2])


# ---------------------------------------------------------------------------
# Declarations-build-dtype direction
# ---------------------------------------------------------------------------


class DeclaredPayload(PayloadBase):
    delta = Field(converter=_IdentityConverter(np.uint32), offset=0)
    flag = Field(converter=_IdentityConverter(np.uint8), offset=4)


def test_declared_dtype_synthesised_from_fields():
    # Field declarations alone build a structured dtype in declaration order.
    assert DeclaredPayload.payload_dtype.names == ("delta", "flag")
    assert DeclaredPayload.payload_dtype.fields["delta"][0] == np.dtype(np.uint32)
    assert DeclaredPayload.payload_dtype.fields["flag"][0] == np.dtype(np.uint8)


def test_declared_dtype_kwarg_init_round_trip():
    p = DeclaredPayload(delta=42, flag=1)
    assert int(p.delta) == 42
    assert int(p.flag) == 1
    p2 = DeclaredPayload.payload_from_buffer(p.payload_array.tobytes())
    assert int(p2.delta) == 42
    assert int(p2.flag) == 1


# ---------------------------------------------------------------------------
# StringConverter
# ---------------------------------------------------------------------------


class _NamedPayload(PayloadBase):
    name = Field(converter=_StringConverter(8), offset=0)
    delta = Field(converter=_IdentityConverter(np.uint16), offset=8)


def test_string_converter_dtype_synthesis():
    # name should occupy 8 bytes, delta 2 bytes.
    assert _NamedPayload.payload_dtype.itemsize == 10
    assert _NamedPayload.payload_dtype.fields["name"][0].subdtype is not None


def test_string_converter_scalar_decode_roundtrip():
    p = _NamedPayload(name="abc", delta=7)
    assert p.name == "abc"
    assert int(p.delta) == 7
    # raw bytes are zero-padded
    raw = bytes(p.payload_array["name"])
    assert raw == b"abc\x00\x00\x00\x00\x00"


def test_string_converter_batch_decode():
    rec1 = _NamedPayload(name="hi", delta=1).payload_array.tobytes()
    rec2 = _NamedPayload(name="bye", delta=2).payload_array.tobytes()
    batch = _NamedPayload.payload_from_buffer(rec1 + rec2)
    names = batch.name
    assert list(names) == ["hi", "bye"]
    np.testing.assert_array_equal(batch.delta, [1, 2])


def test_string_converter_to_dataframe():
    rec1 = _NamedPayload(name="hi", delta=1).payload_array.tobytes()
    rec2 = _NamedPayload(name="bye", delta=2).payload_array.tobytes()
    batch = _NamedPayload.payload_from_buffer(rec1 + rec2)
    df = payload_to_dataframe(batch)
    # Non-identity converter produces one column per field, with no sub-array
    # expansion for the string field.
    assert list(df.columns) == ["name", "delta"]
    assert df["name"].tolist() == ["hi", "bye"]
    assert df["delta"].tolist() == [1, 2]


# ---------------------------------------------------------------------------
# GroupMask in struct-field role (full-byte enum, non-default slot).
# Demonstrates that _GroupMask subsumes the previous _EnumConverter.
# ---------------------------------------------------------------------------


class _ConfigPayload(PayloadBase):
    # Full-byte enum at offset 0; delta is the next element. Distinct slots each
    # declare an explicit offset.
    color = GroupMask(mask=0xFF, enum=_Color, offset=0)
    delta = Field(converter=_IdentityConverter(np.uint8), offset=1)


def test_groupmask_struct_field_scalar_decode():
    p = _ConfigPayload(color=_Color.Green, delta=3)
    assert p.color is _Color.Green
    assert isinstance(p.color, _Color)
    assert int(p.delta) == 3


def test_groupmask_struct_field_batch_decode_returns_ints():
    p1 = _ConfigPayload(color=_Color.Red, delta=1).payload_array.tobytes()
    p2 = _ConfigPayload(color=_Color.Blue, delta=2).payload_array.tobytes()
    batch = _ConfigPayload.payload_from_buffer(p1 + p2)
    np.testing.assert_array_equal(batch.color, [int(_Color.Red), int(_Color.Blue)])


# ---------------------------------------------------------------------------
# Reserved-name collision
# ---------------------------------------------------------------------------


def test_reserved_field_name_raises():
    with pytest.raises(TypeError, match="reserved"):

        class _Bad(PayloadBase):
            _dtype = Field(converter=_IdentityConverter(np.uint8))  # type: ignore[assignment]


def test_value_field_name_allowed():
    # ``value`` is intentionally overridable, since the descriptor wins via MRO.
    class _Single(PayloadBase):
        value = Field(converter=_StringConverter(4))

    p = _Single(value="ok")
    assert p.value == "ok"


# ---------------------------------------------------------------------------
# Sanity: bitfield path is unaffected by the converter refactor
# ---------------------------------------------------------------------------


def test_bitfield_payloads_ndim_aware():
    # Scalar records stay on the declared class; batches route to the auto-derived Batch twin.

    class _Flags(PayloadBase):
        flag = BitMask(enum=_Flag, mask=0x01)
        group = GroupMask(mask=0x06, enum=_Color)

    # 0-D scalar record: flag=1, group bits=01 (Green)
    scalar = _Flags._from_array(np.array((0x03,), dtype=_Flags.payload_dtype))
    assert type(scalar) is _Flags
    assert scalar.flag is _Flag.A
    assert scalar.group is _Color.Green

    # 1-D batch, the Batch sibling with ndarray-typed accessors.
    batch = _Flags.payload_from_buffer(bytes([0x01, 0x02]))
    assert type(batch) is _Flags._PayloadBatchType
    assert isinstance(batch, _Flags)
    np.testing.assert_array_equal(batch.flag, [1, 0])
    np.testing.assert_array_equal(batch.group, [0, 1])


def test_bitfield_kwarg_init_round_trip():
    # PayloadBase.__init__ supports bitfield kwargs with OR-into-slot encoding.

    class _Flags(PayloadBase):
        flag = BitMask(enum=_Flag, mask=0x01)
        group = GroupMask(mask=0x06, enum=_Color)

    p = _Flags(flag=_Flag.A, group=_Color.Green)
    assert p.flag is _Flag.A
    assert p.group is _Color.Green
    # Wire byte: flag bit + (Green << 1) = 0x01 | 0x02 = 0x03. Masked fields on one
    # element share a slot named after the first declared field ("flag").
    assert int(p.payload_array["flag"]) == 0x03
