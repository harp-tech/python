import zlib

import numpy as np
import pytest
from harp.data import parse_to_dataframe
from harp.protocol import GroupMask, HarpMessage

from harp.device import core
from harp.device.schema._emit import (
    NameCollisionError,
    UnknownConverterError,
    UnknownMaskError,
    create_registers,
)

from . import expected_device
from .converters import DataConverter

CONVERTERS = {"DataConverter": DataConverter()}


@pytest.fixture
def device_registers(device_yml):
    return create_registers(device_yml, converters=CONVERTERS)


def _device_registers():
    # expected_device.REGISTER_MAP spreads the core map; the device-specific
    # registers (the ones the emitter builds from device.yml) are address >= 32.
    return {cls.__name__: cls for addr, cls in expected_device.REGISTER_MAP.items() if addr >= 32}


def _layout(dt):
    """Full structural signature: field name + element dtype + offset, and itemsize.

    Name-exact, since the emitter applies the same naming convention as the generator,
    so the reference comparison covers identifiers as well as byte layout.
    """
    if dt.names is None:
        return ("scalar", dt.str, dt.shape, dt.itemsize)
    return ("struct", dt.itemsize, tuple((n, dt.fields[n][0], dt.fields[n][1]) for n in dt.names))


# ---------------------------------------------------------------------------
# Device reference output, layout and type parity with generator output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_device_registers()))
def test_device_register_matches_generator_layout(name, device_registers):
    emitted = device_registers[name]
    expected = _device_registers()[name]
    assert emitted.address == expected.address
    assert emitted.payload_type == expected.payload_type
    assert _layout(emitted.payload_class.payload_dtype) == _layout(
        expected.payload_class.payload_dtype
    )


def test_device_emits_all_registers(device_registers):
    assert set(device_registers) == set(_device_registers())


# ---------------------------------------------------------------------------
# Naming, identical to the statically generated device package
# ---------------------------------------------------------------------------


def _enum_of(reg, field):
    return reg.payload_class._mro_descriptor(field)._enum


def test_field_names_are_snake_case(device_registers):
    fields = device_registers["AnalogData"].payload_class.payload_dtype.names
    assert fields == ("analog0", "analog1", "analog2", "accelerometer")


def test_field_names_match_generator(device_registers):
    # A run of capitals stays one word; a trailing digit never separates.
    fields = device_registers["Version"].payload_class.payload_dtype.names
    assert fields == (
        "protocol_version",
        "firmware_version",
        "hardware_version",
        "core_id",
        "interface_hash",
    )


def test_enum_members_are_screaming_snake_case(device_registers):
    flags = _enum_of(device_registers["PortDIOSet"], "__value__")
    assert list(flags.__members__) == [
        "DIO0",
        "DIO1",
        "DIO2",
        "DIO3",
        "DI_PORT0",
        "TEST_DI_PORT1",
        "SUPPLY_PORT0",
        "PORT_DIO1",
    ]


def test_group_mask_members_match_generator(device_registers):
    assert list(_enum_of(device_registers["StartPulse"], "digital_output").__members__) == [
        "PWM0",
        "PWM1",
        "PWM2",
        "PWM3",
    ]


def test_register_and_payload_class_names_stay_verbatim(device_registers):
    # Only fields and enum members are renamed; type-level names come from the yml.
    reg = device_registers["AnalogData"]
    assert reg.__name__ == "AnalogData"
    assert reg.payload_class.__name__ == "AnalogDataPayload"
    assert _enum_of(device_registers["EncoderMode"], "__value__").__name__ == "EncoderModeMask"


def test_enum_names_match_generator_for_every_enum(device_registers):
    # Every enum the reference module declares has identical members in the emitter.
    for name, reg in _device_registers().items():
        payload = reg.payload_class
        if payload.payload_dtype.names is None:
            continue
        for field in payload._repr_fields:
            expected_desc = payload._mro_descriptor(field)
            expected_enum = getattr(expected_desc, "_enum", None)
            if expected_enum is None:
                continue
            emitted_enum = _enum_of(device_registers[name], field)
            assert emitted_enum.__name__ == expected_enum.__name__
            assert {m.name: int(m.value) for m in emitted_enum} == {
                m.name: int(m.value) for m in expected_enum
            }


# ---------------------------------------------------------------------------
# The declared length decides whether a register is an array
# ---------------------------------------------------------------------------

_LENGTH_YML = """
registers:
  Absent: {address: 32, type: U16, access: Read}
  One: {address: 33, type: U16, length: 1, access: Read}
  Three: {address: 34, type: U16, length: 3, access: Read}
"""


def test_absent_length_emits_scalar_register():
    # A register with no declared length holds a single value, and carries no length at all.
    reg = create_registers(_LENGTH_YML)["Absent"]
    assert not hasattr(reg, "length")
    assert reg.payload_class.payload_dtype.shape == ()


@pytest.mark.parametrize("name, count", [("One", 1), ("Three", 3)])
def test_declared_length_emits_array_register(name, count):
    # Any declared length means an array, 1 included. Reading a declared 1 as a single value
    # gives the same type as declaring nothing, and the generator would then disagree.
    reg = create_registers(_LENGTH_YML)[name]
    assert reg.length == count
    assert reg.payload_class.payload_dtype.shape == (count,)
    values = np.arange(count, dtype=np.uint16)
    np.testing.assert_array_equal(reg.parse(HarpMessage.parse(reg.format(values))), values)


def test_multi_element_struct_register_decodes_through_message():
    # A struct register declares no length, so decode has to size its payload from the
    # payload class.
    reg = create_registers(
        "registers:\n"
        "  Settings:\n"
        "    address: 32\n"
        "    type: U16\n"
        "    length: 3\n"
        "    access: Write\n"
        "    payloadSpec:\n"
        "      Gain: {offset: 0}\n"
        "      Offset: {offset: 1}\n"
        "      Threshold: {offset: 2}\n"
    )["Settings"]
    payload = reg.payload_class(gain=1, offset=2, threshold=3)
    decoded = HarpMessage.parse(reg.format(payload)).decode(reg)
    assert (int(decoded.payload.gain), int(decoded.payload.threshold)) == (1, 3)


def test_core_string_register_decodes_through_message():
    # DeviceName spans 25 U8 elements, so the same defect reached every Harp device.
    frame = core.DeviceName.format("Behavior")
    assert HarpMessage.parse(frame).decode(core.DeviceName).payload == "Behavior"


# ---------------------------------------------------------------------------
# Core registers: the emitter and the generated package, from the same core.yml
# ---------------------------------------------------------------------------


def _core_expected():
    # harp.device.core is the generated Python interface, so it is the reference for
    # what the emitter should build from the same core registers.
    return {cls.__name__: cls for cls in core.REGISTER_MAP.values()}


@pytest.mark.parametrize("name", sorted(_core_expected()))
def test_core_register_structural(name, core_yml):
    emitted = create_registers(core_yml)[name]
    expected = _core_expected()[name]
    assert emitted.address == expected.address
    assert emitted.payload_type == expected.payload_type
    assert (
        emitted.payload_class.payload_dtype.itemsize
        == expected.payload_class.payload_dtype.itemsize
    )
    assert _layout(emitted.payload_class.payload_dtype) == _layout(
        expected.payload_class.payload_dtype
    )


# ---------------------------------------------------------------------------
# Behavioral round-trips
# ---------------------------------------------------------------------------


def _roundtrip(reg, value):
    return reg.parse(HarpMessage.parse(reg.format(value)))


def test_whole_register_groupmask_unwraps_to_enum(device_registers):
    reg = device_registers["EncoderMode"]
    enum_cls = reg.payload_class._mro_descriptor("__value__")._enum
    parsed = _roundtrip(reg, enum_cls["DISPLACEMENT"])
    assert parsed == enum_cls["DISPLACEMENT"]
    assert isinstance(parsed, enum_cls)


def test_whole_register_bitmask_roundtrip(device_registers):
    reg = device_registers["PortDIOSet"]
    flags = reg.payload_class._mro_descriptor("__value__")._enum
    value = flags["DIO0"] | flags["DIO3"]
    assert _roundtrip(reg, value) == value


def test_struct_masked_members_roundtrip(device_registers):
    reg = device_registers["StartPulse"]
    payload_cls = reg.payload_class
    pwm = payload_cls._mro_descriptor("digital_output")._enum
    # digital_output is a 2-bit field (mask 0xC00); only PWM0/PWM1 fit it. This
    # matches the generator output verbatim, as GroupMask(enum=PwmPort, mask=0xC00).
    payload = payload_cls(digital_output=pwm["PWM1"], pulse_width=np.uint16(300))
    parsed = _roundtrip(reg, payload)
    assert parsed.digital_output == pwm["PWM1"]
    assert int(parsed.pulse_width) == 300


def test_custom_converter_roundtrip(device_registers):
    reg = device_registers["CustomMemberConverter"]
    payload_cls = reg.payload_class
    parsed = _roundtrip(reg, payload_cls(header=np.uint8(7), data=-1234))
    assert int(parsed.header) == 7
    assert int(parsed.data) == -1234


# ---------------------------------------------------------------------------
# Core masks reused by a schema that does not declare them
# ---------------------------------------------------------------------------

CORE_MASKS_YML = (
    "device: CoreMasks\n"
    "registers:\n"
    "  EnableFlow: {address: 32, type: U8, access: Write, maskType: EnableFlag}\n"
    "  ResetFlow: {address: 33, type: U8, access: Write, maskType: ResetFlags}\n"
    "  FlowConfiguration:\n"
    "    address: 34\n"
    "    type: U8\n"
    "    access: Write\n"
    "    payloadSpec:\n"
    "      Indicators: {maskType: EnableFlag, mask: 0x1, defaultValue: 1}\n"
)


def _value_enum(reg):
    return reg.payload_class._mro_descriptor("__value__")._enum


def test_undeclared_group_mask_resolves_to_core():
    # Published devices reference EnableFlag without declaring it, so the emitter has
    # to reuse the core definition rather than fail to type the register.
    regs = create_registers(CORE_MASKS_YML)
    assert _value_enum(regs["EnableFlow"]) is core.EnableFlag


def test_undeclared_bit_mask_resolves_to_core():
    regs = create_registers(CORE_MASKS_YML)
    assert _value_enum(regs["ResetFlow"]) is core.ResetFlags


def test_reused_core_mask_roundtrips_as_core_type():
    # Identity matters more than equal members: a value read through a runtime module
    # must satisfy isinstance against the same enum a generated package would use.
    regs = create_registers(CORE_MASKS_YML)
    value = core.ResetFlags.SAVE | core.ResetFlags.RESTORE_NAME
    parsed = _roundtrip(regs["ResetFlow"], value)
    assert isinstance(parsed, core.ResetFlags)
    assert parsed == value


def test_reused_core_mask_resolves_on_payload_member():
    regs = create_registers(CORE_MASKS_YML)
    descriptor = regs["FlowConfiguration"].payload_class._mro_descriptor("indicators")
    assert isinstance(descriptor, GroupMask)
    assert descriptor._enum is core.EnableFlag
    assert descriptor._default is core.EnableFlag.ENABLED


def test_unresolvable_mask_type_is_rejected():
    # Naming the mask matters: the register does declare a maskType, so reporting that
    # one is missing would send the reader looking in the wrong place.
    with pytest.raises(UnknownMaskError, match="'EnableFlg' is neither declared"):
        create_registers(
            "registers:\n  R: {address: 32, type: U8, access: Read, maskType: EnableFlg}\n"
        )


def test_register_with_nothing_to_decode_is_rejected():
    with pytest.raises(ValueError, match="declares no payloadSpec, maskType, or interfaceType"):
        create_registers(
            "registers:\n  R: {address: 32, type: U8, access: Read, converter: Payload}\n"
        )


def test_declared_mask_shadows_core_definition():
    regs = create_registers(
        "registers:\n"
        "  Flow: {address: 32, type: U8, access: Read, maskType: EnableFlag}\n"
        "groupMasks:\n"
        "  EnableFlag:\n"
        "    values:\n"
        "      Closed: {value: 0}\n"
        "      Open: {value: 1}\n"
    )
    emitted = _value_enum(regs["Flow"])
    assert emitted is not core.EnableFlag
    assert list(emitted.__members__) == ["CLOSED", "OPEN"]


# ---------------------------------------------------------------------------
# Converter registry
# ---------------------------------------------------------------------------


def test_unknown_converter_raises(device_yml):
    with pytest.raises(UnknownConverterError):
        create_registers(device_yml)  # CustomMemberConverter needs DataConverter


def test_non_strict_falls_back_to_native(device_yml):
    regs = create_registers(device_yml, require_converters=False)
    # Data decodes as the raw native element (u8[2]) rather than the custom int.
    reg = regs["CustomMemberConverter"]
    assert reg.payload_class.payload_dtype.itemsize == 3


def test_converter_factory_receives_dsl_context(device_yml):
    seen = {}

    def factory(ctx):
        seen["name"], seen["span"], seen["interface_type"] = ctx.name, ctx.span, ctx.interface_type
        return DataConverter()

    regs = create_registers(device_yml, converters={"DataConverter": factory})
    parsed = _roundtrip(
        regs["CustomMemberConverter"],
        regs["CustomMemberConverter"].payload_class(header=np.uint8(1), data=42),
    )
    assert int(parsed.data) == 42
    # The factory was handed the resolved DSL context of the Data field, keyed by the
    # verbatim yml name, since the converter symbol derives from that, not from "data".
    assert seen == {"name": "Data", "span": 2, "interface_type": "int"}


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------


_VISIBILITY_YML = (
    "registers:\n"
    "  Pub: {address: 40, type: U16, access: Read}\n"
    "  Priv: {address: 41, type: U16, access: Read, visibility: private}\n"
)


def test_private_registers_are_emitted():
    # A private register stays in the address space, as it does in a generated package,
    # since the device can still send it.
    assert set(create_registers(_VISIBILITY_YML)) == {"Pub", "_Priv"}


def test_private_register_class_is_underscore_prefixed():
    regs = create_registers(_VISIBILITY_YML)
    assert regs["_Priv"].__name__ == "_Priv"
    assert regs["_Priv"].address == 41
    assert regs["Pub"].__name__ == "Pub"


def test_private_payload_class_is_not_prefixed():
    # Only the register class takes the underscore; its payload keeps {Name}Payload.
    regs = create_registers(
        "registers:\n"
        "  Priv:\n"
        "    address: 41\n"
        "    type: U8\n"
        "    access: Read\n"
        "    visibility: private\n"
        "    payloadSpec:\n"
        "      Foo: {offset: 0}\n"
    )
    assert regs["_Priv"].payload_class.__name__ == "PrivPayload"


# ---------------------------------------------------------------------------
# Payload class sharing, where a structured register with an interfaceType names its
# payload after that type, so registers sharing the type share one class.
# ---------------------------------------------------------------------------


def test_structured_register_payload_named_after_interface_type():
    regs = create_registers(
        "registers:\n"
        "  A:\n"
        "    address: 40\n"
        "    type: U8\n"
        "    access: Read\n"
        "    interfaceType: Shared\n"
        "    payloadSpec:\n"
        "      Foo: {offset: 0}\n"
        "  B:\n"
        "    address: 41\n"
        "    type: U8\n"
        "    access: Read\n"
        "    interfaceType: Shared\n"
        "    payloadSpec:\n"
        "      Foo: {offset: 0}\n"
    )
    assert regs["A"].payload_class.__name__ == "Shared"
    # One class, reused, not two structurally identical copies.
    assert regs["A"].payload_class is regs["B"].payload_class


def test_anchored_registers_share_one_payload_class():
    # How the published schemas actually reuse a payload: device.behavior anchors Rgb0
    # and merges it into Rgb1, so both carry the same interfaceType and payloadSpec by
    # construction. Reuse is keyed on the name alone, matching the generator, which
    # keeps one struct per interfaceType for the C# target too.
    regs = create_registers(
        "registers:\n"
        "  Rgb0: &rgbRegister\n"
        "    address: 71\n"
        "    type: U8\n"
        "    length: 3\n"
        "    access: Write\n"
        "    interfaceType: RgbPayload\n"
        "    payloadSpec:\n"
        "      Green: {offset: 0}\n"
        "      Red: {offset: 1}\n"
        "      Blue: {offset: 2}\n"
        "  Rgb1:\n"
        "    <<: *rgbRegister\n"
        "    address: 72\n"
    )
    shared = regs["Rgb0"].payload_class
    assert shared is regs["Rgb1"].payload_class
    assert shared.__name__ == "RgbPayload"
    assert shared.payload_dtype.names == ("green", "red", "blue")


def test_shared_payload_spanning_elements_without_length_is_reused():
    # A payloadSpec may span several elements without declaring a length, in which case
    # the payload takes its size from the member offsets. Sharing has to survive that,
    # since nothing in the schema requires the length to be spelled out.
    regs = create_registers(
        "registers:\n"
        "  A: &shared\n"
        "    address: 40\n"
        "    type: U16\n"
        "    access: Read\n"
        "    interfaceType: Combo\n"
        "    payloadSpec:\n"
        "      Alpha: {offset: 0}\n"
        "      Beta: {offset: 1}\n"
        "  B:\n"
        "    <<: *shared\n"
        "    address: 41\n"
    )
    assert regs["A"].payload_class is regs["B"].payload_class
    assert regs["A"].payload_class.payload_dtype.itemsize == 4


# ---------------------------------------------------------------------------
# Name collisions introduced by the convention
# ---------------------------------------------------------------------------


def test_colliding_field_names_raise():
    with pytest.raises(NameCollisionError, match="both map to 'foo'"):
        create_registers(
            "registers:\n"
            "  R:\n"
            "    address: 40\n"
            "    type: U8\n"
            "    access: Read\n"
            "    payloadSpec:\n"
            "      Foo: {offset: 0}\n"
            "      FOO: {offset: 1}\n"
        )


def test_colliding_enum_members_raise():
    with pytest.raises(NameCollisionError, match="both map to 'ON'"):
        create_registers(
            "registers:\n"
            "  R: {address: 40, type: U8, access: Read, maskType: M}\n"
            "groupMasks:\n"
            "  M:\n"
            "    values:\n"
            "      On: 0\n"
            "      ON: 1\n"
        )


def _one_field_schema(key: str) -> str:
    return (
        "registers:\n"
        "  R:\n"
        "    address: 40\n"
        "    type: U8\n"
        "    access: Read\n"
        "    payloadSpec:\n"
        f"      {key}: {{offset: 0}}\n"
        "      Other: {offset: 1}\n"
    )


@pytest.mark.parametrize("key", ["PayloadDtype", "PayloadColumns", "PayloadAnything"])
def test_field_name_taking_reserved_prefix_raises(key):
    # Payload members all carry the payload_ prefix, so a field is barred from the
    # prefix rather than from a list of the members that happen to exist today.
    with pytest.raises(NameCollisionError, match="reserved for payload members"):
        create_registers(_one_field_schema(key))


@pytest.mark.parametrize("key", ["Break", "Class", "Return"])
def test_field_name_renaming_to_keyword_raises(key):
    # `Break` renames to `break`, which is only accessible through getattr and is a
    # syntax error in a statically generated module.
    with pytest.raises(NameCollisionError, match="is a Python keyword"):
        create_registers(_one_field_schema(key))


# ---------------------------------------------------------------------------
# Reference bulk round-trip, where the emitted register and the generator oracle are
# wire- and dataframe-compatible for the same payload bytes (cross read/write).
# ---------------------------------------------------------------------------


def _random_records(dtype, n, seed):
    """``n`` deterministic records of ``dtype`` with random ASCII-range bytes.

    Bytes are held to 0..127 so every field varies while staying valid for any
    ``StringConverter`` member and free of float NaN/inf (which would defeat the
    value comparison); padding bytes are filled too but never read back.
    """
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 128, size=n * dtype.itemsize, dtype=np.uint8)
    return raw.view(dtype).copy()


@pytest.mark.parametrize("name", sorted(_device_registers()))
def test_emitted_register_bulk_matches_oracle(name, device_registers):
    emitted = device_registers[name]
    oracle = _device_registers()[name]
    records = _random_records(
        emitted.payload_class.payload_dtype, 5, seed=zlib.crc32(name.encode())
    )

    # Cross-write: same address / payload_type / byte layout -> identical wire bytes.
    buf = bytes(emitted.format_bulk(records))
    assert buf == bytes(oracle.format_bulk(records))

    # Cross-read via harp.data: the shared bytes decode to equal frames through
    # either class, including column names and decoded enum labels, which now agree.
    df_emitted = parse_to_dataframe(emitted, buf, time_index=False)
    df_oracle = parse_to_dataframe(oracle, buf, time_index=False)
    assert list(df_emitted.columns) == list(df_oracle.columns)
    assert df_emitted.equals(df_oracle)
