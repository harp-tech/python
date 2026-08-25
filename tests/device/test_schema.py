import pytest
from pydantic import ValidationError
from pydantic_yaml import to_yaml_str

from harp.device.schema import parse_device_schema
from harp.device.schema._model import Access, DeviceModel, PayloadType, Register


def test_parse_full_device(device_yml):
    m = parse_device_schema(device_yml)
    assert isinstance(m, DeviceModel)
    assert m.device == "Tests"
    assert m.whoAmI is None  # this application-device metadata omits whoAmI
    assert m.description is None  # and omits a top-level description
    assert "AnalogData" in m.registers
    ad = m.registers["AnalogData"]
    assert ad.type is PayloadType.Float
    assert ad.length == 6
    assert list(ad.payloadSpec) == ["Analog0", "Analog1", "Analog2", "Accelerometer"]


def test_colliding_declaration_names_are_rejected():
    # Registers and masks are rendered into one namespace, so a name describing two of
    # them would leave whichever came last and silently lose the other.
    schema = (
        "device: Clash\n"
        "registers:\n"
        "  Mode: {address: 32, type: U8, access: Read, maskType: Mode}\n"
        "groupMasks:\n"
        "  Mode:\n"
        "    values:\n"
        "      Idle: {value: 0}\n"
    )
    with pytest.raises(ValidationError, match="both a register and a group mask"):
        parse_device_schema(schema)


def test_parse_fragment_yields_null_device():
    m = parse_device_schema("registers:\n  Foo: {address: 40, type: U16, access: Read}\n")
    assert isinstance(m, DeviceModel)
    assert m.device is None  # header-less fragment -> identity fields are None
    assert m.registers["Foo"].type is PayloadType.U16


def test_parse_bytes_decodes_as_utf8_regardless_of_locale():
    # A YAML stream declares its own encoding, so reading a schema as bytes decodes it
    # correctly where read_text() without an explicit encoding follows the locale.
    schema = (
        "registers:\n"
        "  Poke:\n"
        "    address: 40\n"
        "    type: U8\n"
        "    access: Read\n"
        "    description: µV threshold\n"
    )
    m = parse_device_schema(schema.encode("utf-8"))
    assert m.registers["Poke"].description == "µV threshold"
    assert (
        m.registers["Poke"].description == parse_device_schema(schema).registers["Poke"].description
    )


def test_reserved_word_mask_keys_stay_strings():
    # Off, On, Yes and No are booleans in YAML 1.1, so a 1.1 parser keys these values by
    # True and False.
    schema = (
        "registers:\n"
        "  Indicators: {address: 32, type: U8, access: Write, maskType: LedState}\n"
        "groupMasks:\n"
        "  LedState:\n"
        "    values:\n"
        "      Off: 0\n"
        "      On: 1\n"
    )
    m = parse_device_schema(schema)
    assert {k: int(v) for k, v in m.groupMasks["LedState"].values.items()} == {"Off": 0, "On": 1}


def test_parse_core_registers(core_yml):
    c = parse_device_schema(core_yml)
    assert "WhoAmI" in c.registers
    assert c.description  # the core metadata declares a top-level description
    # 'None' bit name stays a string, not YAML null.
    assert "None" in c.bitMasks["ResetFlags"].bits


def test_bool_values_preserved(core_yml):
    c = parse_device_schema(core_yml)
    assert c.registers["TimestampSeconds"].volatile is True


def test_access_list_and_scalar(core_yml):
    c = parse_device_schema(core_yml)
    # TimestampSeconds has a list access [Read, Write, Event]; WhoAmI a scalar.
    assert isinstance(c.registers["TimestampSeconds"].access, list)


def test_absent_register_length_stays_absent():
    # An absent length means a single value. A declared 1 means an array of one element. The
    # model has to keep them apart.
    registers = parse_device_schema(
        "registers:\n"
        "  Absent: {address: 32, type: U16, access: Read}\n"
        "  One: {address: 33, type: U16, length: 1, access: Read}\n"
    ).registers
    assert registers["Absent"].length is None
    assert registers["One"].length == 1


def test_absent_member_length_stays_absent():
    members = (
        parse_device_schema(
            "registers:\n"
            "  R:\n"
            "    address: 32\n"
            "    type: U8\n"
            "    length: 4\n"
            "    access: Read\n"
            "    payloadSpec:\n"
            "      Absent: {offset: 0}\n"
            "      One: {offset: 1, length: 1}\n"
        )
        .registers["R"]
        .payloadSpec
    )
    assert members is not None
    assert members["Absent"].length is None
    assert members["One"].length == 1


@pytest.mark.parametrize(
    "schema",
    [
        "registers:\n  R: {address: 32, type: U16, length: 0, access: Read}\n",
        "registers:\n"
        "  R:\n"
        "    address: 32\n"
        "    type: U8\n"
        "    access: Read\n"
        "    payloadSpec:\n"
        "      Zero: {offset: 0, length: 0}\n",
    ],
    ids=["register", "member"],
)
def test_zero_length_is_rejected(schema):
    # registers.json sets a minimum of 1 at both levels, so 0 is never declared.
    with pytest.raises(ValidationError):
        parse_device_schema(schema)


def test_serialized_schema_round_trips(device_yml):
    # An absent length is None, so exclude_none writes a valid device.yml.
    model = parse_device_schema(device_yml)
    written = to_yaml_str(model, exclude_none=True)
    assert "length:" in written  # the declared ones survive
    for name, register in model.registers.items():
        if register.length is not None:
            assert f"length: {register.length}" in written, name
    assert parse_device_schema(written) == model


def test_modified_schema_round_trips(device_yml):
    # Read, change, write back. A register built in code has no length until one is set.
    model = parse_device_schema(device_yml)
    model.registers["DigitalInputs"].length = 4
    model.registers["Added"] = Register(address=60, type=PayloadType.U16, access=Access.Read)
    written = to_yaml_str(model, exclude_none=True)
    reparsed = parse_device_schema(written)
    assert reparsed.registers["DigitalInputs"].length == 4
    assert reparsed.registers["Added"].length is None
    assert reparsed == model
