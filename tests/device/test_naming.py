"""The naming convention must match ``FirmwareNamingConvention`` in harp-tech/generators.

Every pair below is taken from the committed expected output of the generator
(``tests/ExpectedOutput/{core,device}.py`` against ``tests/Metadata/{core,device}.yml``),
so these lock the port to the C# behavior rather than to a re-derivation of it.
"""

import pytest
from harp.device.schema._naming import enum_member_name, field_name

# yml identifier -> generated enum member (SCREAMING_SNAKE_CASE)
ENUM_MEMBERS = [
    # Runs of capitals stay one word, and a trailing digit never separates.
    ("DIO0", "DIO0"),
    ("DIO1", "DIO1"),
    ("DIO2", "DIO2"),
    ("DIO3", "DIO3"),
    # A capital run ending where a lowercase word starts does separate.
    ("DIPort0", "DI_PORT0"),
    ("TestDIPort1", "TEST_DI_PORT1"),
    ("SupplyPort0", "SUPPLY_PORT0"),
    ("PortDIO1", "PORT_DIO1"),
    ("Pwm0", "PWM0"),
    ("Pwm1", "PWM1"),
    ("Pwm2", "PWM2"),
    ("Pwm3", "PWM3"),
    ("Position", "POSITION"),
    ("Displacement", "DISPLACEMENT"),
    # core.yml
    ("RestoreDefault", "RESTORE_DEFAULT"),
    ("RestoreEeprom", "RESTORE_EEPROM"),
    ("Save", "SAVE"),
    ("RestoreName", "RESTORE_NAME"),
    ("UpdateFirmware", "UPDATE_FIRMWARE"),
    ("BootFromDefault", "BOOT_FROM_DEFAULT"),
    ("BootFromEeprom", "BOOT_FROM_EEPROM"),
    ("ClockRepeater", "CLOCK_REPEATER"),
    ("ClockGenerator", "CLOCK_GENERATOR"),
    ("RepeaterCapability", "REPEATER_CAPABILITY"),
    ("GeneratorCapability", "GENERATOR_CAPABILITY"),
    ("ClockUnlock", "CLOCK_UNLOCK"),
    ("ClockLock", "CLOCK_LOCK"),
    ("Standby", "STANDBY"),
    ("Active", "ACTIVE"),
    ("Speed", "SPEED"),
    ("Disabled", "DISABLED"),
    ("Enabled", "ENABLED"),
]

# yml identifier -> generated payload field (snake_case)
PAYLOAD_FIELDS = [
    ("Analog0", "analog0"),
    ("Analog1", "analog1"),
    ("Analog2", "analog2"),
    ("Accelerometer", "accelerometer"),
    ("PwmPort", "pwm_port"),
    ("DutyCycle", "duty_cycle"),
    ("Frequency", "frequency"),
    ("EventsEnabled", "events_enabled"),
    ("Delta", "delta"),
    ("ProtocolVersion", "protocol_version"),
    ("FirmwareVersion", "firmware_version"),
    ("HardwareVersion", "hardware_version"),
    ("CoreId", "core_id"),
    ("InterfaceHash", "interface_hash"),
    ("Header", "header"),
    ("Data", "data"),
    ("Low", "low"),
    ("High", "high"),
    ("DigitalOutput", "digital_output"),
    ("PulseWidth", "pulse_width"),
    ("PulseCount", "pulse_count"),
    # core.yml, where a trailing capital run collapses either way it is spelled.
    ("OperationMode", "operation_mode"),
    ("DumpRegisters", "dump_registers"),
    ("MuteReplies", "mute_replies"),
    ("VisualIndicators", "visual_indicators"),
    ("OperationLed", "operation_led"),
    ("OperationLED", "operation_led"),
    ("Heartbeat", "heartbeat"),
]


@pytest.mark.parametrize(("source", "expected"), ENUM_MEMBERS)
def test_enum_member_name_matches_generator(source, expected):
    assert enum_member_name(source) == expected


@pytest.mark.parametrize(("source", "expected"), PAYLOAD_FIELDS)
def test_field_name_matches_generator(source, expected):
    assert field_name(source) == expected


def test_both_conventions_share_one_casing_pass():
    # The generator derives its field names from the same pass, differing only in case,
    # so the two can never disagree about where a word boundary falls.
    for source, _ in ENUM_MEMBERS + PAYLOAD_FIELDS:
        assert field_name(source) == enum_member_name(source).lower()


def test_already_converted_names_are_stable():
    # The generator output is a fixed point, so regenerating never drifts.
    for _, generated in ENUM_MEMBERS:
        assert enum_member_name(generated) == generated
    for _, generated in PAYLOAD_FIELDS:
        assert field_name(generated) == generated


@pytest.mark.parametrize("source", ["", "_", "0", "A", "a"])
def test_degenerate_inputs_do_not_raise(source):
    assert enum_member_name(source) == source.upper()
