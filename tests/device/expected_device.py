# This file was automatically generated and should not be edited directly.
# To make changes, edit the device metadata and regenerate the interface.

import enum
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from harp.protocol import (
    AnonymousPayload,
    BitMask,
    BoolConverter,
    Field,
    GroupMask,
    HarpVersion,
    HarpVersionConverter,
    IdentityConverter,
    PayloadType,
    RegisterBase,
    RegisterS32,
    RegisterU16,
    RegisterU8,
    StringConverter,
    StructPayload,
)
from harp.device.core import REGISTER_MAP as _CORE_REGISTER_MAP

from .converters import (
    DataConverter,
)


__all__ = [
    "DEVICE_NAME",
    "WHO_AM_I",
    "DEVICE_METADATA",
    "PortDigitalIOS",
    "PwmPort",
    "EncoderModeMask",
    "AnalogDataPayload",
    "ComplexConfigurationPayload",
    "VersionPayload",
    "CustomPayloadPayload",
    "CustomRawPayloadPayload",
    "CustomMemberConverterPayload",
    "BitmaskSplitterPayload",
    "PortDIOSetPayload",
    "StartPulsePayload",
    "StartPulseTrainPayload",
    "EncoderModePayload",
    "DigitalInputs",
    "AnalogData",
    "ComplexConfiguration",
    "Version",
    "CustomPayload",
    "CustomRawPayload",
    "CustomMemberConverter",
    "BitmaskSplitter",
    "Counter0",
    "PortDIOSet",
    "PulseDOPort0",
    "PulseDO0",
    "StartPulse",
    "StartPulseTrain",
    "EncoderMode",
    "REGISTER_MAP",
]

DEVICE_NAME: str = "Tests"
WHO_AM_I: int = 0
DEVICE_METADATA: str = """\
# yaml-language-server: $schema=https://harp-tech.org/draft-02/schema/registers.json
device: Tests
firmwareVersion: "0.1"
hardwareTargets: "1.0"
registers:
  DigitalInputs:
    address: 32
    type: U8
    access: Event
  AnalogData:
    address: 33
    type: Float
    length: 6
    access: Event
    payloadSpec:
      Analog0:
        offset: 0
      Analog1:
        offset: 1
      Analog2:
        offset: 2
      Accelerometer:
        offset: 3
        length: 3
  ComplexConfiguration:
    address: 34
    type: U8
    length: 17
    access: Write
    payloadSpec:
      PwmPort:
        offset: 0
        maskType: PwmPort
      DutyCycle:
        offset: 4
        length: 4
        interfaceType: float
      Frequency:
        offset: 8
        length: 4
        interfaceType: float
      EventsEnabled:
        offset: 12
        interfaceType: bool
      Delta:
        offset: 13
        length: 4
        interfaceType: uint
  Version:
    address: 35
    type: U8
    length: 32
    access: Event
    payloadSpec:
      ProtocolVersion:
        offset: 0
        length: 3
        interfaceType: HarpVersion
      FirmwareVersion:
        offset: 3
        length: 3
        interfaceType: HarpVersion
      HardwareVersion:
        offset: 6
        length: 3
        interfaceType: HarpVersion
      CoreId:
        offset: 9
        length: 3
        interfaceType: string
      InterfaceHash:
        offset: 12
        length: 20
  CustomPayload:
    address: 36
    type: U32
    length: 3
    access: Read
    interfaceType: HarpVersion
    converter: Payload
  CustomRawPayload:
    address: 37
    type: U32
    length: 3
    access: Read
    interfaceType: HarpVersion
    converter: RawPayload
  CustomMemberConverter:
    address: 38
    type: U8
    length: 3
    access: Read
    payloadSpec:
      Header:
        offset: 0
      Data:
        offset: 1
        length: 2
        interfaceType: int
        converter: Payload
  BitmaskSplitter:
    address: 39
    type: U8
    access: Write
    payloadSpec:
      Low:
        mask: 0xF
        interfaceType: int
        converter: Payload
      High:
        mask: 0xF0
        interfaceType: int
  Counter0:
    address: 40
    type: S32
    access: Event
  PortDIOSet:
    address: 41
    type: U8
    access: Write
    maskType: PortDigitalIOS
  PulseDOPort0: &pulseDO
    address: 42
    type: U16
    access: Write
    minValue: 1
  PulseDO0:
    <<: *pulseDO
    address: 43
  StartPulse:
    address: 100
    type: U16
    access: Write
    description: Starts a PWM pulse.
    payloadSpec: &startPulse
      DigitalOutput:
        offset: 0
        mask: 0xC00
        maskType: PwmPort
      PulseWidth:
        offset: 0
        mask: 0x3FF
        interfaceType: ushort
  StartPulseTrain:
    address: 101
    type: U16
    length: 2
    access: Write
    description: Starts a PWM pulse train.
    payloadSpec:
      <<: *startPulse
      Frequency:
        offset: 1
        mask: 0xFF00
        minValue: 1
        maxValue: 255
        defaultValue: 1
        interfaceType: byte
      PulseCount:
        offset: 1
        mask: 0xFF
        interfaceType: byte
  EncoderMode:
    address: 103
    type: U8
    access: Write
    maskType: EncoderModeMask
    description: Configures the operation mode of the encoder.
bitMasks:
  PortDigitalIOS:
    bits:
      DIO0: 0x1
      DIO1: 0x2
      DIO2: 0x4
      DIO3: 0x8
      DIPort0: 0x100
      TestDIPort1: 0x200
      SupplyPort0: 0x400
      PortDIO1: 0x800
groupMasks:
  PwmPort:
    values:
      Pwm0: 0x1
      Pwm1: 0x2
      Pwm2: 0x4
      Pwm3: 0xA
  EncoderModeMask:
    description: Specifies the type of encoder mode.
    values:
      Position: 0x0
      Displacement: 0x1
"""
"""The contents of the metadata file describing the device registers."""


class PortDigitalIOS(enum.IntFlag):
    DIO0 = 0x1
    DIO1 = 0x2
    DIO2 = 0x4
    DIO3 = 0x8
    DI_PORT0 = 0x100
    TEST_DI_PORT1 = 0x200
    SUPPLY_PORT0 = 0x400
    PORT_DIO1 = 0x800


class PwmPort(enum.IntEnum):
    PWM0 = 1
    PWM1 = 2
    PWM2 = 4
    PWM3 = 10


class EncoderModeMask(enum.IntEnum):
    """Specifies the type of encoder mode."""

    POSITION = 0
    DISPLACEMENT = 1


class AnalogDataPayload(StructPayload[np.float32], length=6):
    """Represents the payload of the AnalogData register."""

    analog0: np.float32 = Field(IdentityConverter(np.float32))
    analog1: np.float32 = Field(IdentityConverter(np.float32), offset=1)
    analog2: np.float32 = Field(IdentityConverter(np.float32), offset=2)
    accelerometer: NDArray[np.float32] = Field(
        IdentityConverter(np.dtype((np.float32, (3,)))), offset=3
    )


class ComplexConfigurationPayload(StructPayload[np.uint8], length=17):
    """Represents the payload of the ComplexConfiguration register."""

    pwm_port: PwmPort = GroupMask(enum=PwmPort, mask=0xFF)
    duty_cycle: np.float32 = Field(IdentityConverter(np.float32), offset=4)
    frequency: np.float32 = Field(IdentityConverter(np.float32), offset=8)
    events_enabled: bool = Field(BoolConverter(), offset=12)
    delta: np.uint32 = Field(IdentityConverter(np.uint32), offset=13)


class VersionPayload(StructPayload[np.uint8], length=32):
    """Represents the payload of the Version register."""

    protocol_version: HarpVersion = Field(HarpVersionConverter(np.uint8))
    firmware_version: HarpVersion = Field(HarpVersionConverter(np.uint8), offset=3)
    hardware_version: HarpVersion = Field(HarpVersionConverter(np.uint8), offset=6)
    core_id: str = Field(StringConverter(3), offset=9)
    interface_hash: NDArray[np.uint8] = Field(
        IdentityConverter(np.dtype((np.uint8, (20,)))), offset=12
    )


class CustomPayloadPayload(AnonymousPayload[np.uint32]):
    """Represents the payload of the CustomPayload register."""

    __value__: HarpVersion = Field(HarpVersionConverter(np.uint32))


class CustomRawPayloadPayload(AnonymousPayload[np.uint32]):
    """Represents the payload of the CustomRawPayload register."""

    __value__: HarpVersion = Field(HarpVersionConverter(np.uint32))


class CustomMemberConverterPayload(StructPayload[np.uint8], length=3):
    """Represents the payload of the CustomMemberConverter register."""

    header: np.uint8 = Field(IdentityConverter(np.uint8))
    data: np.int32 = Field(DataConverter(), offset=1)


class BitmaskSplitterPayload(StructPayload[np.uint8]):
    """Represents the payload of the BitmaskSplitter register."""

    low: np.int32 = Field(IdentityConverter(np.int32), mask=0xF)
    high: np.int32 = Field(IdentityConverter(np.int32), mask=0xF0)


class PortDIOSetPayload(AnonymousPayload[np.uint8]):
    """Represents the payload of the PortDIOSet register."""

    __value__: PortDigitalIOS = BitMask(enum=PortDigitalIOS)


class StartPulsePayload(StructPayload[np.uint16]):
    """Represents the payload of the StartPulse register."""

    digital_output: PwmPort = GroupMask(enum=PwmPort, mask=0xC00)
    pulse_width: np.uint16 = Field(IdentityConverter(np.uint16), mask=0x3FF)


class StartPulseTrainPayload(StructPayload[np.uint16], length=2):
    """Represents the payload of the StartPulseTrain register."""

    digital_output: PwmPort = GroupMask(enum=PwmPort, mask=0xC00)
    pulse_width: np.uint16 = Field(IdentityConverter(np.uint16), mask=0x3FF)
    frequency: np.uint8 = Field(
        IdentityConverter(np.uint8), mask=0xFF00, offset=1, default=np.uint8(1)
    )
    pulse_count: np.uint8 = Field(IdentityConverter(np.uint8), mask=0xFF, offset=1)


class EncoderModePayload(AnonymousPayload[np.uint8]):
    """Represents the payload of the EncoderMode register."""

    __value__: EncoderModeMask = GroupMask(enum=EncoderModeMask, mask=0xFF)


class DigitalInputs(RegisterU8):
    address: ClassVar[int] = 32


class AnalogData(RegisterBase[AnalogDataPayload]):
    address: ClassVar[int] = 33
    payload_type: ClassVar[PayloadType] = PayloadType.Float
    payload_class = AnalogDataPayload


class ComplexConfiguration(RegisterBase[ComplexConfigurationPayload]):
    address: ClassVar[int] = 34
    payload_type: ClassVar[PayloadType] = PayloadType.U8
    payload_class = ComplexConfigurationPayload


class Version(RegisterBase[VersionPayload]):
    address: ClassVar[int] = 35
    payload_type: ClassVar[PayloadType] = PayloadType.U8
    payload_class = VersionPayload


class CustomPayload(RegisterBase[HarpVersion]):
    address: ClassVar[int] = 36
    payload_type: ClassVar[PayloadType] = PayloadType.U32
    payload_class = CustomPayloadPayload


class CustomRawPayload(RegisterBase[HarpVersion]):
    address: ClassVar[int] = 37
    payload_type: ClassVar[PayloadType] = PayloadType.U32
    payload_class = CustomRawPayloadPayload


class CustomMemberConverter(RegisterBase[CustomMemberConverterPayload]):
    address: ClassVar[int] = 38
    payload_type: ClassVar[PayloadType] = PayloadType.U8
    payload_class = CustomMemberConverterPayload


class BitmaskSplitter(RegisterBase[BitmaskSplitterPayload]):
    address: ClassVar[int] = 39
    payload_type: ClassVar[PayloadType] = PayloadType.U8
    payload_class = BitmaskSplitterPayload


class Counter0(RegisterS32):
    address: ClassVar[int] = 40


class PortDIOSet(RegisterBase[PortDigitalIOS]):
    address: ClassVar[int] = 41
    payload_type: ClassVar[PayloadType] = PayloadType.U8
    payload_class = PortDIOSetPayload


class PulseDOPort0(RegisterU16):
    address: ClassVar[int] = 42


class PulseDO0(RegisterU16):
    address: ClassVar[int] = 43


class StartPulse(RegisterBase[StartPulsePayload]):
    """Starts a PWM pulse."""

    address: ClassVar[int] = 100
    payload_type: ClassVar[PayloadType] = PayloadType.U16
    payload_class = StartPulsePayload


class StartPulseTrain(RegisterBase[StartPulseTrainPayload]):
    """Starts a PWM pulse train."""

    address: ClassVar[int] = 101
    payload_type: ClassVar[PayloadType] = PayloadType.U16
    payload_class = StartPulseTrainPayload


class EncoderMode(RegisterBase[EncoderModeMask]):
    """Configures the operation mode of the encoder."""

    address: ClassVar[int] = 103
    payload_type: ClassVar[PayloadType] = PayloadType.U8
    payload_class = EncoderModePayload


REGISTER_MAP: dict[int, type[RegisterBase[Any]]] = {
    **_CORE_REGISTER_MAP,
    32: DigitalInputs,
    33: AnalogData,
    34: ComplexConfiguration,
    35: Version,
    36: CustomPayload,
    37: CustomRawPayload,
    38: CustomMemberConverter,
    39: BitmaskSplitter,
    40: Counter0,
    41: PortDIOSet,
    42: PulseDOPort0,
    43: PulseDO0,
    100: StartPulse,
    101: StartPulseTrain,
    103: EncoderMode,
}
