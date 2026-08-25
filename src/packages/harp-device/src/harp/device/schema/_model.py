"""Pydantic object model of a Harp ``device.yml`` / ``registers`` schema.

TODO: hand-maintained for now. Auto-generating it from the upstream
``harp-tech/protocol`` JSON schema is deferred until that schema stabilises.
"""

from enum import Enum
from typing import Annotated, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


class PayloadType(str, Enum):
    """Register payload element type. Values match the schema enum (the yml names)."""

    U8 = "U8"
    S8 = "S8"
    U16 = "U16"
    S16 = "S16"
    U32 = "U32"
    S32 = "S32"
    U64 = "U64"
    S64 = "S64"
    Float = "Float"


class Access(Enum):
    """The operations which can be used to access register data."""

    Read = "Read"
    """The register will accept a request to read the payload value."""

    Write = "Write"
    """The register will accept a request to write the payload value."""

    Event = "Event"
    """The device may send messages to the controller reporting the register contents."""


class Visibility(Enum):
    """Whether a register is exposed in the high-level interface."""

    public = "public"
    """The register is exposed to the high-level interface."""

    private = "private"
    """The register is hidden from the high-level interface."""


class Converter(Enum):
    """A custom converter used to parse or format a payload or payload member value."""

    None_ = "None"
    """No custom conversion is required."""

    Payload = "Payload"
    """The custom converter operates on the specified payload type."""

    RawPayload = "RawPayload"
    """The custom converter operates directly on raw payload bytes."""


class MaskValueItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int = Field(..., description="The numerical mask value.")
    description: Optional[str] = Field(
        None, description="A summary description of the mask value function."
    )

    def __int__(self) -> int:
        return self.value


class MaskValue(RootModel[Union[int, MaskValueItem]]):
    root: Union[int, MaskValueItem]

    def __int__(self) -> int:
        return int(self.root)


class BitMask(BaseModel):
    """A bit mask used for reading or writing specific registers."""

    description: Optional[str] = Field(
        None, description="A summary description of the bit mask function."
    )
    bits: Dict[str, MaskValue] = Field(..., description="The collection of bit mask values.")


class GroupMask(BaseModel):
    """A group mask used for reading or writing specific registers."""

    description: Optional[str] = Field(
        None, description="A summary description of the group mask function."
    )
    values: Dict[str, MaskValue] = Field(..., description="The collection of group mask values.")


class MaskType(RootModel[str]):
    root: str


class InterfaceType(RootModel[str]):
    root: str


class MinValue(RootModel[float]):
    root: float


class MaxValue(RootModel[float]):
    root: float


class DefaultValue(RootModel[float]):
    root: float


class PayloadMember(BaseModel):
    """A named member of a structured register payload."""

    mask: Optional[int] = Field(
        None, description="The mask used to read and write this payload member."
    )
    offset: Optional[int] = Field(
        None,
        description="The zero-based index at which encoding of this payload member starts.",
    )
    length: Annotated[
        Optional[int],
        Field(ge=1, description="The number of elements used to encode this payload member."),
    ] = None
    description: Optional[str] = Field(
        None, description="A summary description of this payload member."
    )
    minValue: Optional[MinValue] = Field(
        None, description="The minimum allowable value for the payload member."
    )
    maxValue: Optional[MaxValue] = Field(
        None, description="The maximum allowable value for the payload member."
    )
    defaultValue: Optional[DefaultValue] = Field(
        None, description="The default value for the payload member."
    )
    maskType: Optional[MaskType] = Field(
        None,
        description="The name of the bit mask or group mask used to represent this payload member.",
    )
    interfaceType: Optional[InterfaceType] = Field(
        None,
        description=(
            "The name of the type used to represent this payload member "
            "in the high-level interface."
        ),
    )
    converter: Optional[Converter] = Field(
        None,
        description="A custom converter used to parse or format this payload member.",
    )


class Register(BaseModel):
    """The functionality and operation of a specific register."""

    address: Annotated[int, Field(le=255, description="The unique 8-bit address of the register.")]
    type: PayloadType = Field(..., description="The type of the register payload.")
    length: Annotated[
        Optional[int], Field(ge=1, description="The length of the register payload.")
    ] = None
    access: Union[Access, List[Access]] = Field(
        ..., description="The expected use of the register."
    )
    description: Optional[str] = Field(
        None, description="A summary description of the register function."
    )
    minValue: Optional[MinValue] = Field(
        None, description="The minimum allowable value for the payload."
    )
    maxValue: Optional[MaxValue] = Field(
        None, description="The maximum allowable value for the payload."
    )
    defaultValue: Optional[DefaultValue] = Field(
        None, description="The default value for the payload."
    )
    maskType: Optional[MaskType] = Field(
        None,
        description="The name of the bit mask or group mask used to represent the payload value.",
    )
    visibility: Optional[Visibility] = Field(
        None, description="Whether the register function is exposed in the high-level interface."
    )
    volatile: Optional[bool] = Field(
        None, description="Whether register values can be saved in non-volatile memory."
    )
    payloadSpec: Optional[Dict[str, PayloadMember]] = Field(
        None,
        description=(
            "A collection of payload members describing the contents of the raw payload value."
        ),
    )
    interfaceType: Optional[InterfaceType] = Field(
        None,
        description=(
            "The name of the type used to represent the payload value in the high-level interface."
        ),
    )
    converter: Optional[Converter] = Field(
        None, description="A custom converter used to parse or format the payload value."
    )


class Registers(BaseModel):
    """A bare register collection, a header-less ``device.yml`` fragment."""

    description: Optional[str] = Field(
        None, description="A summary description of the register interface."
    )
    registers: Dict[str, Register] = Field(
        ..., description="The collection of registers implementing the device function."
    )
    bitMasks: Optional[Dict[str, BitMask]] = Field(
        None,
        description="The collection of masks available to be used with the different registers.",
    )
    groupMasks: Optional[Dict[str, GroupMask]] = Field(
        None,
        description=(
            "The collection of group masks available to be used with the different registers."
        ),
    )

    @model_validator(mode="after")
    def _names_are_distinct(self) -> "Registers":
        """Every generator target renders registers and masks into one namespace."""
        declared = (
            ("register", self.registers),
            ("bit mask", self.bitMasks or {}),
            ("group mask", self.groupMasks or {}),
        )
        seen: Dict[str, str] = {}
        for kind, names in declared:
            for name in names:
                if name in seen:
                    raise ValueError(
                        f"{name!r} is declared as both a {seen[name]} and a {kind}; "
                        f"rename one of them"
                    )
                seen[name] = kind
        return self


class DeviceModel(Registers):
    """A device schema: a `Registers` collection plus optional device identity.

    Every identity field is optional, so a header-less fragment, carrying
    just ``registers``, ``bitMasks`` or ``groupMasks``, is simply a ``DeviceModel`` with
    them all None, and parsing never needs to branch on "fragment vs full document".
    """

    device: Optional[str] = Field(None, description="The name of the device.")
    whoAmI: Optional[int] = Field(None, description="The unique identifier for this device type.")
    firmwareVersion: Optional[str] = Field(
        None, description="The version of the device firmware, as ``major.minor``."
    )
    hardwareTargets: Optional[str] = Field(
        None, description="The version of the device hardware, as ``major.minor``."
    )
