import sys
import types

import harp.device.core
import pytest
from harp.device.core import REGISTER_MAP as CORE_REGISTER_MAP
from harp.device.core import WhoAmI
from harp.device.schema import (
    DeviceModule,
    DeviceModuleLike,
    create_device_module,
    parse_device_schema,
)
from harp.protocol import RegisterBase

from . import expected_device
from .converters import DataConverter

CONVERTERS = {"DataConverter": DataConverter()}
MODULE_CONSTANTS = {"DEVICE_NAME", "REGISTER_MAP", "WHO_AM_I"}


@pytest.fixture
def test_module(device_yml):
    return create_device_module(device_yml, converters=CONVERTERS)


def test_returns_module_named_after_schema(test_module):
    assert isinstance(test_module, types.ModuleType)
    assert test_module.__name__ == "Tests"


def test_returns_device_module(test_module):
    # The subclass is what declares REGISTER_MAP, WHO_AM_I and the register names,
    # so a linter can resolve them on a module built at runtime.
    assert isinstance(test_module, DeviceModule)


def test_device_name_from_schema(test_module):
    # Recordings are written under this name, so a reader can match files by it.
    assert test_module.DEVICE_NAME == "Tests"


def test_whoami_defaults_to_zero_when_absent(test_module):
    # device.yml (application-device metadata) omits whoAmI.
    assert test_module.WHO_AM_I == 0


def test_whoami_from_schema():
    mod = create_device_module(
        "device: D\nwhoAmI: 1216\nregisters:\n  Foo: {address: 40, type: U16, access: Read}\n"
    )
    assert mod.WHO_AM_I == 1216


def test_docstring_from_schema(core_yml):
    # The docstring is the schema description rather than composed from the module name,
    # so a runtime module reads the same as the generated package for the same metadata.
    doc = create_device_module(core_yml).__doc__
    assert doc == parse_device_schema(core_yml).description
    assert doc


def test_docstring_absent_when_undeclared(test_module):
    # device.yml declares no description, and a generated package carries no module
    # docstring either, so both paths agree instead of one composing a name.
    assert test_module.__doc__ is None


def test_registers_are_accessible_by_name(test_module):
    assert test_module.AnalogData.address == 33
    assert test_module.EncoderMode.address == 103


def test_registers_are_accessible_by_address(test_module):
    reg_map = test_module.REGISTER_MAP
    assert reg_map[33].__name__ == "AnalogData"
    assert reg_map[103].__name__ == "EncoderMode"


def test_register_map_spreads_core(test_module):
    reg_map = test_module.REGISTER_MAP
    assert reg_map[0].__name__ == "WhoAmI"  # core register, always spread in
    assert reg_map[33].__name__ == "AnalogData"  # device-specific
    assert reg_map[103].__name__ == "EncoderMode"


def test_core_registers_are_not_named_by_module(test_module):
    # A core register has one definition, in harp.device, so a device module does
    # not re-export it. It is still in the address space the device can send from.
    assert not hasattr(test_module, "WhoAmI")
    assert test_module.REGISTER_MAP[0] is WhoAmI


def test_module_names_schema_declarations(test_module):
    # The module names what the schema declares, registers beside the enums and payload
    # classes they are built from, matching what a generated device package holds.
    named = {n for n in vars(test_module) if not n.startswith("_") and n not in MODULE_CONSTANTS}
    registers = {n for n in named if issubclass(getattr(test_module, n), RegisterBase)}
    addresses = {cls.address for cls in test_module.REGISTER_MAP.values()}
    assert all(getattr(test_module, n).address in addresses for n in registers)
    assert {"EncoderModeMask", "AnalogDataPayload"} <= named - registers
    assert {c.__name__ for c in CORE_REGISTER_MAP.values()}.isdisjoint(named)


def test_emitted_module_matches_device_protocol(test_module):
    assert isinstance(test_module, DeviceModuleLike)


def test_generated_package_matches_device_protocol():
    # expected_device is a sample of generator output, so this pins that what the
    # generator emits is accepted wherever a device module is required.
    assert isinstance(expected_device, DeviceModuleLike)


def test_core_registers_are_not_device_module():
    # They carry REGISTER_MAP but describe no device, so they cannot be passed
    # where a device module is required, such as to a DatasetReader.
    assert hasattr(harp.device.core, "REGISTER_MAP")
    assert not hasattr(harp.device.core, "WHO_AM_I")
    assert not isinstance(harp.device.core, DeviceModuleLike)


def test_unknown_name_raises_attribute_error(test_module):
    # A name not declared by the schema must raise rather than silently resolve.
    with pytest.raises(AttributeError, match="Nonexistent"):
        _ = test_module.Nonexistent


def test_named_registers_are_subset_of_address_space(test_module):
    # The two are deliberately different sets: the module names what the schema
    # declares, the map is everything the device can send.
    for cls in test_module.REGISTER_MAP.values():
        if cls.address >= 32:
            assert getattr(test_module, cls.__name__) is cls
    assert 0 in test_module.REGISTER_MAP


def test_device_register_overrides_core_on_clash():
    # A device register at a core address replaces it in the address space.
    mod = create_device_module(
        "device: Clash\nregisters:\n  Shadow: {address: 0, type: U32, access: Read}\n"
    )
    assert mod.REGISTER_MAP[0].__name__ == "Shadow"
    assert mod.Shadow.address == 0


def test_headerless_fragment_builds_nameless_device():
    # A register-only fragment declares no device, so it is nameless and unregistered.
    # The empty DEVICE_NAME is what sends a reader to the folder for a prefix, while
    # "Device" only keeps the module itself from being anonymous.
    mod = create_device_module("registers:\n  Foo: {address: 40, type: U16, access: Read}\n")
    assert mod.DEVICE_NAME == ""
    assert mod.WHO_AM_I == 0
    assert mod.REGISTER_MAP[40].__name__ == "Foo"
    assert mod.Foo.address == 40
    assert mod.__name__ == "Device"


def test_all_covers_declarations_and_module_constants(test_module):
    exported = set(test_module.__all__)
    assert {"REGISTER_MAP", "WHO_AM_I"} <= exported
    assert {"AnalogData", "EncoderMode"} <= exported
    assert {"EncoderModeMask", "AnalogDataPayload"} <= exported
    assert "WhoAmI" not in exported  # a core register is not re-exported
    assert exported - MODULE_CONSTANTS == {
        n for n in vars(test_module) if not n.startswith("_") and n not in MODULE_CONSTANTS
    }


def test_all_matches_generated_package(test_module):
    # expected_device is generator output for the same schema, so this pins that both
    # paths publish exactly the same surface, not merely equivalent registers.
    assert set(test_module.__all__) == set(expected_device.__all__)


def test_reused_core_masks_are_not_named_by_module():
    # A reused mask has one definition, in harp.device.core, so a device module resolves
    # registers against it without naming it, as it does for the core registers.
    mod = create_device_module(
        "device: CoreMasks\n"
        "registers:\n"
        "  EnableFlow: {address: 32, type: U8, access: Write, maskType: EnableFlag}\n"
    )
    assert not hasattr(mod, "EnableFlag")
    assert "EnableFlag" not in mod.__all__
    assert (
        mod.EnableFlow.payload_class._mro_descriptor("__value__")._enum
        is harp.device.core.EnableFlag
    )


def test_module_is_not_registered_in_sys_modules(test_module):
    # Two schemas may share a device name, so the module is handed back unbound.
    assert sys.modules.get(test_module.__name__) is not test_module


def test_emitted_registers_carry_module_name(test_module):
    assert test_module.AnalogData.__module__ == "Tests"


def test_emitted_registers_are_usable(test_module):
    reg = test_module.AnalogData
    # The emitted register class round-trips through the Device.read/write frame path.
    frame = reg.format(
        reg.payload_class(analog0=1.0, analog1=2.0, analog2=3.0, accelerometer=[4, 5, 6])
    )
    assert isinstance(frame, (bytes, bytearray))
