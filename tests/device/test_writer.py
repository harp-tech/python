import queue
import threading
import types
from collections.abc import Callable, Iterable
from functools import partial

import pytest
from harp.data import DatasetReader, open_dataset, parse_to_dataframe
from harp.device.client import (
    Device,
    DeviceWriter,
    TransportError,
    attach_writer,
)
from harp.device.client._writer import _default_file_formatter
from harp.device.core import WhoAmI
from harp.device.schema import create_device_module
from harp.protocol import HarpMessage, MessageType

from tests.fixtures import make_frame_from_raw

_U16 = 0x02
"""Payload-type byte of a U16 payload, as the size nibble alone."""


def default_file_formatter_with_suffix(device_name: str, address: int, suffix: str) -> str:
    """The standard name carrying its optional trailing field, as a caller would write it."""
    return f"{device_name}_{address}_{suffix}.bin"


class _ScriptedTransport:
    """A transport replying with whatever ``on_write`` returns for each request."""

    def __init__(self) -> None:
        self.on_write: Callable[[bytes], Iterable[bytes]] | None = None
        self.failing = False
        self._inbox: queue.SimpleQueue[bytes] = queue.SimpleQueue()

    def open(self) -> None: ...

    def close(self) -> None: ...

    def write(self, data: bytes) -> None:
        if self.on_write is not None:
            for frame in self.on_write(data):
                self._inbox.put(frame)

    def read(self) -> bytes:
        if self.failing:
            raise TransportError("simulated transport failure")
        try:
            return self._inbox.get(timeout=0.01)
        except queue.Empty:
            return b""

    def inject(self, frame: bytes) -> None:
        self._inbox.put(frame)


def _event(address: int, value: int) -> bytes:
    return make_frame_from_raw(
        MessageType.Event,
        address,
        255,
        _U16,
        value.to_bytes(2, "little"),
        timestamp=b"\x01\x00\x00\x00\x00\x00",
    )


@pytest.fixture
def nameless_module():
    """A header-less register fragment: it declares no device, so DEVICE_NAME is empty."""
    return create_device_module("registers:\n  Foo: {address: 40, type: U16, access: Read}\n")


@pytest.fixture
def module(device_yml):
    # require_converters=False: the test device.yml declares a custom DataConverter that
    # is not injected here, and native decoding is enough to exercise the file layout.
    return create_device_module(device_yml, require_converters=False)


# ---------------------------------------------------------------------------
# File naming and layout
# ---------------------------------------------------------------------------


def test_default_formatter_names_the_plain_standard_form():
    assert _default_file_formatter("Behavior", 44) == "Behavior_44.bin"


def test_writes_one_file_per_register_named_after_the_device(module, tmp_path):
    root = tmp_path / "session.harp"
    with DeviceWriter(module, root) as writer:
        writer.write(HarpMessage.parse(_event(32, 7)))
        writer.write(HarpMessage.parse(_event(33, 9)))
        writer.write(HarpMessage.parse(_event(32, 8)))
    written = {p.name for p in root.glob("*.bin")}
    assert written == {"Tests_32.bin", "Tests_33.bin"}
    assert writer.paths == {32: root / "Tests_32.bin", 33: root / "Tests_33.bin"}


def test_writes_the_complete_frame_in_arrival_order(module, tmp_path):
    frames = [_event(32, value) for value in (1, 2, 3)]
    with DeviceWriter(module, tmp_path) as writer:
        for frame in frames:
            writer.write(HarpMessage.parse(frame))
    # The whole frame is recorded, header and checksum included, so the file is what a
    # register parser reads without consulting anything else.
    assert (tmp_path / "Tests_32.bin").read_bytes() == b"".join(frames)


def test_a_custom_formatter_supplies_the_trailing_suffix_field(module, tmp_path):
    # The standard's optional <suffix> field is one thing a custom formatter is for.
    formatter = partial(default_file_formatter_with_suffix, "Tests", suffix="0")
    with DeviceWriter(module, tmp_path, formatter=formatter) as writer:
        writer.write(HarpMessage.parse(_event(32, 1)))
    assert (tmp_path / "Tests_32_0.bin").is_file()


def test_a_custom_formatter_decides_the_whole_layout(module, tmp_path):
    def by_folder(address):
        return f"{address}/Tests.bin"

    with DeviceWriter(module, tmp_path, formatter=by_folder) as writer:
        writer.write(HarpMessage.parse(_event(32, 1)))
    # A formatter naming a subfolder gets it created rather than failing on the open.
    assert (tmp_path / "32" / "Tests.bin").read_bytes() == _event(32, 1)
    assert writer.paths == {32: tmp_path / "32" / "Tests.bin"}


def test_a_custom_formatter_reads_back_through_the_matching_resolver(module, tmp_path):
    def by_folder(address):
        return f"{address}/Tests.bin"

    def resolve(root, name):
        return {int(p.parent.name): [p] for p in sorted(root.glob(f"*/{name}.bin"))}

    frames = [_event(32, value) for value in (1, 2)]
    with DeviceWriter(module, tmp_path, formatter=by_folder) as writer:
        for frame in frames:
            writer.write(HarpMessage.parse(frame))
    reader = DatasetReader(module, tmp_path, resolver=resolve)
    expected = parse_to_dataframe(module.REGISTER_MAP[32], b"".join(frames))
    assert reader.read(32).equals(expected)


def test_the_prefix_is_the_device_name_of_the_module(module, tmp_path):
    # Nothing can override it, so files are always named after the device that made them.
    with DeviceWriter(module, tmp_path) as writer:
        writer.write(HarpMessage.parse(_event(32, 1)))
    assert writer.name == module.DEVICE_NAME == "Tests"
    assert (tmp_path / "Tests_32.bin").is_file()


def test_a_module_without_a_name_needs_one(nameless_module, tmp_path):
    # A header-less register fragment declares no device, so nothing names the files.
    assert nameless_module.DEVICE_NAME == ""
    with pytest.raises(ValueError, match="DEVICE_NAME"):
        DeviceWriter(nameless_module, tmp_path)


def test_a_custom_formatter_lifts_the_name_requirement(nameless_module, tmp_path):
    # The name only feeds the default formatter, so a custom one needs no device name.
    def by_address(address):
        return f"{address}.bin"

    with DeviceWriter(nameless_module, tmp_path, formatter=by_address) as writer:
        writer.write(HarpMessage.parse(_event(32, 1)))
    assert (tmp_path / "32.bin").is_file()


def test_the_folder_is_created(module, tmp_path):
    root = tmp_path / "nested" / "session.harp"
    DeviceWriter(module, root).close()
    assert root.is_dir()


# ---------------------------------------------------------------------------
# The schema copied into the folder
# ---------------------------------------------------------------------------


def test_the_schema_of_the_module_is_copied_into_the_folder(module, device_yml, tmp_path):
    writer = DeviceWriter(module, tmp_path)
    writer.close()
    assert writer.schema_path == tmp_path / "device.yml"
    assert writer.schema_path.read_text() == device_yml


def test_a_module_carrying_no_schema_raises(tmp_path):
    # DeviceModuleLike requires DEVICE_METADATA, so a module without one is broken, not a
    # folder to write undescribed.
    broken = types.ModuleType("Broken")
    broken.DEVICE_NAME = "Broken"
    broken.WHO_AM_I = 0
    broken.REGISTER_MAP = {}
    with pytest.raises(ValueError, match="DEVICE_METADATA"):
        DeviceWriter(broken, tmp_path)


def test_a_module_carrying_an_empty_schema_raises(module, tmp_path):
    module.DEVICE_METADATA = b""
    with pytest.raises(ValueError, match="DEVICE_METADATA"):
        DeviceWriter(module, tmp_path)


# ---------------------------------------------------------------------------
# Overwrite
# ---------------------------------------------------------------------------


def test_an_existing_schema_is_not_silently_replaced(module, tmp_path):
    DeviceWriter(module, tmp_path).close()
    with pytest.raises(FileExistsError, match="device.yml"):
        DeviceWriter(module, tmp_path)


def test_an_existing_register_file_is_not_silently_replaced(module, tmp_path):
    with DeviceWriter(module, tmp_path) as first:
        first.write(HarpMessage.parse(_event(32, 1)))
    # overwrite= lets the schema through, so the register file is what is under test.
    (tmp_path / "device.yml").unlink()
    with DeviceWriter(module, tmp_path) as second:
        with pytest.raises(FileExistsError):
            second.write(HarpMessage.parse(_event(32, 2)))


def test_overwrite_replaces_both_the_schema_and_the_files(module, tmp_path):
    with DeviceWriter(module, tmp_path) as first:
        first.write(HarpMessage.parse(_event(32, 1)))
        first.write(HarpMessage.parse(_event(32, 2)))
    with DeviceWriter(module, tmp_path, overwrite=True) as second:
        second.write(HarpMessage.parse(_event(32, 3)))
    assert (tmp_path / "Tests_32.bin").read_bytes() == _event(32, 3)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_writing_to_a_closed_writer_raises(module, tmp_path):
    writer = DeviceWriter(module, tmp_path)
    writer.close()
    assert writer.closed
    with pytest.raises(ValueError, match="closed"):
        writer.write(HarpMessage.parse(_event(32, 1)))


def test_close_is_idempotent(module, tmp_path):
    writer = DeviceWriter(module, tmp_path)
    writer.close()
    writer.close()


def test_flush_exposes_frames_before_close(module, tmp_path):
    with DeviceWriter(module, tmp_path) as writer:
        writer.write(HarpMessage.parse(_event(32, 1)))
        writer.flush()
        assert (tmp_path / "Tests_32.bin").read_bytes() == _event(32, 1)


def test_concurrent_writers_do_not_interleave_frames(module, tmp_path):
    frame = _event(32, 1)
    with DeviceWriter(module, tmp_path) as writer:

        def run() -> None:
            for _ in range(50):
                writer.write(HarpMessage.parse(frame))

        threads = [threading.Thread(target=run) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert (tmp_path / "Tests_32.bin").read_bytes() == frame * 200


# ---------------------------------------------------------------------------
# attach_writer
# ---------------------------------------------------------------------------


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = threading.Event()
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        deadline.wait(0.01)
    raise AssertionError("timed out waiting for the recording")


def test_attach_writer_records_the_whole_stream(module, tmp_path):
    transport = _ScriptedTransport()
    root = tmp_path / "session.harp"
    with Device(transport, module) as device:
        with attach_writer(device, root) as writer:
            transport.inject(_event(32, 1))
            transport.inject(_event(33, 2))
            _wait_for(lambda: set(writer.paths) == {32, 33})
        assert writer.closed
    assert (root / "Tests_32.bin").read_bytes() == _event(32, 1)
    assert (root / "Tests_33.bin").read_bytes() == _event(33, 2)


def test_attach_writer_records_read_replies_by_default(module, tmp_path):
    # The register dump a device performs on request arrives as Read messages, so a
    # recording that only kept Event would lose the configuration it ran under.
    transport = _ScriptedTransport()
    reply = make_frame_from_raw(MessageType.Read, 0, 255, _U16, (1216).to_bytes(2, "little"))
    transport.on_write = lambda _data: [reply]
    with Device(transport, module) as device:
        with attach_writer(device, tmp_path) as writer:
            assert int(device.read(WhoAmI).payload) == 1216
            _wait_for(lambda: 0 in writer.paths)
    assert (tmp_path / "Tests_0.bin").read_bytes() == reply


def test_attach_writer_narrows_to_the_requested_message_types(module, tmp_path):
    transport = _ScriptedTransport()
    reply = make_frame_from_raw(MessageType.Read, 0, 255, _U16, (1216).to_bytes(2, "little"))
    transport.on_write = lambda _data: [reply]
    with Device(transport, module) as device:
        with attach_writer(device, tmp_path, message_types=MessageType.Event) as writer:
            device.read(WhoAmI)
            transport.inject(_event(32, 1))
            _wait_for(lambda: 32 in writer.paths)
    assert set(writer.paths) == {32}


def test_attach_writer_takes_the_name_and_schema_from_the_device_module(
    module, device_yml, tmp_path
):
    transport = _ScriptedTransport()
    with Device(transport, module) as device:
        with attach_writer(device, tmp_path) as writer:
            assert writer.name == "Tests"
    assert (tmp_path / "device.yml").read_text() == device_yml


def test_closing_the_writer_detaches_it_from_the_device(module, tmp_path):
    transport = _ScriptedTransport()
    with Device(transport, module) as device:
        writer = attach_writer(device, tmp_path)
        transport.inject(_event(32, 1))
        _wait_for(lambda: 32 in writer.paths)
        writer.close()
        transport.inject(_event(33, 2))
        transport.inject(_event(32, 2))
        # Nothing after the close is recorded, and no handler error is raised for it.
        _wait_for(lambda: True)
    assert set(writer.paths) == {32}
    assert (tmp_path / "Tests_32.bin").read_bytes() == _event(32, 1)


def test_several_recordings_over_one_open_device(module, tmp_path):
    transport = _ScriptedTransport()
    with Device(transport, module) as device:
        with attach_writer(device, tmp_path / "first.harp") as first:
            transport.inject(_event(32, 1))
            _wait_for(lambda: 32 in first.paths)
        with attach_writer(device, tmp_path / "second.harp") as second:
            transport.inject(_event(32, 2))
            _wait_for(lambda: 32 in second.paths)
    assert (tmp_path / "first.harp" / "Tests_32.bin").read_bytes() == _event(32, 1)
    assert (tmp_path / "second.harp" / "Tests_32.bin").read_bytes() == _event(32, 2)


def test_attach_writer_forwards_the_formatter(module, tmp_path):
    def by_folder(address):
        return f"{address}/Tests.bin"

    transport = _ScriptedTransport()
    with Device(transport, module) as device:
        with attach_writer(device, tmp_path, formatter=by_folder) as writer:
            transport.inject(_event(32, 1))
            _wait_for(lambda: 32 in writer.paths)
    assert (tmp_path / "32" / "Tests.bin").is_file()


def test_attach_writer_refuses_a_device_opened_without_a_module(tmp_path):
    # The module is what names the files and describes the folder, so there is nothing
    # to record with when a device was opened without one.
    with Device(_ScriptedTransport()) as device:
        with pytest.raises(ValueError, match="without a module"):
            attach_writer(device, tmp_path)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_a_recorded_folder_reads_back_through_open_dataset(module, tmp_path):
    root = tmp_path / "session.harp"
    frames = [_event(32, value) for value in (1, 2, 3)]
    with DeviceWriter(module, root) as writer:
        for frame in frames:
            writer.write(HarpMessage.parse(frame))
    # The folder carries its own device.yml, so it opens without a module in hand.
    reader = open_dataset(root, require_converters=False)
    assert reader.name == "Tests"
    assert reader.contents["DigitalInputs"] == 32
    assert reader.read(32).equals(parse_to_dataframe(module.REGISTER_MAP[32], b"".join(frames)))


def test_a_suffixed_recording_reads_back_by_suffix(module, tmp_path):
    root = tmp_path / "session.harp"
    formatter = partial(default_file_formatter_with_suffix, "Tests", suffix="0")
    with DeviceWriter(module, root, formatter=formatter) as writer:
        writer.write(HarpMessage.parse(_event(32, 1)))
    reader = DatasetReader(module, root)
    assert reader.paths[32] == [root / "Tests_32_0.bin"]
    assert len(reader.read(32, suffix="0")) == 1
