import re

import numpy as np
import pandas as pd
import pytest
from harp.data import (
    REFERENCE_EPOCH,
    DatasetReader,
    open_dataset,
    parse_to_dataframe,
)
from harp.device.core import TimestampSeconds, WhoAmI
from harp.device.schema import create_device_module


def _timestamps(n):
    return np.arange(n, dtype=np.float64)


def _records(cls, n, seed):
    dtype = cls.payload_class.payload_dtype
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 128, size=n * dtype.itemsize, dtype=np.uint8)
    return np.frombuffer(raw, dtype=dtype, count=n).copy()


@pytest.fixture
def emitted_module(device_yml):
    # require_converters=False: the test device.yml uses a custom DataConverter we don't inject
    # here; native decoding is enough to exercise file resolution and parsing.
    return create_device_module(device_yml, require_converters=False)


@pytest.fixture
def dataset(emitted_module, tmp_path):
    """A dataset folder with three app registers."""
    mod = emitted_module
    name = mod.DEVICE_NAME
    addresses = [a for a in sorted(mod.REGISTER_MAP) if a >= 32][:3]
    specs = {}
    for address in addresses:
        cls = mod.REGISTER_MAP[address]
        records = _records(cls, 5, seed=address)
        timestamps = _timestamps(5)
        buf = bytes(cls.format_bulk(records, timestamps=timestamps))
        (tmp_path / f"{name}_{address}.bin").write_bytes(buf)
        specs[address] = (cls, buf)
    return mod, name, tmp_path, specs


def test_read_by_class_and_by_address(dataset):
    mod, _name, root, specs = dataset
    reader = DatasetReader(mod, root)
    for address, (cls, buf) in specs.items():
        expected = parse_to_dataframe(cls, buf)
        assert reader.read(cls).equals(expected)
        assert reader.read(address).equals(expected)


def test_read_by_class_from_module_namespace(dataset):
    mod, _name, root, specs = dataset
    reader = DatasetReader(mod, root)
    for address, (cls, _buf) in specs.items():
        # The register accessed by name off the module is the one at that address.
        assert reader.read(getattr(mod, cls.__name__)).equals(reader.read(address))


def test_read_by_register_name(dataset):
    mod, _name, root, specs = dataset
    reader = DatasetReader(mod, root)
    for address, (cls, _buf) in specs.items():
        assert reader.read(cls.__name__).equals(reader.read(address))


def test_unknown_name_raises_key_error(dataset):
    mod, _name, root, _specs = dataset
    reader = DatasetReader(mod, root)
    with pytest.raises(KeyError):
        reader.read("NotARegister")


def test_reads_common_registers_not_named_by_module(emitted_module, tmp_path):
    # A device module names only its own registers, but a session folder also holds
    # files for the core ones, so the reader must still decode those.
    mod = emitted_module
    assert not hasattr(mod, "WhoAmI")  # imported from harp.device, not re-exported

    for cls in (WhoAmI, TimestampSeconds):
        records = _records(cls, 4, seed=cls.address)
        buf = bytes(cls.format_bulk(records, timestamps=_timestamps(4)))
        (tmp_path / f"{mod.DEVICE_NAME}_{cls.address}.bin").write_bytes(buf)

    reader = DatasetReader(mod, tmp_path)
    # By address, by the class imported from harp.device, and by name, which resolves
    # through the register map and so covers more than the module namespace.
    assert len(reader.read(WhoAmI.address)) == 4
    assert reader.read(WhoAmI).equals(reader.read(WhoAmI.address))
    assert reader.read("WhoAmI").equals(reader.read(WhoAmI.address))
    assert set(reader.contents) == {"WhoAmI", "TimestampSeconds"}


def test_read_gives_time_index(dataset):
    # Every register reads with the same index, so frames aggregate across sessions.
    mod, _name, root, specs = dataset
    reader = DatasetReader(mod, root)
    for address in specs:
        assert reader.read(address).index.name == "Time"


def test_timestamp_false_gives_range_index(dataset):
    # The diagnostic escape hatch, and the only way to read frames carrying no timestamp.
    mod, _name, root, specs = dataset
    reader = DatasetReader(mod, root)
    df = reader.read(next(iter(specs)), time_index=False)
    assert df.index.name is None


def test_untimestamped_frames_raise_value_error(emitted_module, tmp_path):
    # A recording every device should be incapable of producing, so it surfaces rather
    # than reading back with a silently different index.
    mod = emitted_module
    address = next(a for a in sorted(mod.REGISTER_MAP) if a >= 32)
    cls = mod.REGISTER_MAP[address]
    (tmp_path / f"{mod.DEVICE_NAME}_{address}.bin").write_bytes(
        bytes(cls.format_bulk(_records(cls, 3, seed=1)))
    )

    reader = DatasetReader(mod, tmp_path)

    with pytest.raises(ValueError, match="no timestamp data"):
        reader.read(address)
    assert len(reader.read(address, time_index=False)) == 3


def test_time_index_is_float_seconds_without_epoch(dataset):
    mod, _name, root, specs = dataset
    reader = DatasetReader(mod, root)
    address = next(iter(specs))
    df = reader.read(address)
    assert df.index.name == "Time"
    assert list(df.index) == [0.0, 1.0, 2.0, 3.0, 4.0]  # arange(5) seconds from the fixture


def test_dataset_epoch_gives_absolute_datetime_index(dataset):
    mod, _name, root, specs = dataset
    reader = DatasetReader(mod, root, epoch=REFERENCE_EPOCH)
    address = next(iter(specs))
    df = reader.read(address)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "Time"
    # Harp seconds are measured from the reference epoch (timestamps were arange(5)).
    assert df.index[0] == pd.Timestamp(REFERENCE_EPOCH)
    assert df.index[2] == pd.Timestamp(REFERENCE_EPOCH) + pd.Timedelta(seconds=2)


def test_dataset_epoch_applies_to_every_register(dataset):
    # The anchor is set once for the dataset, so no read can index on a different clock
    # from its siblings and frames from several registers share one index type.
    mod, _name, root, specs = dataset
    reader = DatasetReader(mod, root, epoch=REFERENCE_EPOCH)
    for address in specs:
        assert isinstance(reader.read(address).index, pd.DatetimeIndex)


def test_parse_takes_epoch_without_dataset(dataset):
    # parse_to_dataframe has no dataset to take the anchor from, so it accepts one.
    _mod, _name, _root, specs = dataset
    cls, buf = specs[next(iter(specs))]
    df = parse_to_dataframe(cls, buf, epoch=REFERENCE_EPOCH)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0] == pd.Timestamp(REFERENCE_EPOCH)


def test_suffix_chunks_are_concatenated(emitted_module, tmp_path):
    # Chunk suffixes in this test are ISO 8601 UTC timestamps in basic format, so
    # filename order is chronological order. Written newest first to test the sorting.
    mod = emitted_module
    name = mod.DEVICE_NAME
    address = next(a for a in sorted(mod.REGISTER_MAP) if a >= 32)
    cls = mod.REGISTER_MAP[address]
    chunks = {
        "20260816T090000Z": bytes(
            cls.format_bulk(_records(cls, 3, seed=1), timestamps=_timestamps(3))
        ),
        "20260816T100000Z": bytes(
            cls.format_bulk(_records(cls, 2, seed=2), timestamps=_timestamps(2))
        ),
    }
    for suffix in reversed(list(chunks)):
        (tmp_path / f"{name}_{address}_{suffix}.bin").write_bytes(chunks[suffix])

    reader = DatasetReader(mod, tmp_path)
    combined = parse_to_dataframe(cls, b"".join(chunks.values()))
    assert reader.read(cls).reset_index(drop=True).equals(combined.reset_index(drop=True))
    # A specific chunk can still be selected by suffix.
    earliest = parse_to_dataframe(cls, chunks["20260816T090000Z"])
    assert reader.read(cls, suffix="20260816T090000Z").equals(earliest)


def test_non_module_raises_on_construction(dataset):
    # The register map is read at construction, so a module without one is rejected.
    _mod, name, root, _specs = dataset
    with pytest.raises(AttributeError, match="REGISTER_MAP"):
        DatasetReader(object, root, name=name, validate=False)


def test_explicit_name_overrides(dataset):
    mod, name, root, _specs = dataset
    reader = DatasetReader(mod, root, name=name)
    assert isinstance(reader, DatasetReader)
    assert reader.name == name


def test_name_from_device_name_not_module_name(dataset):
    # The prefix follows DEVICE_NAME rather than __name__, so rebinding the module
    # does not change which files are read.
    mod, name, root, _specs = dataset
    mod.__name__ = "not_the_device_name"
    assert DatasetReader(mod, root).name == name


def test_empty_dataset_reads_empty(emitted_module, tmp_path):
    # A session that logged nothing is a dataset with no data, not a failure.
    reader = DatasetReader(emitted_module, tmp_path)
    assert reader.name == emitted_module.DEVICE_NAME
    assert reader.paths == {}


def test_nameless_module_raises_on_construction(dataset):
    # A header-less schema declares no device, so its module names nothing and file
    # names are not consulted. This is the reader failing to be set up, not empty data.
    _mod, name, root, _specs = dataset
    nameless = create_device_module("registers:\n  Foo: {address: 40, type: U16, access: Read}\n")
    assert nameless.DEVICE_NAME == ""
    with pytest.raises(ValueError, match="Pass name="):
        DatasetReader(nameless, root)
    assert DatasetReader(nameless, root, name=name).name == name


def test_unlogged_register_reads_empty(dataset):
    # The schema describes the columns whether or not anything was recorded, so a
    # declared register with no file is zero rows rather than a failure. WhoAmI sits
    # at address 0, is in the map, and has no file in this dataset.
    mod, _name, root, _specs = dataset
    reader = DatasetReader(mod, root)

    df = reader.read(0)
    buf = bytes(WhoAmI.format_bulk(_records(WhoAmI, 2, seed=0), timestamps=_timestamps(2)))
    populated = parse_to_dataframe(WhoAmI, buf)

    assert len(df) == 0
    assert "WhoAmI" not in reader.contents
    assert list(df.columns) == list(populated.columns)
    assert df.dtypes.equals(populated.dtypes)
    assert df.index.name == "Time"


def test_unknown_suffix_raises_file_not_found(dataset):
    # Naming a chunk that is absent is a mistake about the request, so it still raises.
    mod, _name, root, specs = dataset
    reader = DatasetReader(mod, root)
    with pytest.raises(FileNotFoundError, match="chunk"):
        reader.read(next(iter(specs)), suffix="20991231T000000Z")


def test_unknown_address_raises(dataset):
    mod, _name, root, _specs = dataset
    reader = DatasetReader(mod, root)
    with pytest.raises(KeyError):
        reader.read(9999)


def test_custom_file_resolver_supports_alternative_layout(emitted_module, tmp_path):
    mod = emitted_module
    addresses = [a for a in sorted(mod.REGISTER_MAP) if a >= 32][:2]
    expected = {}
    for address in addresses:
        cls = mod.REGISTER_MAP[address]
        buf = bytes(cls.format_bulk(_records(cls, 3, seed=address), timestamps=_timestamps(3)))
        (tmp_path / f"reg{address}.bin").write_bytes(buf)  # not the Harp layout
        expected[cls.__name__] = parse_to_dataframe(cls, buf)

    def resolver(root, _name):
        found = {}
        for path in sorted(root.glob("reg*.bin")):
            match = re.match(r"^reg(\d+)$", path.stem)
            if match is not None:
                found.setdefault(int(match.group(1)), []).append(path)
        return found

    reader = DatasetReader(mod, tmp_path, resolver=resolver)
    assert set(reader.paths) == set(addresses)
    for address in addresses:
        assert reader.read(address).equals(expected[mod.REGISTER_MAP[address].__name__])


def test_paths_property_lists_discovered_bins(dataset):
    mod, _name, root, specs = dataset
    reader = DatasetReader(mod, root)
    assert set(reader.paths) == set(specs)


def test_contents_maps_names_to_addresses(dataset):
    mod, _name, root, specs = dataset
    reader = DatasetReader(mod, root)
    # The declared address space is much larger; contents is what this folder holds.
    assert reader.contents == {cls.__name__: address for address, (cls, _buf) in specs.items()}
    assert len(mod.REGISTER_MAP) > len(reader.contents)


def test_contents_sorted_by_address(dataset):
    mod, name, root, _specs = dataset
    cls = mod.REGISTER_MAP[0]
    (root / f"{name}_0.bin").write_bytes(bytes(cls.format_bulk(_records(cls, 2, seed=0))))

    reader = DatasetReader(mod, root)

    assert list(reader.contents.values()) == sorted(reader.contents.values())
    assert next(iter(reader.contents)) == "WhoAmI"


def test_contents_keys_read_every_register(dataset):
    # The comprehension over contents is what replaces a load-everything call, so its
    # keys must cover every register with data and produce the same frames as a direct read.
    mod, _name, root, specs = dataset
    reader = DatasetReader(mod, root)

    frames = {name: reader.read(name) for name in reader.contents}

    assert set(frames) == {cls.__name__ for cls, _buf in specs.values()}
    for cls, buf in specs.values():
        assert frames[cls.__name__].equals(parse_to_dataframe(cls, buf))


def test_contents_excludes_undescribed_address(dataset):
    # A file at an address not described by the register map cannot be named, but it
    # stays visible in paths.
    mod, name, root, specs = dataset
    undescribed = max(mod.REGISTER_MAP) + 1
    (root / f"{name}_{undescribed}.bin").write_bytes(b"")

    reader = DatasetReader(mod, root)

    assert undescribed in reader.paths
    assert undescribed not in reader.contents.values()
    assert set(reader.contents) == {cls.__name__ for cls, _buf in specs.values()}


def test_every_register_round_trips(emitted_module, tmp_path):
    # Write one .bin per register of the device.yml device, then read them all back.
    mod = emitted_module
    name = mod.DEVICE_NAME
    expected = {}
    for address, cls in mod.REGISTER_MAP.items():
        records = _records(cls, 4, seed=address)
        buf = bytes(cls.format_bulk(records, timestamps=_timestamps(4)))
        (tmp_path / f"{name}_{address}.bin").write_bytes(buf)
        expected[cls.__name__] = parse_to_dataframe(cls, buf)

    reader = DatasetReader(mod, tmp_path)

    assert set(reader.paths) == set(mod.REGISTER_MAP)
    for address, cls in mod.REGISTER_MAP.items():
        df = reader.read(address)
        assert len(df) == 4
        assert df.equals(expected[cls.__name__])


def test_reader_derives_name_and_contents_from_module(dataset):
    mod, name, root, specs = dataset
    reader = DatasetReader(mod, root)
    assert reader.device_module is mod
    assert reader.name == name
    assert set(reader.contents.values()) == set(specs)


def test_open_dataset_builds_module_from_device_yml(dataset, device_yml):
    mod, _name, root, specs = dataset
    (root / "device.yml").write_text(device_yml)
    # require_converters=False mirrors the emitted_module fixture, which does not
    # inject the custom DataConverter either.
    reader = open_dataset(root, require_converters=False)
    assert isinstance(reader, DatasetReader)
    # Reads match a reader built from an explicitly-generated module.
    reference = DatasetReader(mod, root)
    for address, (cls, _buf) in specs.items():
        assert reader.read(address).equals(reference.read(cls))


def test_open_dataset_accepts_explicit_schema_path(dataset, device_yml, tmp_path):
    _mod, _name, root, specs = dataset
    schema_path = tmp_path / "elsewhere.yml"  # not inside the dataset folder
    schema_path.write_text(device_yml)
    reader = open_dataset(root, schema=schema_path, require_converters=False)
    address = next(iter(specs))
    assert not reader.read(address).empty


def test_open_dataset_accepts_device_module(dataset):
    # The overload taking a module must produce the same reader as constructing one,
    # since it is the only route open to a pre-generated package here.
    mod, _name, root, specs = dataset
    reader = open_dataset(root, mod)
    reference = DatasetReader(mod, root)
    assert reader.device_module is mod
    assert reader.name == reference.name
    for address in specs:
        assert reader.read(address).equals(reference.read(address))


def test_open_dataset_rejects_schema_beside_module(dataset, device_yml, tmp_path):
    # Both describe how to build a module, so accepting them together would ignore one.
    mod, _name, root, _specs = dataset
    schema_path = tmp_path / "elsewhere.yml"
    schema_path.write_text(device_yml)
    with pytest.raises(TypeError, match="device module"):
        open_dataset(root, mod, schema=schema_path)
    with pytest.raises(TypeError, match="device module"):
        open_dataset(root, mod, converters={})


def _with_whoami(device_yml: str, who_am_i: int) -> str:
    return f"whoAmI: {who_am_i}\n{device_yml}"


def test_whoami_mismatch_raises_on_construction(dataset, device_yml, tmp_path):
    # A folder whose schema declares a different device is rejected.
    _mod, name, root, specs = dataset
    (root / "device.yml").write_text(_with_whoami(device_yml, 1216))
    other = create_device_module(_with_whoami(device_yml, 1216), require_converters=False)

    elsewhere = tmp_path / "other_session"
    elsewhere.mkdir()
    for address in specs:
        (elsewhere / f"{name}_{address}.bin").write_bytes(b"")
    (elsewhere / "device.yml").write_text(_with_whoami(device_yml, 1234))

    with pytest.raises(ValueError, match="WhoAmI mismatch"):
        DatasetReader(other, elsewhere)


def test_matching_whoami_does_not_block_read(dataset, device_yml):
    _mod, _name, root, specs = dataset
    (root / "device.yml").write_text(_with_whoami(device_yml, 1216))
    mod = create_device_module(_with_whoami(device_yml, 1216), require_converters=False)
    assert not DatasetReader(mod, root).read(next(iter(specs))).empty


def test_unregistered_module_skips_whoami_check(dataset, device_yml):
    # WHO_AM_I of 0 marks an unregistered device, so there is nothing to check against.
    mod, _name, root, _specs = dataset
    (root / "device.yml").write_text(_with_whoami(device_yml, 1234))
    assert mod.WHO_AM_I == 0
    assert isinstance(DatasetReader(mod, root), DatasetReader)


def test_folder_without_schema_skips_whoami_check(dataset, device_yml):
    _mod, _name, root, _specs = dataset  # no device.yml written into the folder
    mod = create_device_module(_with_whoami(device_yml, 1216), require_converters=False)
    assert isinstance(DatasetReader(mod, root), DatasetReader)


def test_schema_without_whoami_skips_check(dataset, device_yml):
    _mod, _name, root, _specs = dataset
    (root / "device.yml").write_text(device_yml)  # the fixture declares no whoAmI
    mod = create_device_module(_with_whoami(device_yml, 1216), require_converters=False)
    assert isinstance(DatasetReader(mod, root), DatasetReader)


def test_unmodellable_schema_skips_check(dataset, device_yml):
    # Well-formed YAML that this library cannot describe, such as a newer or older revision,
    # must not stop a module that works from decoding the binaries beside it.
    _mod, _name, root, specs = dataset
    (root / "device.yml").write_text("registers: [this is not a register map]\n")
    mod = create_device_module(_with_whoami(device_yml, 1216), require_converters=False)
    assert not DatasetReader(mod, root).read(next(iter(specs))).empty


def test_validate_false_reads_corrupt_schema(dataset, device_yml):
    # The escape hatch: a damaged sidecar must not make a folder unreadable when the
    # module decoding it came from elsewhere.
    _mod, _name, root, specs = dataset
    (root / "device.yml").write_text("registers: {unbalanced\n")
    mod = create_device_module(_with_whoami(device_yml, 1216), require_converters=False)
    reader = DatasetReader(mod, root, validate=False)
    assert not reader.read(next(iter(specs))).empty


def test_validate_false_skips_mismatch(dataset, device_yml, tmp_path):
    _mod, name, root, specs = dataset
    (root / "device.yml").write_text(_with_whoami(device_yml, 1234))
    mod = create_device_module(_with_whoami(device_yml, 1216), require_converters=False)
    assert DatasetReader(mod, root, validate=False).name == name


def test_corrupt_schema_is_not_skipped(dataset, device_yml):
    # A schema that is not well-formed is a broken dataset rather than one this
    # version cannot describe, so it surfaces instead of being skipped.
    _mod, _name, root, _specs = dataset
    (root / "device.yml").write_text("registers: {unbalanced\n")
    mod = create_device_module(_with_whoami(device_yml, 1216), require_converters=False)
    with pytest.raises(Exception) as excinfo:
        DatasetReader(mod, root)
    # Pinning the property rather than the parser: anything deriving from ValueError
    # would have been swallowed by the skip, so this must not.
    assert not isinstance(excinfo.value, ValueError)


def test_open_dataset_missing_schema_raises_file_not_found(dataset):
    _mod, _name, root, _specs = dataset  # no device.yml written into the folder
    with pytest.raises(FileNotFoundError, match="device.yml"):
        open_dataset(root)


def test_external_schema_mismatch_raises_on_construction(dataset, device_yml, tmp_path):
    # Building the module from a schema outside the folder leaves the two free to
    # disagree, so the check still runs. Omitting schema= builds it from the folder
    # itself, where they agree by construction and the check is skipped.
    _mod, _name, root, _specs = dataset
    (root / "device.yml").write_text(_with_whoami(device_yml, 1234))
    schema_path = tmp_path / "elsewhere.yml"
    schema_path.write_text(_with_whoami(device_yml, 1216))

    with pytest.raises(ValueError, match="WhoAmI mismatch"):
        open_dataset(root, schema=schema_path, require_converters=False)
