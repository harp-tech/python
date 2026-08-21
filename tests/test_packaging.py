import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ("harp-protocol", "harp-device", "harp-serial", "harp-data")


def _manifest(package: str) -> dict:
    path = ROOT / "src" / "packages" / package / "pyproject.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]


@pytest.mark.parametrize("package", PUBLISHED)
def test_declared_license_files_match_repository_license(package: str):
    # PEP 639 forbids a parent directory reference in license-files, so each package
    # keeps its own copy. Nothing stops the copies drifting except this.
    expected = (ROOT / "LICENSE").read_bytes()
    declared = _manifest(package)["license-files"]
    assert declared, f"{package} declares no license file"
    for name in declared:
        assert (ROOT / "src" / "packages" / package / name).read_bytes() == expected


@pytest.mark.parametrize("package", PUBLISHED)
def test_published_package_declares_metadata(package: str):
    # A published package with no readme or license renders as a blank project page.
    project = _manifest(package)
    for key in ("readme", "license", "license-files", "classifiers", "urls", "authors"):
        assert key in project, f"{package} declares no {key}"
