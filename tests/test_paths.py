from __future__ import annotations

from pathlib import Path

from nvision.tools.paths import ensure_out_dir, find_project_root, slugify


def test_ensure_out_dir(tmp_path: Path) -> None:
    # Test single directory
    dir1 = tmp_path / "test_dir_1"
    assert not dir1.exists()
    ensure_out_dir(dir1)
    assert dir1.is_dir()

    # Test nested directory
    dir2 = tmp_path / "nested" / "test_dir_2"
    assert not dir2.exists()
    ensure_out_dir(dir2)
    assert dir2.is_dir()

    # Test existing directory
    ensure_out_dir(dir2)
    assert dir2.is_dir()


def test_slugify() -> None:
    # Test basic string
    assert slugify("Hello World") == "hello-world"

    # Test string with special characters
    assert slugify("Test! @# String$%^") == "test-string"

    # Test leading/trailing hyphens and spaces
    assert slugify("  --hello-world--  ") == "hello-world"
    assert slugify("___hello_world___") == "hello-world"

    # Test non-string input
    assert slugify(12345) == "12345"
    assert slugify(12.34) == "12-34"


def test_find_project_root_with_git(tmp_path: Path, monkeypatch) -> None:
    # Create directory structure
    project_root = tmp_path / "project"
    project_root.mkdir()

    # Create .git dir
    git_dir = project_root / ".git"
    git_dir.mkdir()

    # Create file deep inside the structure
    file_path = project_root / "src" / "package" / "module.py"
    file_path.parent.mkdir(parents=True)

    # Patch __file__
    monkeypatch.setattr("nvision.tools.paths.__file__", str(file_path))

    assert find_project_root() == project_root


def test_find_project_root_with_pyproject(tmp_path: Path, monkeypatch) -> None:
    # Create directory structure
    project_root = tmp_path / "project"
    project_root.mkdir()

    # Create pyproject.toml
    pyproject = project_root / "pyproject.toml"
    pyproject.touch()

    # Create file deep inside the structure
    file_path = project_root / "src" / "package" / "module.py"
    file_path.parent.mkdir(parents=True)

    # Patch __file__
    monkeypatch.setattr("nvision.tools.paths.__file__", str(file_path))

    assert find_project_root() == project_root


def test_find_project_root_with_nvision(tmp_path: Path, monkeypatch) -> None:
    # Create directory structure
    project_root = tmp_path / "project"
    project_root.mkdir()

    # Create nvision dir
    nvision_dir = project_root / "nvision"
    nvision_dir.mkdir()

    # Create file deep inside the structure
    file_path = project_root / "src" / "package" / "module.py"
    file_path.parent.mkdir(parents=True)

    # Patch __file__
    monkeypatch.setattr("nvision.tools.paths.__file__", str(file_path))

    assert find_project_root() == project_root


def test_find_project_root_fallback(tmp_path: Path, monkeypatch) -> None:
    # Create directory structure with none of the target markers
    file_path = tmp_path / "src" / "package" / "module.py"
    file_path.parent.mkdir(parents=True)

    # Patch __file__
    monkeypatch.setattr("nvision.tools.paths.__file__", str(file_path))

    assert find_project_root() == Path.cwd()
import pytest

from nvision.tools.paths import slugify


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        ("normal string", "normal-string"),
        ("  extra   spaces  ", "extra-spaces"),
        ("punctuation!@#$%", "punctuation"),
        ("special chars &*()", "special-chars"),
        ("", ""),
        ("!@#$%", ""),
        ("12345", "12345"),
        ("leading and trailing-", "leading-and-trailing"),
        ("-leading and trailing", "leading-and-trailing"),
        ("mixed 123 and symbols !@", "mixed-123-and-symbols"),
        (12345, "12345"),  # Non-string input
        (None, "none"),  # Non-string input
    ],
)
def test_slugify(input_value, expected):
    """Test slugify function with various edge cases."""
    assert slugify(input_value) == expected


def test_slugify_lru_cache():
    """Test that slugify function is cached using lru_cache."""
    # Ensure cache is empty or in a known state
    slugify.cache_clear()

    initial_info = slugify.cache_info()

    # Call once
    result1 = slugify("test string")
    info_after_first_call = slugify.cache_info()

    # Call again with same input
    result2 = slugify("test string")
    info_after_second_call = slugify.cache_info()

    assert result1 == "test-string"
    assert result2 == "test-string"

    # Check cache hits/misses
    assert info_after_first_call.misses == initial_info.misses + 1
    assert info_after_first_call.hits == initial_info.hits

    assert info_after_second_call.misses == info_after_first_call.misses
    assert info_after_second_call.hits == info_after_first_call.hits + 1
