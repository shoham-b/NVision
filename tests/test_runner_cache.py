import logging
from pathlib import Path

from nvision.runner.cache import _decompress_content, embed_graph_content


def test_embed_graph_content_happy_path(tmp_path: Path):
    """Test that existing files are read and compressed into the content key."""
    # Setup
    test_file = tmp_path / "test_graph.txt"
    test_content = "This is some test graph content."
    test_file.write_text(test_content, encoding="utf-8")

    entries = [{"path": "test_graph.txt", "other": "data"}]

    # Execute
    result = embed_graph_content(entries, tmp_path)

    # Verify
    assert len(result) == 1
    assert "content" in result[0]
    assert result[0]["path"] == "test_graph.txt"
    assert result[0]["other"] == "data"

    # Decompress to verify content
    decompressed = _decompress_content(result[0])
    assert decompressed == test_content

    # Original should be untouched
    assert "content" not in entries[0]


def test_embed_graph_content_missing_file(tmp_path: Path):
    """Test that missing files are gracefully skipped without error."""
    # Setup
    entries = [{"path": "missing_file.txt"}]

    # Execute
    result = embed_graph_content(entries, tmp_path)

    # Verify
    assert len(result) == 1
    assert "content" not in result[0]
    assert result[0]["path"] == "missing_file.txt"


def test_embed_graph_content_no_path_key(tmp_path: Path):
    """Test that entries without a path key are skipped."""
    # Setup
    entries = [{"no_path": "here"}]

    # Execute
    result = embed_graph_content(entries, tmp_path)

    # Verify
    assert len(result) == 1
    assert "content" not in result[0]
    assert result[0]["no_path"] == "here"


def test_embed_graph_content_read_exception(tmp_path: Path, caplog, monkeypatch):
    """Test that file read exceptions are caught and logged as warnings."""
    # Setup
    test_file = tmp_path / "error_file.txt"
    test_file.write_text("content", encoding="utf-8")

    entries = [{"path": "error_file.txt"}]

    # Force read_text to raise an exception
    def mock_read_text(*args, **kwargs):
        raise PermissionError("Access denied")

    monkeypatch.setattr(Path, "read_text", mock_read_text)

    # Execute
    with caplog.at_level(logging.WARNING):
        result = embed_graph_content(entries, tmp_path)

    # Verify
    assert len(result) == 1
    assert "content" not in result[0]

    # Check logs
    assert "Failed to read graph content for caching" in caplog.text
    assert "Access denied" in caplog.text
