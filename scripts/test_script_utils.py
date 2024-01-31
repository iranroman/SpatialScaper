import os
import io
import tarfile
import zipfile
import requests
import pytest
import subprocess
from unittest.mock import Mock, patch

from .utils import extract_tar
from .utils import extract_zip
from .utils import download_file
from .utils import combine_multizip


def create_test_tar(tar_path, content):
    with tarfile.open(tar_path, "w:gz") as tar:
        for name, data in content.items():
            tarinfo = tarfile.TarInfo(name=name)
            tarinfo.size = len(data)
            tar.addfile(tarinfo, io.BytesIO(data.encode()))


def test_successful_extraction_tar(tmp_path):
    tar_path = tmp_path / "test.tar.gz"
    dest_path = tmp_path / "dest"
    os.makedirs(dest_path)

    # Create a test tar file
    create_test_tar(tar_path, {"file1.txt": "content1", "file2.txt": "content2"})

    extract_tar(str(tar_path), str(dest_path))

    assert os.path.exists(dest_path / "file1.txt")
    assert os.path.exists(dest_path / "file2.txt")


def test_non_existent_tar_file(tmp_path):
    tar_path = tmp_path / "non_existent.tar.gz"
    dest_path = tmp_path / "dest"
    os.makedirs(dest_path)

    with pytest.raises(FileNotFoundError):
        extract_tar(str(tar_path), str(dest_path))


def create_test_zip(zip_path, content):
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file_name, data in content.items():
            zipf.writestr(file_name, data)


def test_successful_extraction_zip(tmp_path):
    zip_path = tmp_path / "test.zip"
    dest_path = tmp_path / "dest"
    os.makedirs(dest_path)

    # Create a test zip file
    create_test_zip(zip_path, {"file1.txt": "content1", "file2.txt": "content2"})

    extract_zip(str(zip_path), str(dest_path))

    assert os.path.exists(dest_path / "file1.txt")
    assert os.path.exists(dest_path / "file2.txt")


def test_non_existent_zip_file(tmp_path):
    zip_path = tmp_path / "non_existent.zip"
    dest_path = tmp_path / "dest"
    os.makedirs(dest_path)

    with pytest.raises(FileNotFoundError):
        extract_zip(str(zip_path), str(dest_path))


def test_invalid_zip_format(tmp_path):
    zip_path = tmp_path / "invalid.zip"
    dest_path = tmp_path / "dest"
    os.makedirs(dest_path)

    # Create an invalid zip file
    with open(zip_path, "w") as f:
        f.write("not a zip content")

    with pytest.raises(ValueError) as excinfo:
        extract_zip(str(zip_path), str(dest_path))
    assert "not a valid zip file" in str(excinfo.value)


def test_successful_download(tmp_path, monkeypatch):
    url = "http://example.com/testfile"
    local_dest_path = tmp_path / "downloaded_file"
    test_content = b"test data"

    # Mock requests.get to return a response with test content
    mock_response = Mock()
    mock_response.iter_content = lambda block_size: [test_content]
    mock_response.headers = {"content-length": str(len(test_content))}
    monkeypatch.setattr(requests, "get", lambda url, stream=None: mock_response)

    download_file(url, str(local_dest_path))

    assert os.path.exists(local_dest_path)
    with open(local_dest_path, "rb") as f:
        assert f.read() == test_content


def test_invalid_url():
    invalid_url = "http://thisurldoesnotexist123.com"
    local_dest_path = "dummy_path"

    with pytest.raises(RuntimeError):
        download_file(invalid_url, local_dest_path)


@patch("requests.get")
def test_http_error(mock_get):
    mock_get.side_effect = requests.HTTPError()
    local_dest_path = "dummy_path"

    with pytest.raises(RuntimeError):
        download_file("http://example.com/testfile", local_dest_path)


def test_file_writing_error(tmp_path, monkeypatch):
    url = "http://example.com/testfile"
    local_dest_path = tmp_path / "non_existent_directory" / "downloaded_file"

    # No need to mock requests.get since the error should occur before the request

    with pytest.raises(RuntimeError):
        download_file(url, str(local_dest_path))


@patch("subprocess.run")
def test_combine_multizip(mock_run):
    filename = "multi-part.zip"
    destination = "combined.zip"

    combine_multizip(filename, destination)

    # Check if subprocess.run was called with the correct command
    mock_run.assert_called_once_with(
        f"zip -s 0 {filename} --out {destination}", shell=True
    )


@patch("subprocess.run")
def test_combine_multizip_with_shell_false(mock_run):
    filename = "multi-part.zip"
    destination = "combined.zip"

    combine_multizip(filename, destination, shell=False)

    # Check if subprocess.run was called with the correct command and shell=False
    mock_run.assert_called_once_with(
        f"zip -s 0 {filename} --out {destination}", shell=False
    )
