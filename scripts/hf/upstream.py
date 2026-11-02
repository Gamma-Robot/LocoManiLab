"""Fetch individual simulator asset files from their upstream hosts.

Used by ``download.py assets``. Each manifest row names one file and where its bytes
come from; only the listed files are transferred, never whole asset libraries:

    zip:<url>:<member>                 one member of a remote zip (HTTP range requests)
    molmospaces:<source>:<package>:<member>
                                       one member of a MolmoSpaces package; the package
                                       is located through the source's ``pkgs`` index
    file:<url>                         a plain file
    hf:<shard>                         a LocoManiLab asset shard on the Hub (handled by
                                       download.py)

Every file is checked against the manifest's sha256 before it is moved into place.
"""
from __future__ import annotations

import hashlib
import io
import struct
import tarfile
import time
import urllib.request
import zlib
from pathlib import Path

# Pinned MolmoSpaces revision; manifest rows were hashed against it.
MOLMOSPACES_REVISION = "5f802a46816ffc3520e59b5ef49fa8dafee8eeca"
MOLMOSPACES = f"https://huggingface.co/datasets/allenai/molmospaces/resolve/{MOLMOSPACES_REVISION}"
USER_AGENT = "locomanilab-asset-fetch/1"


def _get(url: str, start: int | None = None, end: int | None = None, retries: int = 5) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as resp:
                data = resp.read()
            if start is not None and resp.status != 206:
                raise OSError(f"server ignored the range request for {url}")
            if start is not None and len(data) != end - start + 1:
                raise OSError(f"short read from {url}")
            return data
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError


def fetch_url(url: str) -> bytes:
    return _get(url)


def _probe(url: str) -> tuple[int, str]:
    """Total size and the final URL after redirects (reused for range requests)."""
    headers = {"User-Agent": USER_AGENT, "Range": "bytes=0-0"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as resp:
        return int(resp.headers["Content-Range"].rsplit("/", 1)[1]), resp.geturl()


class RemoteZip:
    """Random access to members of a zip served over HTTP with range support."""

    def __init__(self, url: str):
        self.source = url
        size, self.url = _probe(url)
        self.size = size
        tail_len = min(size, 1 << 16)
        tail = _get(url, size - tail_len, size - 1)
        eocd = tail.rfind(b"PK\x05\x06")
        if eocd < 0:
            raise OSError(f"no zip directory in {url}")
        cd_size, cd_offset = struct.unpack("<II", tail[eocd + 12:eocd + 20])
        if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
            loc = tail.rfind(b"PK\x06\x07")
            (z64_offset,) = struct.unpack("<Q", tail[loc + 8:loc + 16])
            z64 = _get(url, z64_offset, z64_offset + 55)
            cd_size, cd_offset = struct.unpack("<QQ", z64[40:56])
        directory = _get(url, cd_offset, cd_offset + cd_size - 1)
        self.members: dict[str, tuple[int, int, int, int]] = {}
        pos = 0
        while pos + 46 <= len(directory) and directory[pos:pos + 4] == b"PK\x01\x02":
            (method,) = struct.unpack("<H", directory[pos + 10:pos + 12])
            csize, usize = struct.unpack("<II", directory[pos + 20:pos + 28])
            nlen, elen, clen = struct.unpack("<HHH", directory[pos + 28:pos + 34])
            (offset,) = struct.unpack("<I", directory[pos + 42:pos + 46])
            name = directory[pos + 46:pos + 46 + nlen].decode("utf-8")
            extra = directory[pos + 46 + nlen:pos + 46 + nlen + elen]
            if 0xFFFFFFFF in (csize, usize, offset):
                csize, usize, offset = _zip64_sizes(extra, csize, usize, offset)
            self.members[name] = (method, csize, usize, offset)
            pos += 46 + nlen + elen + clen

    def _range(self, start: int, end: int) -> bytes:
        end = min(end, self.size - 1)
        try:
            return _get(self.url, start, end, retries=2)
        except Exception:
            # Signed redirect targets (e.g. Box) expire; resolve the source URL again.
            self.url = _probe(self.source)[1]
            return _get(self.url, start, end)

    def read(self, name: str) -> bytes:
        method, csize, usize, offset = self.members[name]
        # One request for the local header and the data; the slack covers a local
        # extra field longer than the name estimate.
        nbytes = len(name.encode("utf-8"))
        blob = self._range(offset, offset + 30 + nbytes + 1024 + csize - 1)
        nlen, elen = struct.unpack("<HH", blob[26:30])
        start = 30 + nlen + elen
        if start + csize > len(blob):
            blob = self._range(offset, offset + start + csize - 1)
        data = blob[start:start + csize]
        if method == 0:
            return data
        if method == 8:
            return zlib.decompress(data, -15)
        raise OSError(f"unsupported zip compression {method} for {name}")


def _zip64_sizes(extra: bytes, csize: int, usize: int, offset: int) -> tuple[int, int, int]:
    pos = 0
    while pos + 4 <= len(extra):
        tag, length = struct.unpack("<HH", extra[pos:pos + 4])
        if tag == 1:
            values = list(struct.unpack(f"<{length // 8}Q", extra[pos + 4:pos + 4 + length - length % 8]))
            if usize == 0xFFFFFFFF:
                usize = values.pop(0)
            if csize == 0xFFFFFFFF:
                csize = values.pop(0)
            if offset == 0xFFFFFFFF:
                offset = values.pop(0)
            break
        pos += 4 + length
    return csize, usize, offset


class MolmoSpacesSource:
    """Packages of one MolmoSpaces data source (e.g. ``mujoco/objects/thor/20251117``)."""

    def __init__(self, source: str, index_rows: list[dict]):
        self.source = source
        self.rows = {row["path"]: row for row in index_rows}

    def members(self, package: str) -> dict[str, bytes]:
        row = self.rows[package]
        url = f"{MOLMOSPACES}/{self.source}/shards/{row['shard_id']:05d}.tar"
        blob = _get(url, row["offset"], row["offset"] + row["size"] - 1)
        import zstandard

        raw = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(blob)).read()
        out = {}
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            for member in tar.getmembers():
                if member.isfile():
                    out[member.name] = tar.extractfile(member).read()
        return out


def molmospaces_index(source: str, hub) -> list[dict]:
    """Rows (path, shard_id, offset, size) of a MolmoSpaces source's ``pkgs`` parquet."""
    import pyarrow.parquet as pq

    api = hub.HfApi()
    folder = source.replace("/", "__")
    rows: list[dict] = []
    for item in api.list_repo_tree("allenai/molmospaces", repo_type="dataset", path_in_repo=folder,
                                   revision=MOLMOSPACES_REVISION):
        if item.path.endswith(".parquet") and "pkgs" in item.path.rsplit("/", 1)[-1]:
            local = hub.hf_hub_download("allenai/molmospaces", item.path, repo_type="dataset",
                                        revision=MOLMOSPACES_REVISION)
            rows += pq.read_table(local).to_pylist()
    return rows


def write_verified(dest: Path, rel: str, data: bytes, sha256: str) -> None:
    if hashlib.sha256(data).hexdigest() != sha256:
        raise OSError(f"sha256 mismatch for {rel}")
    target = (dest / rel).resolve()
    if dest.resolve() not in target.parents:
        raise OSError(f"unsafe asset path: {rel}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(target)
