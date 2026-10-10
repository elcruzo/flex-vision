"""Archive completed experiment evidence and upload through a scoped PUT URL."""
import argparse
import hashlib
import http.client
import io
import json
from pathlib import Path
import tarfile
import tempfile
from urllib.parse import urlsplit

MAX_BYTES = 512 * 1024 * 1024


def bundle(root, destination):
    root = Path(root)
    if Path(destination).resolve().is_relative_to(root.resolve()):
        raise ValueError('Archive destination must be outside the evidence directory')
    entries = []
    total = 0
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Evidence contains a symbolic link')
        if path.is_file():
            total += path.stat().st_size
            if total > MAX_BYTES:
                raise ValueError('Evidence exceeds the export size limit')
            entries.append(path)
    if not entries:
        raise ValueError('Evidence directory is empty')
    manifest = []
    snapshot_bytes = 0
    with tarfile.open(destination, 'w:gz') as archive:
        for path in entries:
            # Hash and archive the same snapshot, not two reads of a mutable file.
            with path.open('rb') as source:
                data = source.read(MAX_BYTES - snapshot_bytes + 1)
            snapshot_bytes += len(data)
            if snapshot_bytes > MAX_BYTES:
                raise ValueError('Evidence exceeds the export size limit')
            name = path.relative_to(root).as_posix()
            manifest.append({'path': name, 'bytes': len(data),
                             'sha256': hashlib.sha256(data).hexdigest()})
            info = tarfile.TarInfo('evidence/' + name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        data = json.dumps({'files': manifest}, indent=2).encode()
        info = tarfile.TarInfo('manifest.json')
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return manifest


def upload(url, archive, *, allow_local_http=False):
    parts = urlsplit(url)
    local = allow_local_http and parts.scheme == 'http' and parts.hostname in ('127.0.0.1', 'localhost')
    if (parts.scheme != 'https' and not local) or not parts.hostname or parts.username or parts.password or parts.fragment:
        raise ValueError('Require an HTTPS signed PUT URL')
    connection_type = http.client.HTTPConnection if local else http.client.HTTPSConnection
    connection = connection_type(parts.hostname, parts.port, timeout=120)
    try:
        target = parts.path or '/'
        if parts.query:
            target += '?' + parts.query
        connection.putrequest('PUT', target)
        connection.putheader('Content-Length', str(archive.stat().st_size))
        connection.putheader('Content-Type', 'application/gzip')
        connection.endheaders()
        with archive.open('rb') as source:
            while chunk := source.read(1024 * 1024):
                connection.send(chunk)
        response = connection.getresponse()
        if not 200 <= response.status < 300:
            raise RuntimeError('Evidence upload rejected with HTTP ' + str(response.status))
        # Do not interpret ETag as a SHA256 checksum. Retrieval must verify the manifest.
        return response.status
    except (OSError, http.client.HTTPException) as error:
        raise RuntimeError('Evidence upload transport failed') from None
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / 'evidence.tar.gz'
            manifest = bundle(args.root, archive)
            status = upload(config['put_url'], archive)
            receipt = {'status': 'uploaded', 'http_status': status, 'files': len(manifest),
                       'archive_bytes': archive.stat().st_size,
                       'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
            (args.root / 'durable-upload.json').write_text(json.dumps(receipt, indent=2) + '\n')
    finally:
        args.config.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
