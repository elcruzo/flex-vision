"""Exercise archive export through a real local HTTP receiver."""
import hashlib
import http.server
import io
import json
from pathlib import Path
import sys
import tarfile
import threading
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from export_experiment import bundle, upload


def test_bundle_upload_and_retrieval(tmp_path):
    evidence = tmp_path / 'results'
    evidence.mkdir()
    (evidence / 'result.json').write_text('{"status":"failed"}')
    archive = tmp_path / 'bundle.tar.gz'
    bundle(evidence, archive)
    received = []
    class Receiver(http.server.BaseHTTPRequestHandler):
        def do_PUT(self):
            received.append(self.rfile.read(int(self.headers['Content-Length'])))
            self.send_response(200)
            self.end_headers()
        def log_message(self, *args):
            pass
    server = http.server.HTTPServer(('127.0.0.1', 0), Receiver)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        assert upload(f'http://127.0.0.1:{server.server_port}/object?secret=value', archive, allow_local_http=True) == 200
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    with tarfile.open(fileobj=io.BytesIO(received[0]), mode='r:gz') as restored:
        manifest = json.load(restored.extractfile('manifest.json'))
        for item in manifest['files']:
            data = restored.extractfile('evidence/' + item['path']).read()
            assert hashlib.sha256(data).hexdigest() == item['sha256']
            assert len(data) == item['bytes']


def test_link_and_empty_evidence_rejected(tmp_path):
    root = tmp_path / 'results'; root.mkdir()
    with pytest.raises(ValueError, match='empty'):
        bundle(root, tmp_path / 'archive')
    (root / 'link').symlink_to('/etc/passwd')
    with pytest.raises(ValueError, match='symbolic'):
        bundle(root, tmp_path / 'archive')


@pytest.mark.parametrize('url', ['http://example.com/object', 'https://user:secret@example.com/object', 'https://example.com/object#fragment'])
def test_invalid_urls_rejected_without_transport(tmp_path, url):
    with pytest.raises(ValueError, match='HTTPS'):
        upload(url, tmp_path / 'missing')
