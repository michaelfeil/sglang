"""Protocol checks for the profiling client; no SGLang server or GPU needed."""

import contextlib
import gzip
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from profile_masks import request, summarize


class TestProfilerHTTPResponses(unittest.TestCase):
    def test_empty_control_response_and_json_generation_response(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.end_headers()
                if self.path == "/generate":
                    self.wfile.write(json.dumps({"text": "hello"}).encode())
                elif self.path == "/start_profile":
                    self.wfile.write(b"Start profiling.\n")

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            self.assertEqual(request(url, "/start_profile", {}), "Start profiling.\n")
            self.assertIsNone(request(url, "/stop_profile", {}))
            self.assertEqual(request(url, "/generate", {}), {"text": "hello"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class TestTraceSummary(unittest.TestCase):
    def test_cpu_and_gpu_annotations_are_not_combined(self):
        events = [
            {"ph": "X", "cat": "user_annotation", "name": "grammar.fill", "dur": 20},
            {"ph": "X", "cat": "user_annotation", "name": "grammar.apply", "dur": 50},
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "grammar.apply",
                "dur": 2,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".json", ".json.gz"):
                path = Path(directory) / f"trace{suffix}"
                opener = gzip.open if suffix.endswith("gz") else open
                with opener(path, "wt") as output:
                    json.dump({"traceEvents": events}, output)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    summarize(SimpleNamespace(traces=[path]))
                result = json.loads(stdout.getvalue())
                self.assertEqual(result["cpu_ranges_us"]["grammar.apply"]["median"], 50)
                self.assertEqual(result["gpu_ranges_us"]["grammar.apply"]["median"], 2)
                self.assertEqual(result["cpu_ranges_us"]["grammar.apply"]["count"], 1)
                self.assertEqual(
                    result["structured_output_evidence"], "grammar fill observed"
                )


if __name__ == "__main__":
    unittest.main()
