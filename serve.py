"""Local server: python serve.py [port]. ES modules and WebGPU need http://localhost.
POST /dump/<name> is only used by pipeline/dump.html, to save G-buffers for training."""
import http.server, os, re, socketserver, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(ROOT, "pipeline", "out", "gbuf")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5191


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        m = re.fullmatch(r"/dump/([\w.-]+)", self.path)
        if not m:
            self.send_error(404)
            return
        os.makedirs(DUMP, exist_ok=True)
        n = int(self.headers["Content-Length"])
        with open(os.path.join(DUMP, m.group(1)), "wb") as f:
            while n > 0:
                b = self.rfile.read(min(n, 1 << 20))
                f.write(b)
                n -= len(b)
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a):
        pass


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


print(f"http://127.0.0.1:{PORT}/", flush=True)
S(("127.0.0.1", PORT), H).serve_forever()
