"""HTTP endpoint for the packaged news classifier."""

from http.server import BaseHTTPRequestHandler
import json

from src.inference import classify


class handler(BaseHTTPRequestHandler):
    def respond(self, status: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 200_000:
                self.respond(413, {"error": "Paste an article of up to 50,000 characters."})
                return
            data = json.loads(self.rfile.read(length))
            text = data.get("text") if isinstance(data, dict) else None
            if not isinstance(text, str):
                self.respond(400, {"error": "Enter news text as a string."})
                return
            if len(text) > 50_000:
                self.respond(413, {"error": "Paste an article of up to 50,000 characters."})
                return
            self.respond(200, classify(text))
        except (ValueError, UnicodeDecodeError) as error:
            self.respond(400, {"error": str(error)})

    def do_GET(self) -> None:
        self.respond(405, {"error": "Submit news text with POST."})
