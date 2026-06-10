import os

from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        render_url = os.environ.get(
            "RENDER_APP_URL",
            "https://pinn-and-cfd-surrogates.onrender.com",
        )
        self.send_response(302)
        self.send_header("Location", render_url)
        self.end_headers()

    def do_HEAD(self):
        self.do_GET()
