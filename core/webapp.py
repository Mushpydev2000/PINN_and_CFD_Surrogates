import http.server
import webbrowser
import threading

def run_local_site(port: int = 8000):
    handler = http.server.SimpleHTTPRequestHandler
    with http.server.HTTPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}"
        threading.Timer(1, webbrowser.open, args=[url]).start()
        print(f"Serving at {url}  (Ctrl+C to stop)")
        httpd.serve_forever()
