"""Single-command local runner for the TopoCollapse interactive web dashboard."""

import http.server
import os
import socketserver
import webbrowser

PORT = 8000
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)


def main():
    os.chdir(DASHBOARD_DIR)
    url = f"http://localhost:{PORT}/index.html"
    print("=" * 70)
    print(" TOPOCOLLAPSE INTERACTIVE DASHBOARD SERVER")
    print(f" Serving dashboard at: {url}")
    print(" Press Ctrl+C to terminate the server.")
    print("=" * 70)

    # Launch browser automatically
    webbrowser.open(url)

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            httpd.server_close()


if __name__ == "__main__":
    main()
