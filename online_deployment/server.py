from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

PORT = int(os.environ.get('PORT', 8080))

class MyHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()

httpd = HTTPServer(('0.0.0.0', PORT), MyHTTPRequestHandler)
print(f"Server running on port {PORT}")
httpd.serve_forever()