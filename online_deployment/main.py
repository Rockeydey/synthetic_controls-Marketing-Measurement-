import functions_framework
import os

@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function that serves an HTML file."""
    
    # Read the HTML file
    try:
        with open('index.html', 'r') as f:
            html_content = f.read()
        return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}
    except FileNotFoundError:
        return "HTML file not found", 404

    # Alternatively, handle different routes
    if request.path == '/':
        with open('index.html', 'r') as f:
            html_content = f.read()
        return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}
    else:
        return "404 Not Found", 404