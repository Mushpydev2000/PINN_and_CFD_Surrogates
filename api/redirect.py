import os

def handler(request):
    render_url = os.environ.get('RENDER_APP_URL', 'https://your-render-app.onrender.com')
    return {
        'statusCode': 302,
        'headers': {
            'Location': render_url
        },
        'body': ''
    }
