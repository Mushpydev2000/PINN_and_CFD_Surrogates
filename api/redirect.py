import os

from vercel import Response

def handler(request):
    render_url = os.environ.get('RENDER_APP_URL', 'https://pinn-and-cfd-surrogates.onrender.com')
    return Response(
        status=302,
        headers={
            'Location': render_url,
        },
        body=''
    )

app = handler
