import os
import webbrowser
import threading


def run_local_site(port: int = 8000):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'webapp.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Django is required to run the web dashboard. Install it with `pip install django`."
        ) from exc

    url = f"http://127.0.0.1:{port}"
    threading.Timer(1, webbrowser.open, args=[url]).start()
    execute_from_command_line(['manage.py', 'runserver', f'127.0.0.1:{port}'])
