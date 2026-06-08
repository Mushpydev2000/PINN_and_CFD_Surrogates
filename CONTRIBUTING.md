# Contributing to PINN-CFD Surrogate

First off, thank you for considering contributing to this research project!

## Development Setup

1. Fork and clone the repository.
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install the dependencies including Django:
   ```bash
   pip install -r requirements.txt
   ```
4. Run migrations:
   ```bash
   python manage.py migrate
   ```

## Code Style

This project uses `black` for code formatting and `isort` for import sorting.
Before submitting a pull request, please ensure your code is formatted:
```bash
make format
```

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.
3. You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you.
