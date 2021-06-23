from .main import main

# These imports extend the capabilities of main
from . import mock_generator, launch  # noqa

__all__ = [main]
