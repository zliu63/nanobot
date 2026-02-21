"""
nanobot - A lightweight AI agent framework
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("nanobot-ai")
except PackageNotFoundError:
    __version__ = "0.1.0"

__logo__ = "🐈"
