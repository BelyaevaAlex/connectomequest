"""ConnectomeQuest public package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("connectomequest")
except PackageNotFoundError:  # editable source tree
    __version__ = "0.1.0"
