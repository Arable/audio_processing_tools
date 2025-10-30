from importlib import metadata
try:
    __version__ = metadata.version("audio-processing-tools")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0-dev"
