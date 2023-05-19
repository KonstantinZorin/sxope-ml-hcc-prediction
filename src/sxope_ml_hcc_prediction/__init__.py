from importlib import metadata

__version__: str = metadata.version(__package__)

del metadata
