"""Preprocessing package.

Public objects are imported from their defining modules. Keeping this package
initializer lightweight preserves the V2 Data -> Preprocessing dependency
direction and avoids loading the high-level pipeline during low-level imports.
"""

__all__: list[str] = []
