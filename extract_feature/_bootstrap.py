from pathlib import Path
import sys


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_project_paths() -> Path:
    root = project_root()
    lib_dir = root / "lib"
    root_s = str(root)
    lib_s = str(lib_dir)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    if lib_s not in sys.path:
        sys.path.insert(0, lib_s)
    return root


def import_feature_lib():
    ensure_project_paths()
    import battery_archive_feature_lib as lib

    return lib
