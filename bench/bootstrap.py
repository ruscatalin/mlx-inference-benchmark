"""Dependency check: stops with install instructions if a required package is missing.
The benchmark never installs anything itself; packages come from requirements.txt."""
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# import name -> pip name (keep in sync with requirements.txt)
DEPS = {"rich": "rich", "matplotlib": "matplotlib", "huggingface_hub": "huggingface_hub", "yaml": "pyyaml",
        "mlx": "mlx", "pyarrow": "pyarrow", "absl": "absl-py", "langdetect": "langdetect", "nltk": "nltk",
        "immutabledict": "immutabledict", "emoji": "emoji", "syllapy": "syllapy"}


def ensure_deps():
    missing = [pip for mod, pip in DEPS.items() if importlib.util.find_spec(mod) is None]
    if missing:
        sys.exit("Missing Python packages: " + ", ".join(missing) + "\nInstall them with:\n"
                 f"  {sys.executable} -m pip install -r {REPO_ROOT / 'requirements.txt'}")
