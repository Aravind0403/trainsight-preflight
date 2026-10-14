import os
import re
import sys
import tomllib
from pathlib import Path

def main() -> None:
    tag = os.environ.get("RELEASE_TAG", "").strip()
    if not tag:
        print("ERROR: RELEASE_TAG environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    clean_tag = tag.lstrip("v")
    repo_root = Path(__file__).resolve().parent.parent.parent

    # Read pyproject.toml
    pyproject_path = repo_root / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproj_data = tomllib.load(f)
        pyproj_version = pyproj_data.get("project", {}).get("version")

    # Read src/trainsight/__init__.py
    init_path = repo_root / "src" / "trainsight" / "__init__.py"
    init_text = init_path.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init_text)
    init_version = match.group(1) if match else None

    print(f"Validating release tag: '{tag}' (normalized: '{clean_tag}')")
    print(f"  pyproject.toml version: {pyproj_version}")
    print(f"  __init__.py version:    {init_version}")

    errors = []
    if clean_tag != pyproj_version:
        errors.append(f"Tag '{clean_tag}' does not match pyproject.toml ('{pyproj_version}')")
    if clean_tag != init_version:
        errors.append(f"Tag '{clean_tag}' does not match __init__.py ('{init_version}')")

    if errors:
        print("\nERROR: Release version mismatch detected:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        print("\nPlease update the package version before publishing the release.", file=sys.stderr)
        sys.exit(1)

    print("✓ Release tag matches declared package version across all files.")

if __name__ == "__main__":
    main()
