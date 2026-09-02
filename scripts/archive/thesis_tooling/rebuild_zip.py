"""
Rebuild the thesis submission ZIP from the LaTeX source directory.
Includes all source files needed to recompile, plus the final PDF.
"""

import zipfile
import os
from pathlib import Path

# Resolved from this file's location rather than hardcoded, so the script runs from any
# checkout. They previously carried an absolute path from one developer machine.
REPO_ROOT = Path(__file__).resolve().parents[3]
THESIS_DIR = REPO_ROOT / "thesis" / "latex_tum_official"
OUTPUT_ZIP = REPO_ROOT / "thesis_upload.zip"

# Extensions / files to include
INCLUDE_EXTENSIONS = {
    ".tex",
    ".bib",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".cls",
    ".sty",
    ".bst",
    ".xmpdata",
    ".xmpi",
}
INCLUDE_FILES = {".latexmkrc", "MANIFEST.md"}
# Directories to skip entirely
SKIP_DIRS = {"__pycache__", ".git"}


def should_include(path: Path, rel: str) -> bool:
    """Decide whether a file belongs in the submission ZIP."""
    name = path.name
    suffix = path.suffix.lower()
    # Skip build artifacts
    if suffix in {
        ".aux",
        ".bbl",
        ".bcf",
        ".blg",
        ".fdb_latexmk",
        ".fls",
        ".log",
        ".loa",
        ".lof",
        ".lot",
        ".toc",
        ".run.xml",
        ".out",
        ".synctex.gz",
        ".nav",
        ".snm",
        ".vrb",
    }:
        return False
    # Skip Python scripts inside the figures/ directory (not needed for compilation)
    if suffix == ".py":
        return False
    # Include by extension
    if suffix in INCLUDE_EXTENSIONS:
        return True
    # Include specific filenames
    if name in INCLUDE_FILES:
        return True
    return False


def main():
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
        print(f"Removed old {OUTPUT_ZIP.name}")

    count = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(THESIS_DIR):
            # Prune skipped directories
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in sorted(files):
                fpath = Path(root) / fname
                rel = fpath.relative_to(THESIS_DIR)
                if should_include(fpath, str(rel)):
                    arcname = str(rel)
                    zf.write(fpath, arcname)
                    count += 1
                    print(f"  + {arcname}")

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\nCreated {OUTPUT_ZIP.name}: {count} files, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
