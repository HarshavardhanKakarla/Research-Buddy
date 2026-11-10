"""
ingest_cli.py
-------------
Batch-ingest all PDFs/text files in a directory without opening the UI.

Usage:
    python scripts/ingest_cli.py --dir ./data/uploads
"""

import argparse
import sys

sys.path.append(".")  # allow running from project root

from src.config import get_settings
from src.ingestion import ingest_directory


def main():
    parser = argparse.ArgumentParser(description="Batch ingest documents into Pinecone.")
    parser.add_argument("--dir", required=True, help="Directory containing PDFs/.txt/.md files")
    args = parser.parse_args()

    settings = get_settings(require_openai=True)
    results = ingest_directory(args.dir, settings)

    for r in results:
        print(f"{r['source']}: {r['status']} ({r['chunks']} chunks)")

    if not results:
        print("No supported files (.pdf, .txt, .md) found in that directory.")


if __name__ == "__main__":
    main()
