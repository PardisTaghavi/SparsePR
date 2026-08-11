"""Run Cosmos3-Nano-16B through the unified SparsePR CLI."""

import sys

from sparsepr.cli import main


if __name__ == "__main__":
    main(("--model", "cosmos3-nano-16b", *sys.argv[1:]))
