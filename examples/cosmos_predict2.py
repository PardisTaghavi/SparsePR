"""Run Cosmos-Predict2.5-14B through the unified SparsePR CLI."""

import sys

from sparsepr.cli import main


if __name__ == "__main__":
    main(("--model", "cosmos-predict2.5-14b", *sys.argv[1:]))
