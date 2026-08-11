"""Run HunyuanVideo-13B through the unified SparsePR CLI."""

import sys

from sparsepr.cli import main


if __name__ == "__main__":
    main(("--model", "hunyuanvideo-13b", *sys.argv[1:]))
