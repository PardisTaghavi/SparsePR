"""Run Wan2.2-I2V-A14B through the unified SparsePR CLI."""

import sys

from sparsepr.cli import main


if __name__ == "__main__":
    main(("--model", "wan2.2-i2v-a14b", *sys.argv[1:]))
