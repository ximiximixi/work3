from __future__ import annotations

import sys

from dino_ad.cli import main


if __name__ == "__main__":
    main(["generate_report", *sys.argv[1:]])
