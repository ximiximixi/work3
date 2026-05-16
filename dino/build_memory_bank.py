from __future__ import annotations

import sys

from dino_ad.cli import main


if __name__ == "__main__":
    main(["build_memory_bank", *sys.argv[1:]])
