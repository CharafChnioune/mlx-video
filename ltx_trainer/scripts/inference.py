#!/usr/bin/env python3
"""MLX inference wrapper that forwards to mlx_video.generate CLI."""
from __future__ import annotations

import sys
from mlx_video.generate import main

if __name__ == "__main__":
    main()
