#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backward-compatible entry point for chunk creation.

The original project was run with `python main.py`. In the cleaned artifact,
the implementation lives in `src/preprocessing/create_chunks.py`.
"""

from src.preprocessing.create_chunks import main

if __name__ == "__main__":
    main()
