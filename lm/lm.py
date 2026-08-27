#!/usr/bin/env python3
"""Thin entry point — the implementation lives in the mlm/ package.

    python lm.py --selftest          # assert the architecture invariants
    python lm.py --smoke             # tiny synthetic end-to-end run
    python lm.py --steps 20000 --docs 200000

`python -m mlm ...` is the same thing. See mlm/__init__.py (or README.md) for
the full usage story, including --tokenize-to/--data-dir and Kaggle recipes.
"""

from mlm.cli import main

if __name__ == "__main__":
    main()
