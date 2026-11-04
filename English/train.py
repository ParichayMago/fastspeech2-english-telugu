#!/usr/bin/env python3
"""Train only English with the language-local assignment runner and an isolated run ID."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'English'))
from setup.train_assignment import main

if __name__ == '__main__':
    main(language='english')
