#!/usr/bin/env python3
"""Train only Telugu with the language-local assignment runner and an isolated run ID."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'Telugu'))
from setup.train_assignment import main

if __name__ == '__main__':
    main(language='telugu')
