#!/usr/bin/env python3
"""Text-to-speech using the frozen Telugu checkpoint and Telugu phone inventory."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'src'))
from setup.final_inference import main

if __name__=='__main__':main('telugu')
