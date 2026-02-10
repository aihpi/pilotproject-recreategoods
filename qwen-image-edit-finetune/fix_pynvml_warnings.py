#!/usr/bin/env python3
"""
Temporary fix for pynvml deprecation warnings in training.
This script can be run before training to suppress warnings.
"""

import warnings
import sys

def suppress_pynvml_warnings():
    """Suppress pynvml-related deprecation warnings."""
    warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")
    warnings.filterwarnings("ignore", message=".*pynvml.*")
    print("✓ Suppressed pynvml deprecation warnings")

if __name__ == "__main__":
    suppress_pynvml_warnings()