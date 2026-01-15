#!/bin/bash
ENV_NAME="ArtTic-LAB"

# Test script wrapper
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME
fi

# Run the python test runner
python tests/run_tests.py