#!/bin/bash
# Wrapper script for Promptfoo exec provider
# Activates the venv and runs the Python provider
cd "$(dirname "$0")/.."
source .venv/bin/activate
cd evals
python promptfoo_provider.py "$@"
