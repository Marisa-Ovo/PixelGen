#!/usr/bin/env bash
set -e

# Usage: ./generation.sh "/path/to/input.json" "/path/to/output_folder" n_sample
python3 "$(dirname "$0")/generate_main.py" "$1" "$2" "$3"
