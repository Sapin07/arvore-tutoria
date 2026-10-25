#!/usr/bin/env bash
set -e; cd "$(dirname "$0")"; mkdir -p outputs
python3 01_poda_ccp.py | tee outputs/01_poda_ccp.txt
