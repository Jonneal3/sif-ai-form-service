#!/bin/bash
# Quick test script that uses the venv
cd "$(dirname "$0")"
source .venv_sandbox/bin/activate
python3 test_supabase.py

