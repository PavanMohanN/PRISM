@echo off
REM Fast end-to-end proof of the pipeline (minutes on CPU). Writes results\manifest.json.
py experiments\run_all_phases.py --mode smoke --fresh
