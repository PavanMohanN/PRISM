@echo off
REM Phase 1 gate: unit tests (transforms + all analysis cores)
py -m pytest -q tests
if errorlevel 1 (echo TESTS FAILED & exit /b 1)
echo All tests passed.
