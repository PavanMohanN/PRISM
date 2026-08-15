@echo off
REM Phase 2 - core guaranteed properties (reversibility, fair constraints, stability)
set M=%1
if "%M%"=="" set M=full
py experiments\phase2_reversibility.py --mode %M%
py experiments\phase2_constraints_fair.py --mode %M%
py experiments\phase2_stability.py --mode %M%
py tables\make_phase2_tables.py
py figures\make_phase2_figures.py
