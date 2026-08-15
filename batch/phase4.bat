@echo off
REM Phase 4 - posterior quality (calibration breadth, joint geometry, optional mixture)
set M=%1
if "%M%"=="" set M=full
py experiments\phase4_calibration.py --mode %M%
py experiments\phase4_joint_geometry.py --mode %M%
py experiments\phase4_mixture_base.py --mode %M%
py tables\make_phase4_tables.py
py figures\make_phase4_figures.py
