@echo off
REM Phase 5 - accuracy / efficiency / scaling / generality
set M=%1
if "%M%"=="" set M=full
py experiments\phase5_main_pde.py --mode %M%
py experiments\phase5_efficiency_scaling.py --mode %M%
py experiments\phase5_superres.py --mode %M%
py experiments\phase5_generality.py --mode %M%
py tables\make_phase5_tables.py
py figures\make_phase5_figures.py
