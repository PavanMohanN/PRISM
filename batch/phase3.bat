@echo off
REM Phase 3 - ablations (liquid-vs-static, conditioning, hybrid velocity, projection)
set M=%1
if "%M%"=="" set M=full
py experiments\phase3_liquid_vs_static.py --mode %M%
py experiments\phase3_conditioning.py --mode %M%
py experiments\phase3_hybrid_velocity.py --mode %M%
py experiments\phase3_projection.py --mode %M%
py tables\make_phase3_tables.py
