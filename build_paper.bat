@echo off
REM Compile the manuscript (requires paper_assets populated by run_full.bat).
pdflatex -interaction=nonstopmode prism_iclr2027.tex
bibtex prism_iclr2027
pdflatex -interaction=nonstopmode prism_iclr2027.tex
pdflatex -interaction=nonstopmode prism_iclr2027.tex
py tools\preflight_audit.py prism_iclr2027.tex --root .
