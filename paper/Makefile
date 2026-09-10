LATEXMK ?= latexmk
PYTHON ?= ../.venv/bin/python

.PHONY: all bundle check clean

all: main.pdf supplement.pdf

main.pdf: main.tex references.bib neurips_2026.sty generated/revalidation_tradeoff.pdf
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

supplement.pdf: supplement.tex references.bib neurips_2026.sty
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error supplement.tex

bundle:
	$(PYTHON) ../scripts/build_submission.py

check:
	$(PYTHON) ../scripts/build_submission.py --check

clean:
	$(LATEXMK) -C main.tex
	$(LATEXMK) -C supplement.tex
