PYTHON ?= python3

.PHONY: smoke test audit paper reproduce ieom-submission ieom-parity

smoke:
	PYTHONPATH=src python3 -m unittest discover -s tests

test:
	PYTHONPATH=src:scripts python3 -m pytest -q

audit:
	PYTHONPATH=src:scripts python3 scripts/audit_ieom_submission.py

paper:
	cd paper_submission_ieom && latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
	cp paper_submission_ieom/main.pdf paper_submission_ieom/IEOM_Irvine_Planning_Capability_Operational_Value_Validation.pdf
	cd paper_submission_ieom && latexmk -c
	rm -f paper_submission_ieom/main.pdf paper_submission_ieom/main.xdv

reproduce:
	bash scripts/reproduce_ieom_final.sh

ieom-submission:
	$(PYTHON) scripts/build_ieom_dual_submission.py

ieom-parity:
	$(PYTHON) scripts/audit_pdf_docx_content_parity.py \
		--pdf paper_submission_ieom/IEOM_Irvine_Planning_Capability_Operational_Value_Validation.pdf \
		--docx paper_submission_ieom/IEOM_Irvine_Planning_Capability_Operational_Value_Validation.docx \
		--docx-rendered-pdf build/docx_render/IEOM_Irvine_DOCX_rendered.pdf
