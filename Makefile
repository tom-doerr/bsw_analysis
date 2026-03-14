.PHONY: all clean predictions forensic evidence modeling spatial misc reproduce-core recount

PYTHON ?= python

# Step 1: LR predictions (foundation)
predictions:
	$(PYTHON) wahlbezirk_lr.py

# Step 2: analyses that depend on predictions
forensic: predictions
	$(PYTHON) bsw_forensic.py
	$(PYTHON) bsw_claims_test.py
	$(PYTHON) bsw_bd_decorrelate.py

evidence: predictions
	$(PYTHON) evidence_registry.py
	$(PYTHON) top_anomalies_bb.py
	$(PYTHON) calibrate_zero_betabinom.py
	$(PYTHON) triangulate_lr_xgb.py
	$(PYTHON) evidence_dossier.py

modeling: predictions
	$(PYTHON) latent_class_pi.py
	$(PYTHON) bsw_generative.py
	$(PYTHON) bsw_bayesian.py
	$(PYTHON) null_calibration.py
	$(PYTHON) low_tail_undercount.py
	$(PYTHON) bsw_bd_swap.py

spatial: predictions
	$(PYTHON) clustering_test.py
	$(PYTHON) bsw_adjacency_did.py
	$(PYTHON) brief_colocation.py
	$(PYTHON) neighborhood_credibility.py

misc: predictions
	$(PYTHON) official_corrections.py
	$(PYTHON) bsw_recount_bias.py
	$(PYTHON) bsw_affidavits.py
	$(PYTHON) bsw_swing.py
	$(PYTHON) panel_analysis.py

recount: evidence misc
	$(PYTHON) recount_targets.py
	$(PYTHON) generate_casefiles.py
	@echo "Recount targets + case files generated."

all: forensic evidence modeling spatial misc recount
	$(PYTHON) generate_report.py
	@echo "All outputs regenerated."

# Minimal path: predictions + report + tests from public data
reproduce-core: predictions
	$(PYTHON) evidence_registry.py
	$(PYTHON) low_tail_undercount.py
	$(PYTHON) official_corrections.py
	$(PYTHON) bsw_power.py
	$(PYTHON) bsw_affidavits.py
	$(PYTHON) generate_report.py
	$(PYTHON) -m pytest tests/ -v
	@echo "Core reproduction complete."

clean:
	rm -f data/wahlbezirk_lr_predictions.csv
	rm -f data/*_calibration.csv data/*_registry.csv
	rm -f data/*_registry.json data/*_anomalies_bb.csv
	rm -f data/neighborhood_credibility.csv
	rm -f data/evidence_dossier.csv data/evidence_dossier.json
	rm -f data/recount_targets.csv
	rm -rf casefiles/
	@echo "Cleaned generated outputs."
