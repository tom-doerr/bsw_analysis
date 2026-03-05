.PHONY: all clean predictions forensic evidence

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

modeling: predictions
	$(PYTHON) latent_class_pi.py
	$(PYTHON) bsw_generative.py
	$(PYTHON) bsw_bayesian.py
	$(PYTHON) null_calibration.py

spatial: predictions
	$(PYTHON) clustering_test.py
	$(PYTHON) bsw_adjacency_did.py
	$(PYTHON) brief_colocation.py
	$(PYTHON) neighborhood_credibility.py

misc: predictions
	$(PYTHON) bsw_recount_bias.py
	$(PYTHON) bsw_affidavits.py
	$(PYTHON) bsw_swing.py
	$(PYTHON) panel_analysis.py

all: forensic evidence modeling spatial misc
	$(PYTHON) generate_report.py
	@echo "All outputs regenerated."
