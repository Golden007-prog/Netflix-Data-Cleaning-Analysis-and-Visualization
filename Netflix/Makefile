# Netflix Data Analysis Pipeline Makefile

.PHONY: setup raw clean eda all dashboard help

help:
	@echo "Netflix Data Analysis Pipeline"
	@echo "Available targets:"
	@echo "  setup     - Install dependencies and setup environment"
	@echo "  raw       - Execute data collection notebook"
	@echo "  clean     - Execute data cleaning and feature engineering"
	@echo "  eda       - Execute exploratory data analysis"
	@echo "  all       - Execute complete pipeline"
	@echo "  dashboard - Launch Streamlit dashboard"

setup:
	pip install -r requirements.txt
	mkdir -p output

raw:
	papermill notebooks/01_data_collection.ipynb output/01_data_collection.ipynb

scenarios:
	papermill notebooks/02_business_scenarios.ipynb output/02_business_scenarios.ipynb

clean:
	papermill notebooks/03_data_cleaning_feature_engineering.ipynb output/03_data_cleaning_feature_engineering.ipynb

eda:
	papermill notebooks/04_exploratory_data_analysis.ipynb output/04_exploratory_data_analysis.ipynb

mining:
	papermill notebooks/05_association_rule_mining.ipynb output/05_association_rule_mining.ipynb

models:
	papermill notebooks/06_machine_learning_models.ipynb output/06_machine_learning_models.ipynb

dashboard-nb:
	papermill notebooks/07_interactive_dashboard.ipynb output/07_interactive_dashboard.ipynb

all: setup raw scenarios clean eda mining models dashboard-nb
	@echo "Pipeline execution complete!"

dashboard:
	streamlit run notebooks/07_interactive_dashboard.ipynb 