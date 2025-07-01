# Netflix Data Analysis Pipeline

A comprehensive data science project analyzing Netflix content trends, production patterns, and market strategies using advanced analytics and machine learning.

## 🎯 Project Overview

This project provides a complete end-to-end data analysis pipeline for Netflix content data, including:

- **Data Collection & Profiling**: Automated data ingestion and quality assessment
- **Business Intelligence**: Strategic scenario analysis and KPI tracking  
- **Advanced Analytics**: Machine learning models and association rule mining
- **Interactive Dashboard**: Streamlit-powered business intelligence interface

## 📊 Business Scenarios Analyzed

1. **Content Strategy Evolution**: How Netflix's content mix has evolved over time
2. **Global Market Expansion**: Regional content growth and localization strategies
3. **Content Portfolio Optimization**: Optimal mix for maximum engagement
4. **Production Efficiency**: Time-to-market optimization opportunities
5. **Competitive Positioning**: Differentiation strategies in streaming market

## 🏗️ Project Structure

```bash
netflix-analysis/
├── data/
│   ├── raw/                    # Raw Netflix dataset
│   ├── clean/                  # Cleaned and processed data
│   └── eda/                    # EDA outputs and reports
├── notebooks/
│   ├── 01_data_collection.ipynb              # Data ingestion & profiling
│   ├── 02_business_scenarios.ipynb           # Business case definitions
│   ├── 03_data_cleaning_feature_engineering.ipynb  # Data preparation
│   ├── 04_exploratory_data_analysis.ipynb    # Comprehensive EDA
│   ├── 05_association_rule_mining.ipynb      # Genre pattern analysis
│   ├── 06_machine_learning_models.ipynb      # Predictive modeling
│   └── 07_interactive_dashboard.ipynb        # Streamlit dashboard
├── src/
│   └── utils.py                # Utility functions and database connections
├── models/                     # Trained ML models
├── reports/
│   └── eda/                    # Analysis reports and visualizations
├── docs/                       # Project documentation
├── output/                     # Notebook execution outputs
├── requirements.txt            # Python dependencies
├── Makefile                    # Pipeline automation
├── README.md                   # This file
└── .gitignore                  # Git ignore patterns
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL (optional, for database features)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd netflix-analysis
   ```

2. **Set up Python environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Usage

### Running the Complete Pipeline

Execute the entire analysis pipeline:

```bash
# Run all notebooks in sequence
make all
```

### Interactive Dashboard

Launch the Streamlit dashboard:

```bash
# Start interactive dashboard
make dashboard
```

## 🔧 Key Features

### Data Processing
- Automated data quality assessment
- Missing value handling strategies
- Feature engineering for temporal and categorical data
- Geographic data enrichment with continent mapping

### Advanced Analytics
- **Association Rule Mining**: Discover genre combinations and content patterns
- **Machine Learning Models**: Classification and regression for content prediction
- **Statistical Analysis**: Correlation analysis and trend identification
- **Time Series Analysis**: Content evolution and seasonal patterns

### Interactive Visualizations
- Dynamic filtering by year, genre, country, rating
- Comparative analysis tools
- Geographic content distribution maps
- Trend analysis with statistical annotations

---

**Built with ❤️ for data-driven content strategy**