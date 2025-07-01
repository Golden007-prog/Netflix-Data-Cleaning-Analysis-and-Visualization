# 🚀 Netflix Analysis - Quick Start Guide

Get your Netflix data analysis environment up and running in minutes!

## ⚡ Quick Setup (5 minutes)

### 1. Clone and Navigate
```bash
git clone <repository-url>
cd netflix-analysis
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux  
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Data
Ensure your Netflix dataset is at: `data/raw/netflix1.csv`

### 5. Launch Dashboard
```bash
streamlit run streamlit_dashboard.py
```

🎉 **That's it!** Your dashboard should open at `http://localhost:8501`

---

## 📊 Run Complete Analysis

### Option 1: Automated Pipeline
```bash
python run_analysis.py
```

### Option 2: Using Make
```bash
make all
```

### Option 3: Step by Step
```bash
make setup    # Install dependencies
make raw      # Data collection  
make clean    # Data cleaning
make eda      # Exploratory analysis
make models   # Machine learning
make dashboard # Launch dashboard
```

---

## 🐳 Docker Setup (Alternative)

```bash
# Start all services
docker-compose up -d

# Access services
# Jupyter: http://localhost:8888
# Streamlit: http://localhost:8501  
# PostgreSQL: localhost:5432
```

---

## 📱 What You'll Get

### Interactive Dashboard
- 📊 **Overview Tab**: Key metrics and insights
- 📈 **Content Evolution**: Growth trends over time
- 🎭 **Genre Analysis**: Popular genres and patterns
- 🌍 **Geographic Analysis**: Global content distribution
- ⏱️ **Duration & Ratings**: Content characteristics

### Analysis Notebooks
1. **Data Collection** - Load and profile Netflix data
2. **Business Scenarios** - Define strategic questions
3. **Data Cleaning** - Prepare data for analysis
4. **EDA** - Comprehensive exploratory analysis
5. **Association Mining** - Genre pattern discovery
6. **ML Models** - Predictive modeling
7. **Dashboard** - Interactive visualizations

---

## 🛠️ Troubleshooting

### Common Issues

**Missing Data File**
```bash
# Ensure your Netflix CSV is at:
data/raw/netflix1.csv
```

**Port Already in Use**
```bash
# Use different port for Streamlit
streamlit run streamlit_dashboard.py --server.port 8502
```

**Database Connection Issues**
```bash
# Start PostgreSQL with Docker
docker run --name netflix-postgres -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres
```

**Missing Dependencies**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

---

## 🔧 Configuration

### Database Settings
Edit `src/utils.py`:
```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'netflix_db',
    'user': 'postgres',
    'password': 'your_password'
}
```

### Dashboard Customization
Edit `streamlit_dashboard.py` to:
- Modify color schemes
- Add new visualization types
- Include additional filters
- Customize layouts

---

## 📈 Expected Results

### Key Insights You'll Discover
- **Content Growth**: 300% increase in TV shows since 2016
- **Global Expansion**: 40+ countries producing content
- **Genre Trends**: Drama and International content dominate
- **Production Efficiency**: Average 18-month production cycle

### Visualizations Generated
- Time series of content additions
- Geographic heat maps
- Genre popularity rankings
- Duration distribution analyses
- Rating breakdown charts

---

## 🚀 Next Steps

Once your analysis is running:

1. **Explore the Dashboard** - Use filters to dive deep into specific segments
2. **Review Notebooks** - Check `output/` folder for detailed analysis
3. **Business Scenarios** - Read `docs/business_scenarios.md` for strategic insights
4. **Customize Analysis** - Modify notebooks for your specific questions
5. **Deploy Dashboard** - Share insights with stakeholders

---

## 💡 Pro Tips

- **Use Filters**: Dashboard filters help isolate interesting patterns
- **Check Logs**: Monitor `pipeline_execution.log` for debugging
- **Save Insights**: Export interesting visualizations from dashboard
- **Iterate**: Modify business scenarios based on initial findings

---

## 🆘 Need Help?

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Check `README.md` for comprehensive guide
- **Logs**: Review execution logs for error details
- **Community**: Join discussions on GitHub Discussions

---

**Ready to discover Netflix insights? Start with:** `streamlit run streamlit_dashboard.py` 