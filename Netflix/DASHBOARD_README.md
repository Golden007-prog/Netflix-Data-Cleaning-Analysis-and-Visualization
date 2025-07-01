# Netflix ML-Powered Analytics Dashboard

##  Overview

The Netflix ML-Powered Analytics Dashboard is a comprehensive Streamlit application that provides interactive data analysis and machine learning insights for Netflix content strategy and business intelligence.

##  Quick Start

### Option 1: Quick Launch
```bash
python launch_dashboard.py
```

### Option 2: Direct Launch  
```bash
streamlit run streamlit_dashboard.py
```

### Option 3: Docker Deployment
```bash
docker-compose -f docker-compose.dashboard.yml up
```

##  Dashboard Features

###  Overview Page
- **Content Metrics**: Total titles, countries, genres, year span
- **Distribution Charts**: Movies vs TV shows, top countries
- **Time Series**: Content addition trends over years
- **ML Model Status**: Real-time model availability check

###  ML Predictions Page
- **Content Type Classification**: Predict Movie vs TV Show
- **Rating Prediction**: Predict appropriate content ratings
- **Duration Optimization**: Optimal content length recommendations
- **Real-time Inference**: Live predictions with confidence intervals

###  Content Recommendations Page
- **AI-Powered Recommendations**: Content-based filtering
- **Similarity Scoring**: TF-IDF based content similarity
- **Interactive Search**: Dynamic content exploration
- **Recommendation Analytics**: Genre distribution insights

###  Content Clustering Page
- **Market Segmentation**: K-means clustering analysis
- **Interactive Parameters**: Adjustable cluster count and features
- **Cluster Visualization**: Distribution and characteristics
- **Business Insights**: Portfolio optimization recommendations

###  Business Intelligence Page
- **Key Performance Indicators**: Growth, diversity, international content
- **Market Analysis**: Decade trends, genre popularity
- **Strategic Insights**: Data-driven business recommendations
- **Export Capabilities**: Download insights and reports

###  Model Performance Page
- **Accuracy Metrics**: Real-time model performance monitoring
- **Performance Trends**: Historical model accuracy tracking  
- **Model Health Status**: Production readiness indicators
- **Retraining Alerts**: Model refresh recommendations

###  Content Explorer Page
- **Advanced Filtering**: Multi-dimensional content search
- **Dynamic Results**: Real-time data exploration
- **Export Functionality**: Filtered dataset downloads
- **Search Analytics**: Query insights and recommendations

##  Technical Architecture

### Frontend
- **Streamlit**: Interactive web application framework
- **Plotly**: Interactive visualizations and charts
- **Custom CSS**: Netflix brand styling and themes

### Backend
- **Python**: Core application logic and data processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and arrays

### Machine Learning
- **scikit-learn**: Classification and regression models
- **XGBoost**: Gradient boosting for enhanced accuracy
- **TF-IDF**: Text feature extraction for recommendations

### Deployment
- **Docker**: Containerized deployment
- **Docker Compose**: Multi-service orchestration
- **Streamlit Cloud**: Cloud deployment option

##  Requirements

### Core Dependencies
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
plotly>=5.15.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

### Optional Dependencies
```
docker
docker-compose
nginx (for production)
```

##  Deployment Options

### Local Development
1. Install dependencies: `pip install -r dashboard_requirements.txt`
2. Run dashboard: `python launch_dashboard.py`
3. Access at: http://localhost:8501

### Docker Deployment
1. Build image: `docker build -f Dockerfile.dashboard -t netflix-dashboard .`
2. Run container: `docker run -p 8501:8501 netflix-dashboard`
3. Access at: http://localhost:8501

### Production Deployment
1. Use reverse proxy (nginx) for SSL and domain
2. Configure monitoring and logging
3. Set up automated backups
4. Scale with Docker Swarm or Kubernetes

##  Configuration

### Streamlit Configuration (.streamlit/config.toml)
```toml
[theme]
primaryColor = "#E50914"
backgroundColor = "#221F1F"
secondaryBackgroundColor = "#808080"
textColor = "#FFFFFF"
```

### Environment Variables
```bash
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLECORS=false
```

##  Support & Troubleshooting

### Common Issues
- **Port 8501 busy**: Use `--server.port 8502`
- **Module errors**: Install `dashboard_requirements.txt`
- **Data loading issues**: Check data file paths
- **Performance**: Increase server memory limits

### Performance Optimization
- Enable Streamlit caching with `@st.cache_data`
- Use efficient data structures (pandas, numpy)
- Optimize visualization rendering
- Implement lazy loading for large datasets

##  Updates & Maintenance

### Regular Tasks
- Monitor model performance metrics
- Update ML models with new data
- Refresh business intelligence insights
- Review and optimize dashboard performance

### Version Control
- Track changes in git repository
- Tag releases for deployment
- Document feature updates
- Maintain backward compatibility

##  Business Value

### Decision Support
- Data-driven content acquisition recommendations
- Portfolio optimization insights
- Market trend analysis and forecasting
- Performance monitoring and alerting

### ROI Measurement
- Content investment optimization
- Audience engagement predictions
- Market expansion opportunities
- Competitive analysis capabilities

---

**Created**: 2025-07-01 10:57:14
**Version**: 1.0.0
**Author**: Netflix Analytics Team
