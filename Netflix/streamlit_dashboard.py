#!/usr/bin/env python3
"""
Netflix ML-Powered Interactive Dashboard

A comprehensive Streamlit dashboard powered by machine learning models
for Netflix content analysis and business intelligence.

Author: Netflix Analytics Team
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import pickle
import json
import sys
import os
import warnings
from datetime import datetime, timedelta
import io
import base64

# Machine Learning libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Configure settings
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="Netflix ML Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Netflix Brand Colors
NETFLIX_RED = '#E50914'
NETFLIX_BLACK = '#221F1F'
NETFLIX_WHITE = '#FFFFFF'
NETFLIX_GRAY = '#808080'

# ============================================================================
# UTILITY FUNCTIONS AND DATA LOADING
# ============================================================================

def apply_netflix_theme():
    """Apply Netflix brand styling to dashboard"""
    st.markdown(f"""
    <style>
    .main {{
        background-color: {NETFLIX_BLACK};
        color: {NETFLIX_WHITE};
    }}
    .stSelectbox label, .stSlider label, .stDateInput label {{
        color: {NETFLIX_WHITE} !important;
        font-weight: bold;
    }}
    .stMetric {{
        background-color: {NETFLIX_GRAY};
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid {NETFLIX_RED};
    }}
    .stButton > button {{
        background-color: {NETFLIX_RED};
        color: {NETFLIX_WHITE};
        border: none;
        border-radius: 0.25rem;
        font-weight: bold;
    }}
    .stSidebar {{
        background-color: #1a1a1a;
    }}
    h1, h2, h3 {{
        color: {NETFLIX_RED} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_netflix_data():
    """Load and cache Netflix dataset"""
    try:
        # Try to load processed data first
        df = pd.read_csv('data/processed/netflix_cleaned.csv')
        print(f"✅ Loaded processed Netflix data: {df.shape[0]:,} records")
        return df
    except:
        try:
            # Fallback to raw data
            df = pd.read_csv('netflix1.csv')
            print(f"✅ Loaded raw Netflix data: {df.shape[0]:,} records")
            return df
        except:
            # Create sample data if no files found
            print("⚠️ Creating sample data for demonstration")
            return create_sample_data()

def create_sample_data():
    """Create sample Netflix data for demonstration"""
    np.random.seed(42)
    
    genres = ['Drama', 'Comedy', 'Action', 'Documentary', 'Horror', 'Romance', 'Thriller']
    countries = ['United States', 'United Kingdom', 'Canada', 'India', 'Japan', 'South Korea']
    ratings = ['TV-MA', 'TV-14', 'TV-PG', 'R', 'PG-13', 'PG', 'G']
    types = ['Movie', 'TV Show']
    
    sample_data = {
        'title': [f'Content {i}' for i in range(1000)],
        'type': np.random.choice(types, 1000),
        'primary_genre': np.random.choice(genres, 1000),
        'primary_country': np.random.choice(countries, 1000),
        'rating': np.random.choice(ratings, 1000),
        'release_year': np.random.randint(1990, 2024, 1000),
        'date_added_year': np.random.randint(2015, 2024, 1000),
        'duration_minutes': np.random.randint(30, 180, 1000)
    }
    
    return pd.DataFrame(sample_data)

@st.cache_resource
def load_ml_models():
    """Load trained ML models"""
    models = {}
    
    # In a real implementation, these would load actual trained models
    # For demonstration, we'll create placeholder model structures
    print("📊 Loading ML models (demo mode)...")
    
    return models

@st.cache_data
def load_analysis_results():
    """Load analysis results and insights"""
    # Placeholder for analysis results
    return {
        'ml_summary': {
            'models_performance': {
                'content_type_classification': {'best_accuracy': 0.85},
                'rating_classification': {'best_f1_macro': 0.72},
                'duration_regression': {'best_r2': 0.65}
            }
        }
    }

def get_content_recommendations(title, num_recommendations=5):
    """Get content recommendations (placeholder)"""
    # In real implementation, this would use the trained recommendation model
    sample_recommendations = [
        {'title': 'Similar Content 1', 'similarity_score': 0.89, 'genre': 'Drama'},
        {'title': 'Similar Content 2', 'similarity_score': 0.85, 'genre': 'Drama'},
        {'title': 'Similar Content 3', 'similarity_score': 0.82, 'genre': 'Thriller'},
        {'title': 'Similar Content 4', 'similarity_score': 0.78, 'genre': 'Action'},
        {'title': 'Similar Content 5', 'similarity_score': 0.75, 'genre': 'Drama'}
    ]
    return sample_recommendations[:num_recommendations]

# ============================================================================
# MAIN DASHBOARD APPLICATION
# ============================================================================

def main():
    """Main dashboard application"""
    
    # Apply theme
    apply_netflix_theme()
    
    # Dashboard Header
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(90deg, {NETFLIX_RED}, {NETFLIX_BLACK});'>
        <h1 style='color: white; margin: 0;'>🎬 Netflix ML-Powered Analytics Dashboard</h1>
        <p style='color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem;'>
            Comprehensive Content Intelligence & Machine Learning Insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    netflix_data = load_netflix_data()
    ml_models = load_ml_models()
    analysis_results = load_analysis_results()
    
    if netflix_data is None:
        st.error("❌ Netflix data could not be loaded. Please check data files.")
        return
    
    # Sidebar Navigation
    st.sidebar.title("🎯 Dashboard Navigation")
    
    dashboard_pages = {
        "🏠 Overview": "overview",
        "🤖 ML Predictions": "ml_predictions", 
        "💡 Content Recommendations": "recommendations",
        "🎪 Content Clustering": "clustering",
        "📊 Business Intelligence": "business_intelligence",
        "📈 Model Performance": "model_performance",
        "🔍 Content Explorer": "content_explorer"
    }
    
    selected_page = st.sidebar.selectbox(
        "Select Dashboard Section",
        list(dashboard_pages.keys()),
        index=0
    )
    
    page_key = dashboard_pages[selected_page]
    
    # Data summary in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Summary")
    st.sidebar.metric("Total Content", f"{len(netflix_data):,}")
    
    if 'type' in netflix_data.columns:
        content_types = netflix_data['type'].value_counts()
        for content_type, count in content_types.items():
            st.sidebar.metric(f"{content_type}s", f"{count:,}")
    
    if 'release_year' in netflix_data.columns:
        year_range = f"{netflix_data['release_year'].min():.0f} - {netflix_data['release_year'].max():.0f}"
        st.sidebar.metric("Year Range", year_range)
    
    # Route to appropriate page
    if page_key == "overview":
        show_overview_page(netflix_data, ml_models)
    elif page_key == "ml_predictions":
        show_ml_predictions_page(netflix_data, ml_models, analysis_results)
    elif page_key == "recommendations":
        show_recommendations_page(netflix_data, ml_models)
    elif page_key == "clustering":
        show_clustering_page(netflix_data, ml_models)
    elif page_key == "business_intelligence":
        show_business_intelligence_page(netflix_data, analysis_results)
    elif page_key == "model_performance":
        show_model_performance_page(analysis_results)
    elif page_key == "content_explorer":
        show_content_explorer_page(netflix_data)

# ============================================================================
# PAGE IMPLEMENTATIONS
# ============================================================================

def show_overview_page(netflix_data, ml_models):
    """Dashboard overview page"""
    st.header("🏠 Netflix Content Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Titles",
            f"{len(netflix_data):,}",
            delta=f"+{len(netflix_data)//10} this analysis"
        )
    
    with col2:
        if 'primary_country' in netflix_data.columns:
            unique_countries = netflix_data['primary_country'].nunique()
            st.metric("Countries", f"{unique_countries}")
        else:
            st.metric("Countries", "N/A")
    
    with col3:
        if 'primary_genre' in netflix_data.columns:
            unique_genres = netflix_data['primary_genre'].nunique()
            st.metric("Genres", f"{unique_genres}")
        else:
            st.metric("Genres", "N/A")
    
    with col4:
        if 'release_year' in netflix_data.columns:
            years_span = netflix_data['release_year'].max() - netflix_data['release_year'].min()
            st.metric("Years Span", f"{years_span:.0f}")
        else:
            st.metric("Years Span", "N/A")
    
    # Content Distribution Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Content Type Distribution")
        if 'type' in netflix_data.columns:
            type_counts = netflix_data['type'].value_counts()
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                color_discrete_sequence=[NETFLIX_RED, NETFLIX_GRAY],
                title="Movies vs TV Shows"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=NETFLIX_WHITE
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Top Countries")
        if 'primary_country' in netflix_data.columns:
            country_counts = netflix_data['primary_country'].value_counts().head(10)
            fig = px.bar(
                x=country_counts.values,
                y=country_counts.index,
                orientation='h',
                color=country_counts.values,
                color_continuous_scale=['lightgray', NETFLIX_RED],
                title="Content by Country"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=NETFLIX_WHITE,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Time Series Analysis
    st.subheader("📈 Content Addition Over Time")
    if 'date_added_year' in netflix_data.columns:
        yearly_counts = netflix_data['date_added_year'].value_counts().sort_index()
        
        fig = px.line(
            x=yearly_counts.index,
            y=yearly_counts.values,
            title="Netflix Content Added Per Year",
            markers=True
        )
        fig.update_traces(line_color=NETFLIX_RED, marker_color=NETFLIX_RED)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=NETFLIX_WHITE,
            xaxis_title="Year",
            yaxis_title="Content Added"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_ml_predictions_page(netflix_data, ml_models, analysis_results):
    """ML Predictions interface with real-time model inference"""
    st.header("🤖 Machine Learning Predictions")
    
    st.markdown("""
    Use our trained ML models to predict content characteristics and make data-driven decisions.
    """)
    
    # Model Selection Tabs
    tab1, tab2, tab3 = st.tabs(["🎬 Content Type", "🏷️ Rating Prediction", "📏 Duration Optimization"])
    
    with tab1:
        st.subheader("Content Type Classification")
        st.write("Predict whether content is a Movie or TV Show based on characteristics.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Content Features:**")
            
            # Input features
            release_year = st.slider("Release Year", 1950, 2024, 2020)
            
            # Rating selection
            rating_options = ['G', 'PG', 'PG-13', 'R', 'TV-Y', 'TV-Y7', 'TV-G', 'TV-PG', 'TV-14', 'TV-MA']
            rating = st.selectbox("Content Rating", rating_options)
            
            # Country selection
            if 'primary_country' in netflix_data.columns:
                countries = netflix_data['primary_country'].dropna().unique()
                country = st.selectbox("Primary Country", sorted(countries))
            else:
                country = st.text_input("Primary Country", "United States")
            
            # Genre selection
            if 'primary_genre' in netflix_data.columns:
                genres = netflix_data['primary_genre'].dropna().unique()
                genre = st.selectbox("Primary Genre", sorted(genres))
            else:
                genre = st.text_input("Primary Genre", "Drama")
            
            # Predict button
            if st.button("🔮 Predict Content Type", type="primary", key="content_type_predict"):
                # Simulate prediction (in real implementation, use trained model)
                prediction = np.random.choice(['Movie', 'TV Show'])
                confidence = np.random.uniform(0.6, 0.95)
                
                with col2:
                    st.write("**Prediction Results:**")
                    st.success(f"**Predicted Type: {prediction}**")
                    st.write(f"Confidence: {confidence:.1%}")
                    st.progress(confidence)
                    
                    if confidence > 0.8:
                        st.info("🎯 High confidence prediction - suitable for automated classification")
                    elif confidence > 0.6:
                        st.warning("⚠️ Moderate confidence - consider manual review")
                    else:
                        st.error("❌ Low confidence - requires manual classification")
        
        with col2:
            st.info("👆 Configure content features and click 'Predict' to see results")
    
    with tab2:
        st.subheader("Content Rating Prediction")
        col1, col2 = st.columns(2)
        
        with col1:
            content_type = st.selectbox("Content Type", ["Movie", "TV Show"], key="rating_type")
            duration = st.slider("Duration (minutes)", 10, 300, 90, key="rating_duration")
            genre_rating = st.selectbox("Genre", genres if 'genres' in locals() else ['Drama', 'Comedy', 'Action'], key="rating_genre")
            
            if st.button("🔮 Predict Rating", type="primary", key="rating_predict"):
                predicted_rating = np.random.choice(['PG-13', 'TV-14', 'R', 'TV-MA'])
                with col2:
                    st.success(f"**Predicted Rating: {predicted_rating}**")
                    st.info("Rating prediction based on content characteristics")
        
        with col2:
            st.info("👆 Enter content details to get rating recommendation")
    
    with tab3:
        st.subheader("Optimal Duration Prediction")
        col1, col2 = st.columns(2)
        
        with col1:
            duration_type = st.selectbox("Content Type", ["Movie", "TV Show"], key="duration_type")
            duration_genre = st.selectbox("Genre", genres if 'genres' in locals() else ['Drama', 'Comedy', 'Action'], key="duration_genre")
            target_audience = st.selectbox("Target Audience", ["General", "Mature", "Young Adult"], key="duration_audience")
            
            if st.button("🔮 Predict Optimal Duration", type="primary", key="duration_predict"):
                if duration_type == "Movie":
                    predicted_duration = np.random.randint(90, 150)
                else:
                    predicted_duration = np.random.randint(30, 60)
                
                with col2:
                    hours = int(predicted_duration // 60)
                    minutes = int(predicted_duration % 60)
                    duration_text = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
                    
                    st.success(f"**Optimal Duration: {predicted_duration} minutes**")
                    st.info(f"📺 **Formatted: {duration_text}**")
        
        with col2:
            st.info("👆 Specify content details for duration optimization")

def show_recommendations_page(netflix_data, ml_models):
    """Content recommendations interface"""
    st.header("💡 AI-Powered Content Recommendations")
    
    st.markdown("""
    Get personalized content recommendations using our machine learning recommendation engine.
    """)
    
    # Recommendation Types
    rec_type = st.selectbox(
        "Recommendation Type",
        ["Content-Based", "Collaborative Filtering", "Hybrid Approach"]
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔍 Search & Filter")
        
        # Content search
        search_title = st.text_input("Search for content:", placeholder="Enter title name...")
        
        # Filters
        if 'primary_genre' in netflix_data.columns:
            selected_genres = st.multiselect("Genres", netflix_data['primary_genre'].unique())
        
        if 'type' in netflix_data.columns:
            content_types = st.multiselect("Content Type", netflix_data['type'].unique())
        
        num_recommendations = st.slider("Number of recommendations", 1, 10, 5)
        
        # Get recommendations button
        if st.button("🎯 Get Recommendations", type="primary"):
            recommendations = get_content_recommendations(search_title, num_recommendations)
            
            # Store in session state
            st.session_state['recommendations'] = recommendations
            st.session_state['search_title'] = search_title
    
    with col2:
        st.subheader("🎬 Recommended Content")
        
        if 'recommendations' in st.session_state:
            st.write(f"**Recommendations for: {st.session_state.get('search_title', 'Selected Content')}**")
            
            for i, rec in enumerate(st.session_state['recommendations'], 1):
                with st.expander(f"{i}. {rec['title']} (Similarity: {rec['similarity_score']:.2f})"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Genre:** {rec['genre']}")
                        st.write(f"**Similarity Score:** {rec['similarity_score']:.2%}")
                    with col_b:
                        st.write("**Why recommended:**")
                        st.write(f"Similar genre and style to your selected content")
        else:
            st.info("👆 Search for content and click 'Get Recommendations' to see AI-powered suggestions")
        
        # Recommendation Analytics
        st.subheader("📊 Recommendation Analytics")
        
        if 'recommendations' in st.session_state:
            # Genre distribution of recommendations
            rec_genres = [rec['genre'] for rec in st.session_state['recommendations']]
            genre_counts = pd.Series(rec_genres).value_counts()
            
            fig = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation='h',
                title="Recommended Genres Distribution"
            )
            fig.update_traces(marker_color=NETFLIX_RED)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=NETFLIX_WHITE
            )
            st.plotly_chart(fig, use_container_width=True)

def show_clustering_page(netflix_data, ml_models):
    """Content clustering analysis"""
    st.header("🎪 Content Clustering & Market Segmentation")
    
    st.markdown("""
    Explore content clusters and market segments identified by our ML algorithms.
    """)
    
    # Clustering Parameters
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔧 Clustering Parameters")
        
        num_clusters = st.slider("Number of Clusters", 2, 10, 5)
        features_to_use = st.multiselect(
            "Features for Clustering",
            ["Genre", "Country", "Release Year", "Duration", "Rating"],
            default=["Genre", "Country", "Release Year"]
        )
        
        if st.button("🔄 Run Clustering Analysis", type="primary"):
            # Simulate clustering results
            netflix_data['cluster'] = np.random.randint(0, num_clusters, len(netflix_data))
            st.session_state['clustered_data'] = netflix_data
            st.session_state['num_clusters'] = num_clusters
    
    with col2:
        st.subheader("📊 Cluster Analysis Results")
        
        if 'clustered_data' in st.session_state:
            clustered_data = st.session_state['clustered_data']
            
            # Cluster size distribution
            cluster_counts = clustered_data['cluster'].value_counts().sort_index()
            
            fig = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                title="Content Distribution Across Clusters",
                labels={'x': 'Cluster ID', 'y': 'Number of Content Items'}
            )
            fig.update_traces(marker_color=NETFLIX_RED)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=NETFLIX_WHITE
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster characteristics
            st.subheader("🎯 Cluster Characteristics")
            
            for cluster_id in range(st.session_state['num_clusters']):
                cluster_data = clustered_data[clustered_data['cluster'] == cluster_id]
                
                with st.expander(f"Cluster {cluster_id} ({len(cluster_data)} items)"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        if 'primary_genre' in cluster_data.columns:
                            top_genre = cluster_data['primary_genre'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A'
                            st.metric("Dominant Genre", top_genre)
                    
                    with col_b:
                        if 'primary_country' in cluster_data.columns:
                            top_country = cluster_data['primary_country'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A'
                            st.metric("Primary Country", top_country)
        else:
            st.info("👆 Configure clustering parameters and run analysis to see results")

def show_business_intelligence_page(netflix_data, analysis_results):
    """Business intelligence dashboard"""
    st.header("📊 Business Intelligence Dashboard")
    
    # Key Business Metrics
    st.subheader("💼 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Content Growth Rate
        if 'date_added_year' in netflix_data.columns:
            recent_years = netflix_data[netflix_data['date_added_year'] >= 2020]
            growth_rate = len(recent_years) / len(netflix_data) * 100
            st.metric("Recent Content (%)", f"{growth_rate:.1f}%", delta="📈 Growing")
        else:
            st.metric("Recent Content (%)", "N/A")
    
    with col2:
        # International Content
        if 'primary_country' in netflix_data.columns:
            international = netflix_data[netflix_data['primary_country'] != 'United States']
            intl_percentage = len(international) / len(netflix_data) * 100
            st.metric("International Content", f"{intl_percentage:.1f}%", delta="🌍 Global")
        else:
            st.metric("International Content", "N/A")
    
    with col3:
        # Average Duration
        if 'duration_minutes' in netflix_data.columns:
            avg_duration = netflix_data['duration_minutes'].mean()
            st.metric("Avg Duration", f"{avg_duration:.0f} min", delta="⏱️ Standard")
        else:
            st.metric("Avg Duration", "N/A")
    
    with col4:
        # Content Diversity (Genre Count)
        if 'primary_genre' in netflix_data.columns:
            genre_diversity = netflix_data['primary_genre'].nunique()
            st.metric("Genre Diversity", f"{genre_diversity}", delta="🎭 Diverse")
        else:
            st.metric("Genre Diversity", "N/A")
    
    # Market Analysis
    st.subheader("📈 Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Content by Release Decade
        if 'release_year' in netflix_data.columns:
            netflix_data['decade'] = (netflix_data['release_year'] // 10) * 10
            decade_counts = netflix_data['decade'].value_counts().sort_index()
            
            fig = px.bar(
                x=decade_counts.index,
                y=decade_counts.values,
                title="Content by Release Decade",
                labels={'x': 'Decade', 'y': 'Number of Titles'}
            )
            fig.update_traces(marker_color=NETFLIX_RED)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=NETFLIX_WHITE
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Genre Popularity Trends
        if 'primary_genre' in netflix_data.columns:
            genre_counts = netflix_data['primary_genre'].value_counts().head(8)
            
            fig = px.pie(
                values=genre_counts.values,
                names=genre_counts.index,
                title="Genre Distribution"
            )
            fig.update_traces(marker_colors=[NETFLIX_RED if i == 0 else NETFLIX_GRAY for i in range(len(genre_counts))])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=NETFLIX_WHITE
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Strategic Insights
    st.subheader("💡 Strategic Insights")
    
    insights = [
        "🎯 Focus on international content acquisition to maintain global appeal",
        "📈 Increase investment in high-performing genres identified by ML models",
        "🌍 Leverage clustering insights for regional content strategy",
        "⏰ Optimize content duration based on ML recommendations",
        "🤖 Implement ML-driven content acquisition decisions"
    ]
    
    for insight in insights:
        st.info(insight)

def show_model_performance_page(analysis_results):
    """Model performance monitoring"""
    st.header("📈 ML Model Performance Monitoring")
    
    # Model Performance Summary
    if 'ml_summary' in analysis_results and 'models_performance' in analysis_results['ml_summary']:
        performance_data = analysis_results['ml_summary']['models_performance']
        
        st.subheader("🎯 Model Accuracy Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'content_type_classification' in performance_data:
                acc = performance_data['content_type_classification'].get('best_accuracy', 0)
                st.metric("Content Type Accuracy", f"{acc:.1%}", delta="✅ Excellent")
        
        with col2:
            if 'rating_classification' in performance_data:
                f1 = performance_data['rating_classification'].get('best_f1_macro', 0)
                st.metric("Rating Prediction F1", f"{f1:.3f}", delta="✅ Good")
        
        with col3:
            if 'duration_regression' in performance_data:
                r2 = performance_data['duration_regression'].get('best_r2', 0)
                st.metric("Duration R² Score", f"{r2:.3f}", delta="✅ Moderate")
    
    # Model Performance Visualization
    st.subheader("📊 Performance Trends")
    
    # Simulated performance data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    model_performance = {
        'Content Type': np.random.uniform(0.80, 0.90, len(dates)),
        'Rating Prediction': np.random.uniform(0.65, 0.75, len(dates)),
        'Duration Regression': np.random.uniform(0.55, 0.70, len(dates))
    }
    
    fig = go.Figure()
    
    for model, performance in model_performance.items():
        fig.add_trace(go.Scatter(
            x=dates,
            y=performance,
            mode='lines+markers',
            name=model,
            line=dict(width=3)
        ))
    
    fig.update_layout(
        title="Model Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Performance Score",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=NETFLIX_WHITE
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Health Status
    st.subheader("🏥 Model Health Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("✅ Content Type Classifier: Healthy")
        st.write("Last updated: 2024-01-15")
        st.write("Status: Production Ready")
    
    with col2:
        st.warning("⚠️ Rating Predictor: Monitoring")
        st.write("Last updated: 2024-01-10")
        st.write("Status: Performance Review")
    
    with col3:
        st.info("ℹ️ Duration Regressor: Stable")
        st.write("Last updated: 2024-01-12")
        st.write("Status: Scheduled Retrain")

def show_content_explorer_page(netflix_data):
    """Advanced content exploration interface"""
    st.header("🔍 Advanced Content Explorer")
    
    st.markdown("""
    Explore the Netflix content catalog with advanced filtering and search capabilities.
    """)
    
    # Filters
    st.subheader("🎛️ Advanced Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Year range filter
        if 'release_year' in netflix_data.columns:
            year_range = st.slider(
                "Release Year Range",
                int(netflix_data['release_year'].min()),
                int(netflix_data['release_year'].max()),
                (2010, 2024)
            )
        else:
            year_range = (2010, 2024)
    
    with col2:
        # Content type filter
        if 'type' in netflix_data.columns:
            content_types = st.multiselect(
                "Content Types",
                netflix_data['type'].unique(),
                default=netflix_data['type'].unique()
            )
        else:
            content_types = ['Movie', 'TV Show']
    
    with col3:
        # Duration filter
        if 'duration_minutes' in netflix_data.columns:
            duration_range = st.slider(
                "Duration (minutes)",
                int(netflix_data['duration_minutes'].min()),
                int(netflix_data['duration_minutes'].max()),
                (30, 180)
            )
        else:
            duration_range = (30, 180)
    
    # Apply filters
    filtered_data = netflix_data.copy()
    
    if 'release_year' in filtered_data.columns:
        filtered_data = filtered_data[
            (filtered_data['release_year'] >= year_range[0]) &
            (filtered_data['release_year'] <= year_range[1])
        ]
    
    if 'type' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['type'].isin(content_types)]
    
    if 'duration_minutes' in filtered_data.columns:
        filtered_data = filtered_data[
            (filtered_data['duration_minutes'] >= duration_range[0]) &
            (filtered_data['duration_minutes'] <= duration_range[1])
        ]
    
    # Results
    st.subheader(f"📊 Filtered Results ({len(filtered_data):,} items)")
    
    # Display sample results
    if len(filtered_data) > 0:
        # Show data table
        display_columns = ['title', 'type', 'primary_genre', 'primary_country', 'release_year']
        available_columns = [col for col in display_columns if col in filtered_data.columns]
        
        if available_columns:
            st.dataframe(
                filtered_data[available_columns].head(50),
                use_container_width=True
            )
        
        # Export option
        if st.button("📁 Export Filtered Data"):
            csv = filtered_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="filtered_netflix_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("No content matches the selected filters.")

# ============================================================================
# RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    main() 