"""
Utility functions for Netflix Data Analysis Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psycopg2
from sqlalchemy import create_engine
import warnings
from typing import Optional, List, Tuple, Dict
import re
from datetime import datetime

warnings.filterwarnings('ignore')

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'netflix_db',
    'user': 'postgres',
    'password': 'password'
}

def connect_db():
    """
    Create database connection using psycopg2
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def get_engine():
    """
    Create SQLAlchemy engine for pandas operations
    """
    try:
        connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        print(f"Error creating engine: {e}")
        return None

def clean_dates(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Clean and standardize date columns
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    return df

def parse_duration(duration_str: str) -> Optional[int]:
    """
    Parse duration string and return numeric value in minutes
    For TV shows, converts seasons to episode count (assuming 10 episodes per season)
    """
    if pd.isna(duration_str):
        return None
    
    if 'min' in duration_str:
        # Extract minutes for movies
        return int(re.findall(r'\d+', duration_str)[0])
    elif 'Season' in duration_str:
        # Extract seasons and convert to episode count
        # Assuming average 10 episodes per season for consistency
        seasons = int(re.findall(r'\d+', duration_str)[0])
        return seasons * 10  # Return episode count instead of minutes
    else:
        return None

def clean_countries(country_str: str) -> List[str]:
    """
    Clean and split country strings
    """
    if pd.isna(country_str):
        return []
    return [country.strip() for country in country_str.split(',')]

def clean_genres(genre_str: str) -> List[str]:
    """
    Clean and split genre strings
    """
    if pd.isna(genre_str):
        return []
    return [genre.strip() for genre in genre_str.split(',')]

def get_continent_mapping() -> Dict[str, str]:
    """
    Map countries to continents
    """
    return {
        'United States': 'North America',
        'United Kingdom': 'Europe',
        'Canada': 'North America',
        'India': 'Asia',
        'France': 'Europe',
        'Germany': 'Europe',
        'Japan': 'Asia',
        'South Korea': 'Asia',
        'Spain': 'Europe',
        'Italy': 'Europe',
        'Australia': 'Oceania',
        'Brazil': 'South America',
        'Mexico': 'North America',
        'Netherlands': 'Europe',
        'Turkey': 'Asia',
        'Argentina': 'South America',
        'Belgium': 'Europe',
        'China': 'Asia',
        'Egypt': 'Africa',
        'South Africa': 'Africa',
        'Nigeria': 'Africa',
        'Russia': 'Europe',
        'Sweden': 'Europe',
        'Norway': 'Europe',
        'Denmark': 'Europe',
        'Switzerland': 'Europe',
        'Austria': 'Europe',
        'Thailand': 'Asia',
        'Philippines': 'Asia',
        'Indonesia': 'Asia',
        'Malaysia': 'Asia',
        'Singapore': 'Asia',
        'New Zealand': 'Oceania',
        'Ireland': 'Europe',
        'Israel': 'Asia',
        'Chile': 'South America',
        'Colombia': 'South America',
        'Peru': 'South America',
        'Venezuela': 'South America',
        'Poland': 'Europe',
        'Czech Republic': 'Europe',
        'Hungary': 'Europe',
        'Romania': 'Europe',
        'Greece': 'Europe',
        'Portugal': 'Europe',
        'Finland': 'Europe',
        'Luxembourg': 'Europe',
        'Iceland': 'Europe',
        'Croatia': 'Europe',
        'Slovenia': 'Europe',
        'Slovakia': 'Europe',
        'Bulgaria': 'Europe',
        'Lithuania': 'Europe',
        'Latvia': 'Europe',
        'Estonia': 'Europe',
        'Malta': 'Europe',
        'Cyprus': 'Europe',
        'Morocco': 'Africa',
        'Kenya': 'Africa',
        'Ghana': 'Africa',
        'Tanzania': 'Africa',
        'Uganda': 'Africa',
        'Senegal': 'Africa',
        'Cameroon': 'Africa',
        'Zimbabwe': 'Africa',
        'Botswana': 'Africa',
        'Zambia': 'Africa',
        'Algeria': 'Africa',
        'Tunisia': 'Africa',
        'Libya': 'Africa',
        'Sudan': 'Africa',
        'Ethiopia': 'Africa',
        'Vietnam': 'Asia',
        'Cambodia': 'Asia',
        'Laos': 'Asia',
        'Myanmar': 'Asia',
        'Bangladesh': 'Asia',
        'Pakistan': 'Asia',
        'Afghanistan': 'Asia',
        'Iran': 'Asia',
        'Iraq': 'Asia',
        'Jordan': 'Asia',
        'Lebanon': 'Asia',
        'Syria': 'Asia',
        'Saudi Arabia': 'Asia',
        'UAE': 'Asia',
        'Kuwait': 'Asia',
        'Qatar': 'Asia',
        'Bahrain': 'Asia',
        'Oman': 'Asia',
        'Yemen': 'Asia',
        'Kazakhstan': 'Asia',
        'Uzbekistan': 'Asia',
        'Kyrgyzstan': 'Asia',
        'Tajikistan': 'Asia',
        'Turkmenistan': 'Asia',
        'Mongolia': 'Asia',
        'North Korea': 'Asia',
        'Taiwan': 'Asia',
        'Hong Kong': 'Asia',
        'Macau': 'Asia'
    }

def plot_genre_popularity(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Create genre popularity plot
    """
    # Explode genres and count
    genre_counts = df['listed_in'].str.split(', ').explode().value_counts().head(top_n)
    
    fig = px.bar(
        x=genre_counts.index,
        y=genre_counts.values,
        title=f'Top {top_n} Most Popular Genres',
        labels={'x': 'Genre', 'y': 'Count'}
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_content_over_time(df: pd.DataFrame) -> go.Figure:
    """
    Create content additions over time plot
    """
    yearly_counts = df.groupby(['date_added_year', 'type']).size().reset_index(name='count')
    
    fig = px.line(
        yearly_counts,
        x='date_added_year',
        y='count',
        color='type',
        title='Content Additions Over Time',
        labels={'date_added_year': 'Year Added', 'count': 'Number of Titles'}
    )
    return fig

def plot_duration_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create duration distribution plot
    """
    movies = df[df['type'] == 'Movie']['duration_minutes'].dropna()
    tv_shows = df[df['type'] == 'TV Show']['duration_minutes'].dropna()
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Movies (Minutes)', 'TV Shows (Seasons)'))
    
    fig.add_trace(
        go.Histogram(x=movies, name='Movies', nbinsx=30),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=tv_shows/600, name='TV Shows', nbinsx=30),  # Convert back to seasons
        row=1, col=2
    )
    
    fig.update_layout(title_text="Duration Distribution by Content Type")
    return fig

def plot_country_distribution(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Create country distribution plot
    """
    country_counts = df['country'].str.split(', ').explode().value_counts().head(top_n)
    
    fig = px.bar(
        x=country_counts.values,
        y=country_counts.index,
        orientation='h',
        title=f'Top {top_n} Countries by Content Count',
        labels={'x': 'Number of Titles', 'y': 'Country'}
    )
    return fig

def plot_rating_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create rating distribution plot
    """
    rating_counts = df['rating'].value_counts()
    
    fig = px.pie(
        values=rating_counts.values,
        names=rating_counts.index,
        title='Content Rating Distribution'
    )
    return fig

def calculate_content_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate key content metrics
    """
    metrics = {
        'total_titles': len(df),
        'movies_count': len(df[df['type'] == 'Movie']),
        'tv_shows_count': len(df[df['type'] == 'TV Show']),
        'countries_count': len(df['country'].str.split(', ').explode().unique()),
        'genres_count': len(df['listed_in'].str.split(', ').explode().unique()),
        'avg_movie_duration': df[df['type'] == 'Movie']['duration_minutes'].mean(),
        'avg_tv_seasons': (df[df['type'] == 'TV Show']['duration_minutes'] / 600).mean(),
        'most_common_rating': df['rating'].mode().iloc[0] if not df['rating'].empty else 'N/A',
        'most_productive_year': df['date_added_year'].mode().iloc[0] if not df['date_added_year'].empty else 'N/A',
        'top_country': df['country'].str.split(', ').explode().value_counts().index[0] if not df['country'].empty else 'N/A'
    }
    return metrics

def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create correlation heatmap for numeric features
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title='Feature Correlation Heatmap',
        color_continuous_scale='RdBu_r',
        aspect="auto"
    )
    return fig

def setup_plotting_style():
    """
    Set up consistent plotting style
    """
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Plotly theme
    import plotly.io as pio
    pio.templates.default = "plotly_white"

def print_data_summary(df: pd.DataFrame, title: str = "Data Summary"):
    """
    Print comprehensive data summary
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))
    
    if 'duplicated' in df.columns or df.duplicated().any():
        print(f"\nDuplicated rows: {df.duplicated().sum()}")
    
    print(f"\n{'='*50}") 