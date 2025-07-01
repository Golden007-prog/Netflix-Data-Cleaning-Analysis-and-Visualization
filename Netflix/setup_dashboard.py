#!/usr/bin/env python3
"""
Netflix ML Dashboard Setup Script

Creates all necessary files for the Netflix ML-powered analytics dashboard.
Fixes Unicode encoding issues by using UTF-8 encoding for all file operations.
"""

import os
import subprocess
import sys
from pathlib import Path

def create_dashboard_launcher():
    """Create a Python script to launch the Streamlit dashboard"""
    
    launcher_script = '''#!/usr/bin/env python3
"""
Netflix ML Dashboard Launcher

Quick launcher for the Netflix ML-powered analytics dashboard.
"""

import subprocess
import sys
import os
import webbrowser
import time

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'scikit-learn', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        print("All packages installed!")
    else:
        print("All required packages are available!")

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("Netflix ML Dashboard Launcher")
    print("=" * 50)
    
    # Check dependencies
    check_dependencies()
    
    # Set environment variables for better performance
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_ENABLECORS'] = 'false'
    
    # Launch dashboard
    print("Launching Netflix ML Dashboard...")
    print("Dashboard will open in your default browser")
    print("To stop the dashboard, press Ctrl+C in this terminal")
    print("=" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\\nDashboard stopped by user")
    except Exception as e:
        print(f"Error launching dashboard: {str(e)}")

if __name__ == "__main__":
    launch_dashboard()
'''
    
    # Write launcher script with UTF-8 encoding
    with open('launch_dashboard.py', 'w', encoding='utf-8') as f:
        f.write(launcher_script)
    
    print("Dashboard launcher script created: launch_dashboard.py")

def create_dashboard_config():
    """Create Streamlit configuration file"""
    
    config_content = '''[global]
dataFrameSerialization = "legacy"

[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#E50914"
backgroundColor = "#221F1F"
secondaryBackgroundColor = "#808080"
textColor = "#FFFFFF"
'''
    
    # Create .streamlit directory
    os.makedirs('.streamlit', exist_ok=True)
    
    # Write config file with UTF-8 encoding
    with open('.streamlit/config.toml', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("Streamlit configuration created: .streamlit/config.toml")

def create_requirements_file():
    """Create requirements file for the dashboard"""
    
    requirements = '''# Netflix ML Dashboard Requirements
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
plotly>=5.15.0
scikit-learn>=1.3.0
joblib>=1.3.0
python-dateutil>=2.8.0
'''
    
    with open('dashboard_requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements)
    
    print("Dashboard requirements file created: dashboard_requirements.txt")

def create_dockerfile():
    """Create Dockerfile for containerized deployment"""
    
    dockerfile_content = '''FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    software-properties-common \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY dashboard_requirements.txt .
RUN pip3 install -r dashboard_requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the dashboard
ENTRYPOINT ["streamlit", "run", "streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
'''
    
    with open('Dockerfile.dashboard', 'w', encoding='utf-8') as f:
        f.write(dockerfile_content)
    
    print("Docker configuration created: Dockerfile.dashboard")

def create_docker_compose():
    """Create Docker Compose file for easy deployment"""
    
    compose_content = '''version: '3.8'

services:
  netflix-dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data:ro
      - ./models:/app/models:ro
      - ./reports:/app/reports:ro
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_ENABLECORS=false
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 5
'''
    
    with open('docker-compose.dashboard.yml', 'w', encoding='utf-8') as f:
        f.write(compose_content)
    
    print("Docker Compose configuration created: docker-compose.dashboard.yml")

def display_launch_instructions():
    """Display instructions for launching the dashboard"""
    
    instructions = f"""
NETFLIX ML DASHBOARD - LAUNCH INSTRUCTIONS
{'=' * 60}

QUICK START:
   1. Run: python launch_dashboard.py
   2. Dashboard opens automatically in browser
   3. Navigate to http://localhost:8501

MANUAL LAUNCH:
   1. Install requirements: pip install -r dashboard_requirements.txt  
   2. Run dashboard: streamlit run streamlit_dashboard.py
   3. Open browser to http://localhost:8501

DOCKER DEPLOYMENT:
   1. Build image: docker build -f Dockerfile.dashboard -t netflix-dashboard .
   2. Run container: docker run -p 8501:8501 netflix-dashboard
   3. Access at http://localhost:8501

DOCKER COMPOSE:
   1. Launch stack: docker-compose -f docker-compose.dashboard.yml up
   2. Access at http://localhost:8501
   3. Stop stack: docker-compose -f docker-compose.dashboard.yml down

DASHBOARD FEATURES:
   • Overview: Key metrics and content distribution
   • ML Predictions: Real-time content type, rating, duration prediction
   • Recommendations: AI-powered content recommendations  
   • Clustering: Market segmentation and content clustering
   • Business Intelligence: KPIs and strategic insights
   • Model Performance: ML model monitoring and metrics
   • Content Explorer: Advanced filtering and search

TROUBLESHOOTING:
   • Port 8501 busy: Use --server.port 8502
   • Module errors: Install requirements.txt
   • Data loading issues: Check data file paths
   • Performance: Increase server memory limits

{'=' * 60}
Dashboard ready for launch! Choose your preferred method above.
"""
    
    print(instructions)

def main():
    """Main setup function"""
    print("Setting up Netflix ML Dashboard deployment...")
    print("=" * 60)
    
    # Create all necessary files
    create_dashboard_launcher()
    create_dashboard_config()
    create_requirements_file()
    create_dockerfile()
    create_docker_compose()
    
    print("=" * 60)
    print("Dashboard deployment setup completed!")
    print("Files created:")
    print("   • launch_dashboard.py (Quick launcher)")
    print("   • .streamlit/config.toml (Streamlit config)")
    print("   • dashboard_requirements.txt (Dependencies)")
    print("   • Dockerfile.dashboard (Container config)")
    print("   • docker-compose.dashboard.yml (Compose config)")
    print("   • streamlit_dashboard.py (Main dashboard)")
    
    # Display launch instructions
    display_launch_instructions()

if __name__ == "__main__":
    main() 