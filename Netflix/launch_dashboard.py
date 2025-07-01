#!/usr/bin/env python3
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
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error launching dashboard: {str(e)}")

if __name__ == "__main__":
    launch_dashboard()
