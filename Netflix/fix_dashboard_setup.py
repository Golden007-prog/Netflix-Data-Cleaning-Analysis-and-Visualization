#!/usr/bin/env python3
"""
Simple setup script to fix Unicode encoding issues
"""

import os

def create_files():
    # Create launcher script
    launcher_script = """#!/usr/bin/env python3
import subprocess
import sys
import os

def check_dependencies():
    required_packages = ['streamlit', 'pandas', 'numpy', 'plotly', 'scikit-learn', 'joblib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing packages: " + ", ".join(missing_packages))
        print("Installing missing packages...")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("All packages installed!")
    else:
        print("All required packages are available!")

def launch_dashboard():
    print("Netflix ML Dashboard Launcher")
    print("=" * 50)
    check_dependencies()
    
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_ENABLECORS"] = "false"
    
    print("Launching Netflix ML Dashboard...")
    print("Dashboard will open in your default browser")
    print("To stop the dashboard, press Ctrl+C in this terminal")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\\nDashboard stopped by user")
    except Exception as e:
        print("Error launching dashboard: " + str(e))

if __name__ == "__main__":
    launch_dashboard()
"""

    with open('launch_dashboard.py', 'w', encoding='utf-8') as f:
        f.write(launcher_script)

    print('Dashboard launcher created: launch_dashboard.py')

    # Create Streamlit config
    os.makedirs('.streamlit', exist_ok=True)
    config_content = """[global]
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
"""

    with open('.streamlit/config.toml', 'w', encoding='utf-8') as f:
        f.write(config_content)

    print('Streamlit config created: .streamlit/config.toml')

    # Create requirements file
    requirements = """# Netflix ML Dashboard Requirements
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
plotly>=5.15.0
scikit-learn>=1.3.0
joblib>=1.3.0
python-dateutil>=2.8.0
"""

    with open('dashboard_requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements)

    print('Requirements file created: dashboard_requirements.txt')
    print()
    print('SETUP COMPLETE!')
    print('=' * 40)
    print('Files created:')
    print('  • launch_dashboard.py')
    print('  • .streamlit/config.toml')
    print('  • dashboard_requirements.txt')
    print('  • streamlit_dashboard.py (already exists)')
    print()
    print('TO LAUNCH DASHBOARD:')
    print('1. Quick Launch: python launch_dashboard.py')
    print('2. Manual Launch: streamlit run streamlit_dashboard.py')
    print('3. Access at: http://localhost:8501')
    print()
    print('Dashboard ready for launch!')

if __name__ == "__main__":
    create_files() 