import os

# Create Streamlit config
with open('.streamlit/config.toml', 'w', encoding='utf-8') as f:
    f.write('[global]\n')
    f.write('dataFrameSerialization = \
legacy\\n\n')
    f.write('[server]\n')
    f.write('headless = true\n')
    f.write('enableCORS = false\n')
    f.write('enableXsrfProtection = false\n\n')
    f.write('[browser]\n')
    f.write('gatherUsageStats = false\n\n')
    f.write('[theme]\n')
    f.write('primaryColor = \#E50914\\n')
    f.write('backgroundColor = \#221F1F\\n')
    f.write('secondaryBackgroundColor = \#808080\\n')
    f.write('textColor = \#FFFFFF\\n')

print('✅ Streamlit config created')

# Create requirements file
with open('dashboard_requirements.txt', 'w', encoding='utf-8') as f:
    f.write('# Netflix ML Dashboard Requirements\n')
    f.write('streamlit>=1.28.0\n')
    f.write('pandas>=1.5.0\n')
    f.write('numpy>=1.21.0\n')
    f.write('plotly>=5.15.0\n')
    f.write('scikit-learn>=1.3.0\n')
    f.write('joblib>=1.3.0\n')
    f.write('python-dateutil>=2.8.0\n')

print('✅ Requirements file created')
print()
print('🚀 SETUP COMPLETE!')
print('Files created:')
print('  • launch_dashboard.py')
print('  • .streamlit/config.toml')
print('  • dashboard_requirements.txt')
print('  • streamlit_dashboard.py (already exists)')
print()
print('TO LAUNCH DASHBOARD:')
print('1. python launch_dashboard.py')
print('2. streamlit run streamlit_dashboard.py')
print('3. Access at: http://localhost:8501')

