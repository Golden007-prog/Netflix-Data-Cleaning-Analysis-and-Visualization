from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="netflix-analysis",
    version="1.0.0",
    author="Netflix Data Analysis Team",
    author_email="analysis@netflix-project.com",
    description="A comprehensive data science project analyzing Netflix content trends and strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/netflix-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "netflix-analysis=run_analysis:main",
            "netflix-dashboard=streamlit_dashboard:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    zip_safe=False,
) 