#!/usr/bin/env python3
"""
Netflix Analysis Pipeline Runner

This script orchestrates the execution of the complete Netflix data analysis pipeline.
It runs all notebooks in sequence and provides progress tracking.
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_execution.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class NetflixAnalysisPipeline:
    """Netflix Analysis Pipeline Orchestrator"""
    
    def __init__(self):
        self.notebooks = [
            {
                'name': '01_data_collection.ipynb',
                'description': 'Data Collection & Profiling',
                'required': True
            },
            {
                'name': '02_business_scenarios.ipynb', 
                'description': 'Business Scenarios Definition',
                'required': True
            },
            {
                'name': '03_data_cleaning_feature_engineering.ipynb',
                'description': 'Data Cleaning & Feature Engineering',
                'required': True
            },
            {
                'name': '04_exploratory_data_analysis.ipynb',
                'description': 'Exploratory Data Analysis',
                'required': True
            },
            {
                'name': '05_association_rule_mining.ipynb',
                'description': 'Association Rule Mining',
                'required': False
            },
            {
                'name': '06_machine_learning_models.ipynb',
                'description': 'Machine Learning Models',
                'required': False
            },
            {
                'name': '07_interactive_dashboard.ipynb',
                'description': 'Interactive Dashboard Setup',
                'required': False
            }
        ]
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.notebooks_dir = os.path.join(self.base_dir, 'notebooks')
        self.output_dir = os.path.join(self.base_dir, 'output')
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        logger.info("🔍 Checking prerequisites...")
        
        # Check if data file exists
        data_file = os.path.join(self.base_dir, 'data', 'raw', 'netflix1.csv')
        if not os.path.exists(data_file):
            logger.error(f"❌ Netflix dataset not found at {data_file}")
            return False
        
        # Check if notebooks exist
        missing_notebooks = []
        for notebook in self.notebooks:
            notebook_path = os.path.join(self.notebooks_dir, notebook['name'])
            if not os.path.exists(notebook_path):
                missing_notebooks.append(notebook['name'])
        
        if missing_notebooks:
            logger.error(f"❌ Missing notebooks: {', '.join(missing_notebooks)}")
            return False
        
        # Check if papermill is available
        try:
            subprocess.run(['papermill', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ papermill not found. Please install it with: pip install papermill")
            return False
        
        logger.info("✅ All prerequisites met!")
        return True
    
    def run_notebook(self, notebook_name, description):
        """Execute a single notebook using papermill"""
        logger.info(f"📓 Executing: {description}")
        
        input_path = os.path.join(self.notebooks_dir, notebook_name)
        output_path = os.path.join(self.output_dir, notebook_name)
        
        start_time = time.time()
        
        try:
            # Run notebook with papermill
            result = subprocess.run([
                'papermill', 
                input_path, 
                output_path,
                '--log-output'
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ {description} completed successfully in {execution_time:.1f}s")
                return True
            else:
                logger.error(f"❌ {description} failed:")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {description} timed out after 30 minutes")
            return False
        except Exception as e:
            logger.error(f"❌ Error executing {description}: {str(e)}")
            return False
    
    def run_pipeline(self, skip_optional=False):
        """Run the complete analysis pipeline"""
        logger.info("🚀 Starting Netflix Analysis Pipeline")
        logger.info(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not self.check_prerequisites():
            logger.error("❌ Prerequisites not met. Aborting pipeline execution.")
            return False
        
        success_count = 0
        total_notebooks = len(self.notebooks)
        
        if skip_optional:
            notebooks_to_run = [nb for nb in self.notebooks if nb['required']]
            total_notebooks = len(notebooks_to_run)
        else:
            notebooks_to_run = self.notebooks
        
        logger.info(f"📊 Executing {total_notebooks} notebooks...")
        
        for i, notebook in enumerate(notebooks_to_run, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"📋 Step {i}/{total_notebooks}: {notebook['description']}")
            logger.info(f"{'='*60}")
            
            success = self.run_notebook(notebook['name'], notebook['description'])
            
            if success:
                success_count += 1
            else:
                if notebook['required']:
                    logger.error(f"❌ Required notebook failed: {notebook['name']}")
                    logger.error("🛑 Stopping pipeline execution due to required notebook failure")
                    break
                else:
                    logger.warning(f"⚠️ Optional notebook failed: {notebook['name']}")
                    logger.info("📝 Continuing with pipeline execution...")
        
        # Pipeline completion summary
        logger.info(f"\n{'='*80}")
        logger.info("🎬 NETFLIX ANALYSIS PIPELINE SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"✅ Successfully executed: {success_count}/{total_notebooks} notebooks")
        logger.info(f"📁 Output files saved to: {self.output_dir}")
        
        if success_count == total_notebooks:
            logger.info("🎉 Pipeline completed successfully!")
            logger.info("\n🚀 Next steps:")
            logger.info("   1. Review output notebooks in the 'output' directory")
            logger.info("   2. Launch the Streamlit dashboard: streamlit run streamlit_dashboard.py")
            logger.info("   3. Explore the generated reports and visualizations")
            return True
        else:
            logger.warning(f"⚠️ Pipeline completed with {total_notebooks - success_count} failures")
            return False
    
    def generate_summary_report(self):
        """Generate a summary report of the analysis"""
        logger.info("📊 Generating pipeline summary report...")
        
        summary_path = os.path.join(self.output_dir, 'pipeline_summary.md')
        
        with open(summary_path, 'w') as f:
            f.write("# Netflix Analysis Pipeline - Execution Summary\n\n")
            f.write(f"**Execution Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Pipeline Components\n\n")
            
            for i, notebook in enumerate(self.notebooks, 1):
                output_file = os.path.join(self.output_dir, notebook['name'])
                status = "✅ Completed" if os.path.exists(output_file) else "❌ Failed/Skipped"
                f.write(f"{i}. **{notebook['description']}**\n")
                f.write(f"   - Notebook: `{notebook['name']}`\n")
                f.write(f"   - Status: {status}\n")
                f.write(f"   - Required: {'Yes' if notebook['required'] else 'No'}\n\n")
            
            f.write("## Generated Outputs\n\n")
            f.write("- Executed notebooks in `output/` directory\n")
            f.write("- Analysis reports and visualizations\n")
            f.write("- Cleaned datasets ready for dashboard\n")
            f.write("- Business scenarios documentation\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. **Review Results**: Check executed notebooks for detailed analysis\n")
            f.write("2. **Launch Dashboard**: Run `streamlit run streamlit_dashboard.py`\n")
            f.write("3. **Explore Insights**: Review business scenarios and recommendations\n")
            f.write("4. **Deploy**: Consider deploying dashboard for stakeholder access\n")
        
        logger.info(f"📋 Summary report generated: {summary_path}")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Netflix Analysis Pipeline Runner')
    parser.add_argument('--skip-optional', action='store_true', 
                       help='Skip optional notebooks (ML models, etc.)')
    parser.add_argument('--notebooks', nargs='+', 
                       help='Run specific notebooks only')
    
    args = parser.parse_args()
    
    pipeline = NetflixAnalysisPipeline()
    
    if args.notebooks:
        # Run specific notebooks
        logger.info(f"🎯 Running specific notebooks: {', '.join(args.notebooks)}")
        success_count = 0
        
        for notebook_name in args.notebooks:
            # Find notebook description
            description = "Custom Notebook"
            for nb in pipeline.notebooks:
                if nb['name'] == notebook_name:
                    description = nb['description']
                    break
            
            success = pipeline.run_notebook(notebook_name, description)
            if success:
                success_count += 1
        
        logger.info(f"✅ Completed {success_count}/{len(args.notebooks)} notebooks")
    else:
        # Run full pipeline
        success = pipeline.run_pipeline(skip_optional=args.skip_optional)
        
        # Generate summary report
        pipeline.generate_summary_report()
        
        if success:
            logger.info("🎉 Pipeline execution completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Pipeline execution completed with errors")
            sys.exit(1)

if __name__ == "__main__":
    main() 