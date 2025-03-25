"""
Script to run daily Keno analysis automatically.
"""

import os
import time
import schedule
import subprocess
import logging
from datetime import datetime
from performance_tracker import PerformanceTracker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('daily_analysis_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_analysis():
    """Run the daily analysis script."""
    try:
        logger.info("Starting daily analysis...")
        
        # Initialize performance tracker
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "KenoPastYears")
        tracker = PerformanceTracker(data_dir)
        
        # Run performance analysis
        performance = tracker.analyze_performance(days=100)
        
        if performance:
            logger.info("Analysis completed successfully")
            
            # Log key metrics
            recent = performance['recent_performance']
            logger.info(f"Recent Performance: Win Rate: {recent['win_percentage']:.2f}%, ROI: {recent['roi']:.2f}%")
            
            # Check if we're meeting the 100-game goal
            if recent['net_profit'] > 0:
                logger.info("✓ Goal Achieved: Profitable over 100 games!")
            else:
                logger.warning("✗ Goal Not Met: Not profitable over 100 games")
        else:
            logger.error("Failed to complete performance analysis")
            
    except Exception as e:
        logger.error(f"Error running daily analysis: {str(e)}")

def main():
    # Run analysis immediately on startup
    run_analysis()
    
    # Schedule daily analysis at 11:59 PM
    schedule.every().day.at("23:59").do(run_analysis)
    
    logger.info("Scheduler started. Will run analysis daily at 11:59 PM")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in scheduler: {str(e)}")
            time.sleep(60)  # Wait a minute before retrying

if __name__ == "__main__":
    main() 