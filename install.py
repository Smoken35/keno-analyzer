"""
Installation script for the Keno scraper.
"""

import os
import sys
import subprocess
import venv
import shutil

def create_virtual_environment():
    """Create a virtual environment."""
    print("Creating virtual environment...")
    venv.create('venv', with_pip=True)
    
    # Get the path to the virtual environment's Python executable
    if sys.platform == 'win32':
        python_path = os.path.join('venv', 'Scripts', 'python.exe')
    else:
        python_path = os.path.join('venv', 'bin', 'python')
    
    return python_path

def install_dependencies(python_path):
    """Install required packages."""
    print("Installing dependencies...")
    subprocess.check_call([python_path, '-m', 'pip', 'install', '-r', 'requirements.txt'])

def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    directories = [
        'keno_data',
        'keno_data/daily',
        'keno_data/csv',
        'keno_data/json',
        'tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def setup_logging():
    """Set up logging configuration."""
    print("Setting up logging...")
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Create an empty log file
    with open(os.path.join(log_dir, 'keno_scraper.log'), 'w') as f:
        f.write('')

def main():
    """Main installation function."""
    try:
        # Create virtual environment
        python_path = create_virtual_environment()
        
        # Install dependencies
        install_dependencies(python_path)
        
        # Create directories
        create_directories()
        
        # Set up logging
        setup_logging()
        
        print("\nInstallation completed successfully!")
        print("\nTo activate the virtual environment:")
        if sys.platform == 'win32':
            print("    venv\\Scripts\\activate")
        else:
            print("    source venv/bin/activate")
        print("\nTo run the scraper:")
        print("    python src/keno/data/playnow_scraper.py --daily-update")
        print("\nTo run tests:")
        print("    python run_tests.py")
        
    except Exception as e:
        print(f"Error during installation: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 