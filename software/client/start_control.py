#!/usr/bin/env python3
"""
Enhanced Radio Communication - Control Device Launcher
-----------------------------------------------------
This script launches the control device application and handles dependencies.
"""

import os
import sys
import subprocess
import pkg_resources

# Required packages
REQUIRED_PACKAGES = [
    'flask',
    'zeroconf'
]

def check_dependencies():
    """Check if all required packages are installed"""
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            pkg_resources.get_distribution(package)
        except pkg_resources.DistributionNotFound:
            missing.append(package)
    
    return missing

def install_dependencies(packages):
    """Install missing dependencies"""
    print(f"Installing missing dependencies: {', '.join(packages)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

def main():
    """Main entry point"""
    # Change to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check and install dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        install_dependencies(missing_packages)
    
    # Start the application
    print("Starting Enhanced Radio Communication Control Device...")
    subprocess.call([sys.executable, "app.py"])

if __name__ == "__main__":
    main()
