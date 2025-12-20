#!/usr/bin/env python3
"""
Setup script for  Hand Rehabilitation System
This script helps users set up the environment and install dependencies.
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print a welcome banner."""
    print("=" * 60)
    print("🤲  Hand Rehabilitation System - Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        print("   Please download from: https://www.python.org/downloads/")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True

def create_virtual_environment():
    """Create a virtual environment."""
    print("\n🔧 Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "hand_exercise_env"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error creating virtual environment")
        return False

def get_activation_command():
    """Get the appropriate activation command based on OS."""
    system = platform.system().lower()
    if system == "windows":
        return "hand_exercise_env\\Scripts\\activate"
    else:
        return "source hand_exercise_env/bin/activate"

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print("✅ Pip upgraded")
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def verify_installation():
    """Verify that all packages are installed correctly."""
    print("\n🔍 Verifying installation...")
    try:
        import streamlit
        import cv2
        import mediapipe
        import sklearn
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import joblib
        import PIL
        print("✅ All packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def check_files():
    """Check if required files exist."""
    print("\n📁 Checking required files...")
    required_files = [
        "app.py",
        "requirements.txt",
        "landmarks_features.csv",
        "exercise_classifier.pkl",
        "scaler.pkl",
        "fin.mp4",
        "all_exe.png"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (missing)")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} files are missing")
        return False
    else:
        print("✅ All required files found")
        return True

def print_next_steps():
    """Print next steps for the user."""
    activation_cmd = get_activation_command()
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("1. Activate the virtual environment:")
    print(f"   {activation_cmd}")
    print("\n2. Run the application:")
    print("   streamlit run app.py")
    print("\n3. Open your browser to: http://localhost:8501")
    print("\n4. Check 'Run Hand Detection' to start using the app")
    print("\n📖 For detailed instructions, see README.md")
    print("\n" + "=" * 60)

def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
    
    # Check files
    check_files()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
