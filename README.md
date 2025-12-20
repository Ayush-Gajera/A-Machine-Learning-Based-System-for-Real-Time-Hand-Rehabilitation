# Hand Rehabilitation System

A real-time hand exercise detection and feedback system using computer vision and machine learning. This application can detect 8 different hand exercises and provide real-time feedback to users.

## 🎯 Features

- **Real-time Hand Detection**: Uses MediaPipe for accurate hand landmark detection
- **8 Exercise Types**: Supports multiple hand rehabilitation exercises
- **Live Feedback**: Provides real-time guidance and corrections
- **Reference Images**: Shows proper form for each exercise
- **Machine Learning**: Uses Random Forest classifier for exercise classification
- **User-friendly Interface**: Clean Streamlit web interface

## 📋 Supported Exercises

1. **Ball Grip Wrist Down** - Ball squeezing with wrist in downward position
2. **Ball Grip Wrist Up** - Ball squeezing with wrist in upward position  
3. **Pinch** - Thumb and index finger pinch exercise
4. **Thumb Extend** - Thumb extension and opposition exercises
5. **Opposition** - Thumb opposition to other fingers
6. **Extend Out** - Finger extension exercises
7. **Finger Bend** - Finger bending and curling exercises
8. **Side Squeezer** - Lateral finger squeezing exercises

## 🛠️ Prerequisites

Before you begin, make sure you have the following installed:

### Required Software:
- **Python 3.8 or higher** (Download from [python.org](https://www.python.org/downloads/))
- **Git** (Download from [git-scm.com](https://git-scm.com/downloads))
- **Webcam** (Built-in or external USB camera)

### System Requirements:
- **Operating System**: Windows 10/11, macOS, or Linux
- **RAM**: Minimum 4GB, Recommended 8GB
- **Storage**: At least 2GB free space
- **Internet Connection**: Required for initial setup

## 📦 Installation Guide

### Step 1: Download the Project

**Option A: Using Git (Recommended)**
```bash
git clone <repository-url>
cd final
```

**Option B: Manual Download**
1. Download the project ZIP file
2. Extract it to a folder on your computer
3. Open command prompt/terminal in that folder

### Step 2: Set Up Python Environment

**For Windows:**
1. Open Command Prompt as Administrator
2. Navigate to your project folder:
   ```cmd
   cd "C:\path\to\your\project\folder"
   ```
3. Create a virtual environment:
   ```cmd
   python -m venv hand_exercise_env
   ```
4. Activate the virtual environment:
   ```cmd
   hand_exercise_env\Scripts\activate
   ```

**For macOS/Linux:**
1. Open Terminal
2. Navigate to your project folder:
   ```bash
   cd /path/to/your/project/folder
   ```
3. Create a virtual environment:
   ```bash
   python3 -m venv hand_exercise_env
   ```
4. Activate the virtual environment:
   ```bash
   source hand_exercise_env/bin/activate
   ```

### Step 3: Install Dependencies

Once your virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```

**If you encounter any errors:**
1. Update pip first:
   ```bash
   python -m pip install --upgrade pip
   ```
2. Try installing packages individually:
   ```bash
   pip install streamlit==1.32.0
   pip install opencv-python==4.8.1.78
   pip install mediapipe==0.10.7
   pip install scikit-learn==1.3.0
   pip install pandas==2.0.3
   pip install numpy==1.24.3
   pip install matplotlib==3.7.2
   pip install seaborn==0.12.2
   pip install joblib==1.3.2
   pip install pillow==10.0.0
   ```

### Step 4: Verify Installation

Test if everything is installed correctly:

```bash
python -c "import streamlit, cv2, mediapipe, sklearn; print('✅ All packages installed successfully!')"
```

## 🚀 Running the Application

### Step 1: Prepare Your Environment

1. **Activate your virtual environment** (if not already activated):
   - Windows: `hand_exercise_env\Scripts\activate`
   - macOS/Linux: `source hand_exercise_env/bin/activate`

2. **Ensure you're in the project directory**:
   ```bash
   # Check current directory
   dir  # Windows
   ls   # macOS/Linux
   ```

### Step 2: Start the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

### Step 3: Access the Application

1. The application will automatically open in your default web browser
2. If it doesn't open automatically, go to: `http://localhost:8501`
3. You should see the Hand Exercise Detection App interface

## 📱 Using the Application

### Interface Overview

The application has three main sections:

1. **Demonstration Video**: Shows example exercises
2. **Live Hand Detection**: Real-time camera feed with exercise detection
3. **Reference Image**: Shows proper form for detected exercises

### How to Use

1. **Start Detection**:
   - Check the "Run Hand Detection" checkbox
   - Allow camera access when prompted

2. **Perform Exercises**:
   - Position your hand in front of the camera
   - Perform any of the 8 supported exercises
   - The system will detect the exercise and show feedback

3. **View Feedback**:
   - Exercise name appears on the video feed
   - Feedback messages guide your form
   - Reference image shows proper technique

### Exercise Tips

- **Good Lighting**: Ensure your hand is well-lit
- **Clear Background**: Use a plain background for better detection
- **Hand Position**: Keep your hand clearly visible to the camera
- **Steady Movement**: Move slowly and deliberately

## 🔧 Troubleshooting

### Common Issues and Solutions

**Issue 1: Camera not working**
- **Solution**: Check camera permissions in your browser
- **Alternative**: Try refreshing the page or restarting the application

**Issue 2: "Module not found" errors**
- **Solution**: Make sure your virtual environment is activated
- **Alternative**: Reinstall requirements: `pip install -r requirements.txt`

**Issue 3: Application runs but no detection**
- **Solution**: Check if your webcam is working in other applications
- **Alternative**: Try changing camera index in the code (line 625: `cv2.VideoCapture(0)` instead of `cv2.VideoCapture(1)`)

**Issue 4: Slow performance**
- **Solution**: Close other applications using the camera
- **Alternative**: Reduce camera resolution in the code

**Issue 5: "Permission denied" errors**
- **Solution**: Run command prompt/terminal as administrator
- **Alternative**: Check folder permissions

### Getting Help

If you're still having issues:

1. **Check the console output** for error messages
2. **Verify your Python version**: `python --version`
3. **Check installed packages**: `pip list`
4. **Try a different browser** (Chrome, Firefox, Edge)

## 📁 Project Structure

```
final/
├── app.py                          # Main application file
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── landmarks_features.csv         # Training data
├── exercise_classifier.pkl        # Trained model
├── scaler.pkl                     # Feature scaler
├── fin.mp4                        # Demo video
├── all_exe.png                    # Reference image
├── images/                        # Exercise reference images
│   ├── Ball_Grip_Wrist_Down.jpg
│   ├── Ball_Grip_Wrist_UP.jpg
│   ├── Pinch.png
│   ├── Thumb_Extend.jpg
│   ├── Opposition.jpg
│   ├── Extend_Out.png
│   ├── Finger_Bend.png
│   └── Side_Squzzer.png
├── models/                        # Model files
├── data/                          # Data files
└── src/                           # Source code modules
    ├── models/                    # Machine learning models
    ├── feedback/                  # Exercise feedback functions
    └── utils/                     # Utility functions
```

## 🧠 How It Works

### Technical Overview

1. **Hand Detection**: MediaPipe detects 21 hand landmarks in real-time
2. **Feature Extraction**: 229 features extracted from landmark positions
3. **Machine Learning**: Random Forest classifier predicts exercise type
4. **Feedback Generation**: Exercise-specific feedback based on hand positions
5. **Visual Output**: Real-time video with overlays and reference images

### Machine Learning Model

- **Algorithm**: Random Forest Classifier
- **Features**: 190 distance features + 39 angle features
- **Accuracy**: Typically 85-95% depending on lighting and hand position
- **Training Data**: Custom dataset of hand exercise recordings

## 🔄 Updating the Application

To update the application:

1. **Backup your data** (if you have custom models or data)
2. **Pull latest changes**:
   ```bash
   git pull origin main
   ```
3. **Update dependencies**:
   ```bash
   pip install -r requirements.txt --upgrade
   ```
4. **Restart the application**:
   ```bash
   streamlit run app.py
   ```

## 📊 Performance Tips

### For Better Detection

1. **Lighting**: Use bright, even lighting
2. **Background**: Plain, uncluttered background
3. **Distance**: Keep hand 20-50cm from camera
4. **Angle**: Face camera directly
5. **Movement**: Move slowly and deliberately

### For Better Performance

1. **Close other applications** using the camera
2. **Use a dedicated GPU** if available
3. **Reduce browser tabs** and applications
4. **Restart application** if performance degrades

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Note**: This application is designed for educational and rehabilitation purposes. Always consult with healthcare professionals for proper exercise guidance. 