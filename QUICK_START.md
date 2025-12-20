# 🚀 Quick Start Guide

Get the  Hand Rehabilitation System running in 5 minutes!

## ⚡ Super Quick Setup

### Option 1: Automated Setup (Recommended)
```bash
python setup.py
```

### Option 2: Manual Setup
```bash
# 1. Create virtual environment
python -m venv hand_exercise_env

# 2. Activate it
# Windows:
hand_exercise_env\Scripts\activate
# macOS/Linux:
source hand_exercise_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

## 🎯 What You'll See

1. **Demo Video**: Watch example exercises
2. **Live Detection**: Your camera feed with hand tracking
3. **Reference Images**: Proper form for each exercise
4. **Real-time Feedback**: Guidance messages on screen

## 🎮 How to Use

1. **Start**: Check "Run Hand Detection"
2. **Allow Camera**: Grant camera permissions
3. **Perform Exercises**: Do any of the 8 hand exercises
4. **Get Feedback**: Read the guidance messages

## 🔧 If Something Goes Wrong

**Camera not working?**
- Try refreshing the page
- Check browser camera permissions
- Try `cv2.VideoCapture(0)` instead of `cv2.VideoCapture(1)` in app.py

**Import errors?**
- Make sure virtual environment is activated
- Run: `pip install -r requirements.txt`

**Slow performance?**
- Close other applications
- Ensure good lighting
- Use a plain background

## 📞 Need Help?

- Check the full README.md for detailed instructions
- Look at the troubleshooting section
- Verify all files are present in the project folder

---

**That's it! You should be up and running in no time! 🎉**
