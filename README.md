# Hand Exercise Detection and Feedback System

This repository contains the code, documentation, and resources for the Hand Exercise Detection and Feedback System, a machine learning-based application designed to detect and provide real-time feedback on various hand exercises using webcam input. The system leverages MediaPipe for hand landmark detection and a Random Forest Classifier for exercise classification, with features such as confidence-based feedback adjustment and user-friendly annotations.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Dataset Description](#dataset-description)
- [System Architecture](#system-architecture)
- [Evaluation Measures](#evaluation-measures)
- [Streamlit Application](#streamlit-application)
- [Usage](#usage)
- [Project Timeline](#project-timeline)
- [Future Scope](#future-scope)
- [Contributing](#contributing)

## Project Overview
The Hand Exercise Detection and Feedback System is designed for physiotherapy and rehabilitation purposes. It helps users perform prescribed hand exercises with proper posture and provides real-time feedback to ensure the exercises are executed correctly.

**Key objectives include:**
- Accurate detection of hand exercises using a Random Forest Classifier.
- Real-time feedback through webcam-based live detection.
- Adjustable confidence-based feedback to account for variations in detection accuracy.

## Features
- **Real-Time Detection:** Detects and classifies eight types of hand exercises from live webcam input.
- **Feedback Annotation:** Annotates the video feed with exercise names, detection confidence, and corrective feedback.
- **Confidence Adjustment:** Reduces displayed confidence for low-detection scenarios to ensure accurate user feedback.
- **Customizable Training:** Allows users to train the model with new data or use a pre-trained model.
- **Data Collection and Review:** Enables efficient data recording and outlier removal during preprocessing.
- **Visualization Tools:** Displays performance metrics and system evaluation in detailed plots.

## Technologies Used
- Programming Language: Python
- Machine Learning: scikit-learn (Random Forest Classifier)
- Computer Vision: MediaPipe
- Web Application Framework: Streamlit
- Visualization: Matplotlib, Seaborn
- Development Environment: OpenCV, NumPy, Pandas

## Installation
Clone the repository:
```bash
git clone https://github.com/your-username/hand-exercise-detection.git
cd hand-exercise-detection
```
## Install the required dependencies:
```bash
pip install -r requirements.txt
```
## Run the Streamlit application:
```bash
streamlit run app.py
```


## Dataset Description
The dataset contains eight hand exercises captured using a webcam positioned 150 cm from the floor. Frames were extracted at 30 FPS to ensure detailed coverage.

**Exercises Captured:**
- Ball_Grip_Wrist_Down
- Ball_Grip_Wrist_UP
- Pinch
- Thumb_Extend
- Opposition
- Extend Out
- Finger Bend
- Side Squeezer

**Feature Extraction:**
- Distances and angles between hand landmarks are computed using MediaPipe.
- Processed data is stored in a CSV file for training and evaluation.

## System Architecture
**Data Collection:**
- Webcam feed captures hand exercise videos.
- Frames are preprocessed to extract hand landmarks using MediaPipe.

**Feature Engineering:**
- Distances and angles are computed for hand landmarks.

**Model Training:**
- A Random Forest Classifier is trained on labeled data to classify exercises.

**Feedback Mechanism:**
- Real-time predictions annotate the live video feed with exercise names, confidence scores, and corrective feedback.
- Confidence-based adjustment ensures accurate user feedback.

**Web Application:**
- Streamlit displays live annotated webcam feed and interactive controls.

## Evaluation Measures
Performance metrics were computed on the validation set:

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 92.4%  |
| Precision | 90.2%  |
| Recall    | 88.5%  |
| F1-Score  | 89.3%  |

## Streamlit Application
The Streamlit app provides an interactive interface to:
- View real-time webcam feed with annotations.
- Train or load the Random Forest model.
- Adjust parameters for feature extraction and feedback logic.

To launch the app:
```bash
streamlit run app.py
```


## Usage
**Data Collection:**
- Use `data_collection.py` to record videos for training.
- Extract frames and landmarks using `landmark_extractor.py`.

**Training the Model:**
- Run `train_model.py` to train the Random Forest Classifier.
- Save the trained model for deployment.

**Real-Time Detection:**
- Use `app.py` to detect and annotate hand exercises live.

**Evaluation:**
- Run `evaluation.py` to compute performance metrics and generate plots.

## Project Timeline
| Sprint      | Duration             | Tasks                                    |
|-------------|----------------------|------------------------------------------|
| Sprint 1    | July 2024            | Data Collection, Feature Engineering     |
| Sprint 2    | August 2024          | Model Training, Feedback Logic Implementation |
| Sprint 3    | September 2024       | Streamlit App Development                |
| Sprint 4    | October-November 2024| Deployment, User Testing, Research Paper Writing |

## Future Scope
- Extend the system to support additional exercises and rehabilitation use cases.
- Incorporate deep learning models for improved detection accuracy.
- Develop a mobile version of the application.

## Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes and open a pull request
