# Meme Classifier

This project is a Python-based application that uses computer vision to classify facial expressions and hand gestures to display corresponding memes. It leverages **MediaPipe**, **OpenCV**, and **NumPy** for real-time video processing and gesture recognition.

## Features

- **Real-time face and hand tracking** using MediaPipe.
- **Gesture and expression-based meme classification**:
  - Pointing, peace sign, fist, side-eye, and more.
- **Dynamic meme display** based on detected gestures and expressions.
- **Customizable thresholds** for gesture and expression detection.

## Requirements

- Python 3.8 or higher
- Libraries:
  - `mediapipe==1.0.0`
  - `numpy==2.5.1`
  - `opencv-contrib-python==5.0.0.93`

Install the dependencies using:

```bash
pip install -r requirements.txt
```
Project Structure

- `main.py`: The main script for running the application.
- `requirements.txt`: Lists the required Python libraries.
- `.gitignore`: Specifies files and directories to ignore in version control.
- `models/`: Contains the MediaPipe model files for face and hand tracking.
- `memes/`: Directory containing the meme images.

How It Works

1. Face and Hand Detection:
   - MediaPipe detects face landmarks and hand landmarks from the webcam feed.
2. Feature Extraction:
   - Calculates angles, distances, and other metrics to classify gestures and expressions.
3. Meme Classification:
   - Based on the detected gestures and expressions, a corresponding meme is selected.
4. Meme Display:
   - The selected meme is displayed in a resizable OpenCV window.

Usage

1. Place the required MediaPipe model files in the `models/` directory:
   - `face_landmarker.task`
   - `hand_landmarker.task`
2. Add your meme images to the `memes/` directory.
3. Run the application:

python main.py

4. Use your webcam to interact with the application. The program will classify gestures and display the corresponding meme.

Controls

- Press `q` to quit the application.
