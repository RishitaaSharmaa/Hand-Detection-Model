# Hand-Detection-Model

This project demonstrates a real-time hand detection system using MediaPipe's `HandLandmarker` API, OpenCV for visualization, and Python for processing. The system identifies hand landmarks, draws them on a live video feed, and connects them with lines to represent the hand's skeletal structure.

## Features
- Real-time hand landmark detection.
- Visualizes hand landmarks with points and connections.
- Detects up to two hands simultaneously.
- Easy-to-use callback-based architecture.

## Requirements

To run this project, ensure you have the following installed:

- Python 3.8+
- OpenCV
- MediaPipe

### Install Required Libraries

Install the required libraries using pip:
```bash
pip install mediapipe opencv-python
```

## File Structure

- `main.py`: The main script for hand detection.
- `hand_landmarker.task`: The pre-trained hand landmark detection model (update the path in the script).

## How It Works

1. The program captures video from your webcam.
2. It uses the MediaPipe `HandLandmarker` model to detect hand landmarks in the video feed.
3. Detected landmarks are drawn on the video feed as green dots and connected by blue lines to represent the skeletal structure of the hand.
4. The processed video feed is displayed in a window.

## Running the Script

1. Download the `hand_landmarker.task` model file from MediaPipe's official repository or the provided link.
2. Update the `model_path` variable in the script to point to the correct location of the `hand_landmarker.task` file.
3. Run the script:

4. Press the `q` key to exit the program.

## Customization

- **Number of Hands:** Change the `num_hands` parameter in the `HandLandmarkerOptions` to adjust the maximum number of hands to detect.
- **Landmark Appearance:** Modify the `cv2.circle` and `cv2.line` parameters to change the visual style of landmarks and connections.


