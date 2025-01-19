import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

detected_landmarks = []

def result_callback(result, output_image, timestamp_ms):
    global detected_landmarks
    if result.hand_landmarks:
        detected_landmarks = result.hand_landmarks
    else:
        detected_landmarks = []

#landmarks and lines
def draw_landmarks_and_connections(image, landmarks):
    h, w, _ = image.shape
    try:
        # Draw points
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Green dots

        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            start_landmark = landmarks[start_idx]
            end_landmark = landmarks[end_idx]
            start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
            end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
            cv2.line(image, start_point, end_point, (255, 0, 0), 2)  # Blue lines
    except Exception as e:
        print(f"Error drawing landmarks and connections: {e}")

def main():
    global detected_landmarks
    model_path = r"C:\\Users\\DeLL\\OneDrive\\Documents\\coding\\python\\mediapipe\\hand_landmarker (3).task"  # Update this to the correct path
    try:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=result_callback,
        )

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access webcam")
            return

        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Frame capture failed")
                    break

                # Convert frame to RGB format
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to MediaPipe's Image format
                try:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                    landmarker.detect_async(mp_image, timestamp)
                except Exception as e:
                    print(f"Error in detection: {e}")

                # Draw landmarks on the frame
                for landmarks in detected_landmarks:
                    draw_landmarks_and_connections(frame, landmarks)

                cv2.imshow("Hand Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except Exception as e:
        print(f"Error initializing HandLandmarker: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
