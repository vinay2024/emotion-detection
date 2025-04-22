# predict_live_deepface.py
import cv2
from deepface import DeepFace
import numpy as np
import time

# --- Configuration ---pip
# Path to the Haar Cascade file for faster initial face detection (optional but recommended)
# If not using Haar, deepface will use its internal detector which might be slower in a loop
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
USE_HAAR_DETECTOR = True # Set to False to rely solely on deepface's internal detector

# --- Load Face Detector ---
if USE_HAAR_DETECTOR:
    try:
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        if face_cascade.empty():
            raise IOError(f"Could not load Haar Cascade classifier from {FACE_CASCADE_PATH}")
        print("Haar Cascade face detector loaded successfully.")
    except Exception as e:
        print(f"Error loading Haar Cascade, falling back to deepface detector: {e}")
        USE_HAAR_DETECTOR = False
else:
     print("Using deepface's internal face detector.")


# --- Start Video Capture ---
cap = cv2.VideoCapture(0) # 0 is typically the default webcam

if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

print("Starting video stream...")
print("Note: The first analysis might take longer as models are downloaded.")

# Frame counter for analyzing every N frames (optional performance tweak)
frame_count = 0
analyze_every_n_frames = 3 # Analyze every 3 frames

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    frame_count += 1
    if frame_count % analyze_every_n_frames != 0 and USE_HAAR_DETECTOR:
         # Optional: Still draw old boxes if not analyzing this frame?
         # Or just display the raw frame for speed.
         cv2.imshow('Emotion Detection (DeepFace) - Press Q to Exit', frame)
         if cv2.waitKey(1) & 0xFF == ord('q'):
             break
         continue # Skip analysis for this frame


    # --- Face Detection and Emotion Analysis ---
    try:
        if USE_HAAR_DETECTOR:
            # 1. Detect faces using Haar Cascade (faster initial detection)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                # 2. Extract face ROI (Region of Interest) - use COLOR image for DeepFace
                face_roi = frame[y:y + h, x:x + w]

                # Ensure the ROI is valid
                if face_roi.size == 0:
                    continue

                # 3. Analyze the *face ROI* using DeepFacepip
                # Pass the NumPy array directly
                # Specify 'emotion' as the only action for speed
                # Set enforce_detection=False because we already detected the face
                try:
                    analysis_result = DeepFace.analyze(
                        img_path=face_roi,
                        actions=['emotion'],
                        enforce_detection=False, # Don't re-detect face within the ROI
                        silent=True # Suppress progress bars within the loopsudo zypper install libgthread-2_0-0 libgthread-2_0-0-32bit
                    )
                    # analysis_result is a list of dictionaries, one per face (but we pass one ROI)
                    if analysis_result and len(analysis_result) > 0:
                         dominant_emotion = analysis_result[0]['dominant_emotion']
                         confidence = analysis_result[0]['emotion'][dominant_emotion] # Get confidence score

                         # Draw bounding box and label
                         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                         text = f"{dominant_emotion} ({confidence:.1f}%)"
                         text_y = y - 10 if y - 10 > 10 else y + 10
                         cv2.putText(frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                except ValueError as ve:
                     # DeepFace might still fail if the ROI isn't recognized internally
                     # print(f"DeepFace could not process ROI: {ve}")
                     # Optionally draw a different color box for faces Haar detected but DeepFace failed on
                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1) # Red box for failed analysis
                except Exception as e:
                     print(f"Error during DeepFace analysis: {e}")
                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)


        else:
            # Alternative: Let DeepFace handle both detection and analysis (can be slower in loop)
            # This analyzes the *entire frame*
            analysis_results = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=True, # Detect faces in the whole frame
                detector_backend='opencv', # or 'ssd', 'mtcnn', etc.
                silent=True
             )
            # Results contain info for all detected faces
            for result in analysis_results:
                 box = result['region'] # {'x': ..., 'y': ..., 'w': ..., 'h': ...}
                 x, y, w, h = box['x'], box['y'], box['w'], box['h']
                 dominant_emotion = result['dominant_emotion']
                 confidence = result['emotion'][dominant_emotion]

                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                 text = f"{dominant_emotion} ({confidence:.1f}%)"
                 text_y = y - 10 if y - 10 > 10 else y + 10
                 cv2.putText(frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    except Exception as e:
        # Catch errors if DeepFace fails on the whole frame (e.g., no faces found)
        # print(f"Error in frame analysis: {e}") # Can be verbose if no faces are common
        pass # Continue to next frame

    # Display the resulting frame
    cv2.imshow('Emotion Detection (DeepFace) - Press Q to Exit', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("Releasing video capture and destroying windows...")
cap.release()
cv2.destroyAllWindows()
print("Application exited.")