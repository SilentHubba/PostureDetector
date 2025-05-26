import cv2
import mediapipe as mp
import numpy as np
from joblib import load
import time
from collections import Counter
import subprocess
import os

def send_notification(title, message):
    script = f'display notification "{message}" with title "{title}"'
    subprocess.run(["osascript", "-e", script])

def check_posture():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the model file
    model_path = os.path.join(script_dir, "posture_model.joblib")
    encoder_path = os.path.join(script_dir, "label_encoder.joblib")

    # Load trained model and label encoder
    model = load(model_path)
    encoder = load(encoder_path)

    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Start webcam
    cap = cv2.VideoCapture(0)
    time.sleep(1)  # Give camera time to warm up
    start_time = time.time()

    predictions = []

    while cap.isOpened() and (time.time() - start_time < 3):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # Extract keypoints (33 * 4 = 132 values)
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

            #if len(keypoints) == 132:
            X = np.array(keypoints).reshape(1, -1)
            pred_idx = model.predict(X)[0]
            pred_label = encoder.inverse_transform([pred_idx])[0]
            predictions.append(pred_label)

    cap.release()
    return predictions

#print(predictions)

predictions = check_posture()

print(predictions)

if predictions:
    most_common = Counter(predictions).most_common(1)[0]
    print(f"Most common posture: {most_common[0]} (appeared {most_common[1]} times)")
    if most_common[0] == "bad":
        send_notification("Posture Alert", "Incorrect Posture")
else:
    print("No posture could be detected within the time limit.")
