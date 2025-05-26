import cv2
import mediapipe as mp
import numpy as np
from joblib import load
import time
from collections import Counter
import subprocess
import os

def send_notification(title, message):
    """
    Sends a MacOS notification with the given title and message.
    """
    script = f'display notification "{message}" with title "{title}"'
    subprocess.run(["osascript", "-e", script])

def check_posture():
    # Gets the absolute path to this file so the shell script
    # can access the files listed
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
    time.sleep(1)  # Give camera 1 second to warm up
    start_time = time.time()

    # Store the predictions of all photos so it can 
    # get the majority of 3 seconds later
    predictions = []

    # Take pictures of the user for 3 seconds
    while cap.isOpened() and (time.time() - start_time < 3):
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the image because that's how it is in training
        frame = cv2.flip(frame, 1)

        # Use mediapipe to see if there's a person in the image
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # If mediapipe recognizes that there's a person
        if results.pose_landmarks:
            # Extract keypoints (33 * 4 = 132 values)
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

            # Guess whether or not the user's posture is good or bad
            # with the model
            X = np.array(keypoints).reshape(1, -1)
            pred_idx = model.predict(X)[0]
            pred_label = encoder.inverse_transform([pred_idx])[0]
            predictions.append(pred_label)

    cap.release()
    return predictions

# Get the array of predictions
predictions = check_posture()

# Print to see
print(predictions)

if predictions:
    most_common = Counter(predictions).most_common(1)[0]
    print(f"Most common posture: {most_common[0]} (appeared {most_common[1]} times)")
    
    # Send the user a MacOS notification if their posture 
    # is classified as bad
    if most_common[0] == "bad":
        send_notification("Posture Alert", "Incorrect Posture")
else:
    print("No posture could be detected within the time limit.")
