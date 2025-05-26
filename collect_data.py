import cv2
import mediapipe as mp
import pandas as pd
import os

# Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Output file
csv_path = 'dataset.csv'
all_data = []

# Start webcam
cap = cv2.VideoCapture(0)
label = None  # "good" or "bad"

print("Press 'g' for GOOD posture, 'b' for BAD posture, 'q' to quit and save.")

# Loop through while the camera is on
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    # Use mediapipe to see where the person is
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # If mediapipe sees a person
    if results.pose_landmarks:
        # Draw it for the user to see in the window
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract 33 keypoints (each has x, y, z, visibility)
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

        # Display current label
        if label is not None:
            cv2.putText(frame, f"Label: {label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save labeled data
        if label in ['good', 'bad']:
            keypoints.append(label)
            all_data.append(keypoints)

    # Show the image in a window for the user to see
    cv2.imshow("Posture Collector", frame)

    # Get user input
    # This works if you hold it down, so you don't 
    # have to continuously click 'b' or 'g' for images
    key = cv2.waitKey(10)
    if key == ord('g'):
        label = 'good'
    elif key == ord('b'):
        label = 'bad'
    elif key == ord('q'):
        break

# Close the camera and window
cap.release()
cv2.destroyAllWindows()
pose.close()

# Save to CSV
columns = [f"{coord}_{i}" for i in range(33) for coord in ['x', 'y', 'z', 'vis']] + ['label']
df = pd.DataFrame(all_data, columns=columns)
df.to_csv(csv_path, index=False, mode='a', header=False)

print(f"Saved {len(all_data)} samples to {csv_path}")
