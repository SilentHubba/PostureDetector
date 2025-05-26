import cv2
import mediapipe as mp
import numpy as np
from joblib import load

# Load trained model and label encoder
model = load("posture_model.joblib")
encoder = load("label_encoder.joblib")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    label_text = "No pose detected"

    if results.pose_landmarks:
        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract keypoints (33 * 4 = 132 values)
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

        # Convert to numpy and reshape for prediction
        X = np.array(keypoints).reshape(1, -1)

        # Predict
        prediction = model.predict(X)[0]
        label = encoder.inverse_transform([prediction])[0]

        label_text = f"Posture: {label.upper()}"
        color = (0, 255, 0) if label == "good" else (0, 0, 255)
        cv2.putText(frame, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Posture Detection", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
