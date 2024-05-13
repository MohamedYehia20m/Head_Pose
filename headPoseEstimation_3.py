import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
eye_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=8)  # thicker green circle


def calculate_ear(eye_landmarks):
    if len(eye_landmarks) < 2:
        return None

    # Calculate vertical distances
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])  # vertical distance between top and bottom landmarks
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])  # vertical distance between top and bottom landmarks

    # Calculate horizontal distances
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])  # horizontal distance between left and right landmarks

    # Calculate eye aspect ratio
    ear = (v1 + v2) / (2 * h)

    return ear


def calculate_yawn_aspect_ratio(inner_mouth_landmarks):
    if len(inner_mouth_landmarks) < 2:
        return None

    # Calculate width (distance between landmarks 292 and 62)
    width = np.linalg.norm(inner_mouth_landmarks[0] - inner_mouth_landmarks[2])

    # Calculate height (distance between landmarks 15 and 12)
    height = np.linalg.norm(inner_mouth_landmarks[1] - inner_mouth_landmarks[3])

    # Calculate yawn aspect ratio
    yar = height / width

    return yar


# Open video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the color space from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get the result
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Define the indices that outline the left eye region
            left_eye_indices = [33, 160, 158, 133, 153, 144]

            # Draw circles around left eye landmarks and label them with indices
            for idx in left_eye_indices:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw green circles with thickness
                cv2.putText(frame, str(idx), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)

            # Calculate and display EAR for the left eye
            left_eye_landmarks = np.array(
                [[landmark.x * frame.shape[1], landmark.y * frame.shape[0], landmark.z * 3000]
                 for idx, landmark in enumerate(face_landmarks.landmark)
                 if idx in left_eye_indices])

            left_ear = calculate_ear(left_eye_landmarks)
            if left_ear is not None:
                cv2.putText(frame, f'Left EAR: {left_ear:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

            # Define the indices that outline the right eye region
            right_eye_indices = [362, 385, 387, 263, 373, 380]

            # Draw circles around right eye landmarks and label them with indices
            for idx in right_eye_indices:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw green circles with thickness
                cv2.putText(frame, str(idx), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)

            # Calculate and display EAR for the right eye
            right_eye_landmarks = np.array(
                [[landmark.x * frame.shape[1], landmark.y * frame.shape[0], landmark.z * 3000]
                 for idx, landmark in enumerate(face_landmarks.landmark)
                 if idx in right_eye_indices])

            right_ear = calculate_ear(right_eye_landmarks)
            if right_ear is not None:
                cv2.putText(frame, f'Right EAR: {right_ear:.2f}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

            # Define the indices that outline the inner mouth region (inner smile)
            inner_mouth_indices = [62,
                                   12,  # upper lip
                                   292,
                                   15]  # lower lip

            # Draw circles around inner mouth landmarks and label them with indices
            for idx in inner_mouth_indices:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # Draw blue circles with thickness
                cv2.putText(frame, str(idx), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)

            # Calculate and display Yawn Aspect Ratio
            inner_mouth_landmarks = np.array(
                [[landmark.x * frame.shape[1], landmark.y * frame.shape[0], landmark.z * 3000]
                 for idx, landmark in enumerate(face_landmarks.landmark)
                 if idx in inner_mouth_indices])

            yar = calculate_yawn_aspect_ratio(inner_mouth_landmarks)
            if yar is not None:
                cv2.putText(frame, f'Yawn Aspect Ratio: {yar:.2f}', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

            # Draw face mesh
            # mp_drawing.draw_landmarks(
            #    image=frame,
            #    landmark_list=face_landmarks,
            #    connections=mp_face_mesh.FACEMESH_CONTOURS,
            #    landmark_drawing_spec=drawing_spec,
            #    connection_drawing_spec=drawing_spec
            # )

    # Display the frame
    cv2.imshow('Face Mesh with Eye Aspect Ratio, Inner Mouth Landmarks, and Yawn Aspect Ratio', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
