import cv2
import mediapipe as mp
import numpy as np
import time


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize constants
ear_threshold = 0.25
yar_threshold = 0.4
drowsy_time_threshold = 5  # in seconds
sleeping_time_threshold = 10  # in seconds

# Initialize time tracking variables
drowsy_start_time = None
sleeping_start_time = None
not_looking_forward_start_time = None

# Initialize driver state
driver_state = 0


def distance(p1, p2):
    """ Calculate distance between two points
    :param p1: First Point
    :param p2: Second Point
    :return: Euclidean distance between the points. (Using only the x and y coordinates).
    """
    return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))


def calculate_ear(eye_landmarks):
    """
    Calculate the Eye Aspect Ratio (EAR) of the eye.
    :param eye_landmarks: List of eye landmarks.
    :return: Eye Aspect Ratio (EAR) of the eye.
    """
    if len(eye_landmarks) < 8:
        return None

    # Calculate vertical distances
    v1 = distance(eye_landmarks[1], eye_landmarks[7])  # vertical distance between top and bottom landmarks
    v2 = distance(eye_landmarks[2], eye_landmarks[6])  # vertical distance between top and bottom landmarks
    v3 = distance(eye_landmarks[3], eye_landmarks[5])  # vertical distance between top and bottom landmarks

    # Calculate horizontal distances
    h = distance(eye_landmarks[0], eye_landmarks[4])  # horizontal distance between left and right landmarks

    # Calculate eye aspect ratio
    ear = (v1 + v2 + v3) / (3 * h)

    return ear


right_eye_indices = [33, 160, 159, 158, 133, 153, 145, 144]
left_eye_indices = [263, 387, 386, 385, 362, 380, 374, 373]
inner_mouth_indices = [62, 12, 292, 15]


def calculate_yawn_aspect_ratio(inner_mouth_landmarks):
    """
    Calculate the Yawn Aspect Ratio (YAR) of the mouth.
    :param inner_mouth_landmarks: List of inner mouth landmarks.
    :return: Yawn Aspect Ratio (YAR) of the mouth.
    """
    if len(inner_mouth_landmarks) < 4:
        return None

    # Calculate width (distance between landmarks 292 and 62)
    width = distance(inner_mouth_landmarks[0], inner_mouth_landmarks[2])

    # Calculate height (distance between landmarks 15 and 12)
    height = distance(inner_mouth_landmarks[1], inner_mouth_landmarks[3])

    # Calculate yawn aspect ratio
    yar = height / width

    return yar



# Open the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    img_h, img_w, img_c = frame.shape
    face_3d = []
    face_2d = []

    # Convert the color space from BGR to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get the results from the face mesh model using the RGB image as input and store it in the results variable
    results = face_mesh.process(image_rgb)

    """
    The results variable contains the following attributes:
    - multi_face_landmarks: The 3D landmarks of the face.
    - multi_face_landmarks_2d: The 2D landmarks of the face.
    """
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                '''
                33: Left eye outer corner
                263: Right eye outer corner
                1: Nose tip
                61: Left mouth corner
                291: Right mouth corner
                199: Chin
                '''
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        """
        Camera Matrix
        The camera matrix is used to map the 3D points of the face to their 2D projections in image plane.
        The camera matrix is used to find the rotation and translation vectors that map the 3D points of the face to
        their 2D projections.
        """
        focal_length = 1 * img_w
        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                               [0, focal_length, img_w / 2],
                               [0, 0, 1]])
        """
        Distortion Coefficients
        The distortion coefficients are used to correct the radial and tangential lens distortion.
        The distortion coefficients are used to find the rotation and translation vectors that map the 3D points of the
        face to their 2D projections.
        """
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        '''
        cv2.solvePnP: Solves the Perspective-n-Point problem to find the rotation and translation vectors that map
        3D points (face_3d) to their corresponding 2D projections (face_2d) given the camera matrix and distortion
        coefficients.
        success: Boolean indicating if the function was successful.
        rot_vec: Rotation vector (Rodrigues vector) representing the rotation of the face.
        trans_vec: Translation vector representing the translation of the face.
        '''
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # Convert Rotation Vector to Rotation Matrix
        # The rotation matrix represents the rotation of the face in the 3D space
        rmat, jac = cv2.Rodrigues(rot_vec)

        '''
        cv2.RQDecomp3x3: Decomposes the rotation matrix into the Euler angles.
        angles: Euler angles representing the rotation of the face.
        mtxR: The decomposed rotation matrix.
        mtxQ: The decomposed matrix.
        Qx: The x-axis of the decomposed matrix.
        Qy: The y-axis of the decomposed matrix.
        Qz: The z-axis of the decomposed matrix.
        '''
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Convert angles from radians to degrees
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360

        if y < -10 or y > 10 or x < -10 or x > 10:
            text = "Not Looking Forward"
            if not not_looking_forward_start_time:
                not_looking_forward_start_time = time.time()
        else:
            text = "Looking Forward"
            not_looking_forward_start_time = None

        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
        left_eye_landmarks_2d = np.array([(lm.x * img_w, lm.y * img_h) for lm in left_eye_landmarks])
        left_ear = calculate_ear(left_eye_landmarks_2d)

        right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]
        right_eye_landmarks_2d = np.array([(lm.x * img_w, lm.y * img_h) for lm in right_eye_landmarks])
        right_ear = calculate_ear(right_eye_landmarks_2d)

        avg_ear = (left_ear + right_ear) / 2
        if avg_ear is not None:
            cv2.putText(frame, f'Avg EAR: {avg_ear: .2f}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        inner_mouth_landmarks = [face_landmarks.landmark[i] for i in inner_mouth_indices]
        inner_mouth_landmarks_2d = np.array([(lm.x * img_w, lm.y * img_h) for lm in inner_mouth_landmarks])
        yar = calculate_yawn_aspect_ratio(inner_mouth_landmarks_2d)
        if yar is not None:
            cv2.putText(frame, f'YAR: {yar: .2f}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        box_color = (0, 255, 0)  # Default color: Green
        
        # Detect drowsiness based on EAR
        if avg_ear < ear_threshold:
            driver_state = 1  # Drowsy
            # cv2.putText(frame, "Drowsiness Detected", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # box_color = (0, 255, 255)  # Red color if any condition is met
            if not drowsy_start_time:
                drowsy_start_time = time.time()
            elif time.time() - drowsy_start_time > drowsy_time_threshold:
                driver_state = 2  # Sleeping
                # cv2.putText(frame, "Sleeping Detected!", (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # box_color = (0, 0, 255)  # Red color if any condition is met
            if driver_state == 1:
                cv2.putText(frame, "Drowsiness Detected", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                box_color = (0, 255, 255)
            elif driver_state == 2:
                cv2.putText(frame, "Sleeping Detected!", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                box_color = (0, 0, 255)
        else:
            drowsy_start_time = None
        
        # Detect sleeping based on looking forward

        # Detect drowsiness based on YAR
        if yar > yar_threshold:
            driver_state = 1  # Drowsy
            cv2.putText(frame, "Yawn Detected", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            box_color = (0, 255, 255)  # Red color if any condition is met

        # Detect not looking forward
        if text == "Not Looking Forward":
            driver_state = 1  # Drowsy
            box_color = (0, 255, 255)  # Red color if any condition is met
            if not_looking_forward_start_time and time.time() - not_looking_forward_start_time > sleeping_time_threshold:
                driver_state = 2  # Sleeping
                cv2.putText(frame, "Sleeping Detected!", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                box_color = (0, 0, 255)  # Red color if any condition is met

        # Reset driver state to normal if no drowsy or sleeping conditions are met
        if box_color == (0, 255, 0):
            driver_state = 0

        # Determine the bounding box around the face with padding
        x_min = int(np.min(face_2d[:, 0]))
        y_min = int(np.min(face_2d[:, 1]))
        x_max = int(np.max(face_2d[:, 0]))
        y_max = int(np.max(face_2d[:, 1]))

        padding = 40  # Add some padding around the face
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - 80)
        x_max = min(img_w, x_max + padding)
        y_max = min(img_h, y_max)

        # Draw the bounding box around the face
        top_left = (x_min-padding, y_min-padding)
        bottom_right = (x_max + padding, y_max + padding)

        # Draw the box around the face
        cv2.rectangle(frame, top_left, bottom_right, box_color, 2)
        print(driver_state)
    cv2.imshow('Head Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
