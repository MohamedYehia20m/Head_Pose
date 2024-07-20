# Drowsiness and Attention Detection using OpenCV and MediaPipe

This project detects drowsiness and inattentiveness in drivers using facial landmarks. It utilizes OpenCV for image processing and MediaPipe for facial landmark detection.

## Features

- **Eye Aspect Ratio (EAR) Calculation**: Detects drowsiness by measuring the EAR.
- **Yawn Aspect Ratio (YAR) Calculation**: Detects yawning by measuring the YAR.
- **Head Pose Estimation**: Detects inattentiveness by analyzing the head pose.
- **Driver State Detection**: Identifies driver states such as "Looking Forward", "Drowsy", and "Sleeping".

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-repo/drowsiness-detection.git
    cd drowsiness-detection
    ```

2. Install the required packages:
    ```sh
    pip install opencv-python mediapipe numpy
    ```

## Usage

1. Run the script:
    ```sh
    python main.py
    ```

2. The script will start the webcam and begin processing the video stream. Press `q` to quit.

## Code Overview

### Main Components

- **Face Mesh Initialization**:
    ```python
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    ```

- **Distance Calculation**:
    ```python
    def distance(p1, p2):
        return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))
    ```

- **EAR Calculation**:
    ```python
    def calculate_ear(eye_landmarks):
        # Calculate vertical and horizontal distances
        # Compute EAR
        return ear
    ```

- **YAR Calculation**:
    ```python
    def calculate_yawn_aspect_ratio(inner_mouth_landmarks):
        # Calculate width and height
        # Compute YAR
        return yar
    ```

- **Head Pose Estimation**:
    ```python
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, jac = cv2.Rodrigues(rot_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    ```

- **Driver State Detection**:
    ```python
    if avg_ear < ear_threshold:
        # Drowsiness detection
    if yar > yar_threshold:
        # Yawn detection
    if text == "Not Looking Forward":
        # Head pose detection
    ```

### Driver State Conditions

- **Drowsiness Detection**: Based on EAR and YAR thresholds.
- **Sleeping Detection**: Based on continuous drowsiness detection and head pose analysis.
- **Inattentiveness Detection**: Based on head pose angles.

### Output

- Bounding box around the face indicating driver state.
- Display text on the frame to indicate "Drowsiness Detected", "Sleeping Detected", or "Not Looking Forward".

## Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
