import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import os
import requests
import zipfile
import matplotlib.pyplot as plt
import numpy as np


# Define paths and URLs
ANNOTATIONS_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
EXTRACT_FOLDER = 'C:/Users/shrut/OneDrive/Desktop/Final Project - CV/coco_annotations'
VIDEO_PATH = 'C:/Users/shrut/OneDrive/Desktop/Final Project - CV/video.mp4'
MODEL_URL = 'https://tfhub.dev/google/movenet/singlepose/lightning/4'

# Load the TensorFlow Hub model
model = hub.load(MODEL_URL)
movenet = model.signatures['serving_default']

def download_and_extract_annotations(url, extract_path):
    """Download and extract COCO annotations."""
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(extract_path, 'annotations.zip'), 'wb') as f:
            f.write(response.content)
        with zipfile.ZipFile(os.path.join(extract_path, 'annotations.zip'), 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        os.remove(os.path.join(extract_path, 'annotations.zip'))
    else:
        raise Exception(f"Failed to download file: Status code {response.status_code}")

def process_frame(frame):
    """Process a single frame using MoveNet."""
    # Convert the frame to RGB from BGR (OpenCV default)
    img_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize the image
    img_data = tf.image.resize(img_data, [192, 192])
    
    # Normalize the image to range [0,1] and cast to float32
    img_data = img_data / 255.0
    img_data = tf.cast(img_data, dtype=tf.float32)

    # MoveNet expects int32 inputs, so convert the normalized data to int32
    img_data = tf.cast(img_data * 255, dtype=tf.int32)

    # Add batch dimension
    img_data = tf.expand_dims(img_data, axis=0)

    # Model inference
    outputs = movenet(img_data)

    # Extract keypoints from the model output
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]
    return keypoints



def draw_skeleton(frame, keypoints, pose_pairs, confidence_threshold=0.1):
    """Draw the skeleton on the frame using only high-confidence keypoints."""
    h, w, _ = frame.shape
    points = []
    confidences = []

    # Iterate over all keypoints and process only those above the confidence threshold
    for i, (y, x, conf) in enumerate(keypoints):
        if conf > confidence_threshold:
            x = int(x * w)
            y = int(y * h)
            points.append((x, y))
            confidences.append(conf)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        else:
            points.append(None)

    # Draw connections only between high-confidence points
    for partFrom, partTo in pose_pairs:
        if points[partFrom] is not None and points[partTo] is not None:
            cv2.line(frame, points[partFrom], points[partTo], (0, 255, 0), 3)

    return frame, confidences

def smooth_keypoints(prev_keypoints, curr_keypoints, alpha=0.5):
    """Smooth keypoints using exponential moving average."""
    if prev_keypoints is None:
        return curr_keypoints
    return alpha * curr_keypoints + (1 - alpha) * prev_keypoints

def process_video(video_path, pose_pairs):
    cap = cv2.VideoCapture(video_path)
    all_confidences = []
    prev_keypoints = None  # Initialize previous keypoints storage for smoothing

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = process_frame(frame)
        keypoints = smooth_keypoints(prev_keypoints, keypoints)  # Apply smoothing
        prev_keypoints = keypoints  # Update previous keypoints
        frame, frame_confidences = draw_skeleton(frame, keypoints, pose_pairs)
        all_confidences.extend(frame_confidences)  # Collect confidence scores of only drawn keypoints

        cv2.imshow('Singlepose-Lightning-Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if all_confidences:
        avg_confidence = np.mean(all_confidences)
        print(f"Average Confidence Score: {avg_confidence:.2f}")
        plot_confidences(all_confidences, avg_confidence)


def plot_confidences(confidences, average_confidence):
    plt.figure(figsize=(10, 5))
    plt.plot(confidences, label='Confidence per Frame')
    plt.axhline(y=average_confidence, color='r', linestyle='--', label=f'Average Confidence: {average_confidence:.2f}')
    plt.xlabel('Frame Number')
    plt.ylabel('Confidence Value')
    plt.title('Singlepose-Lightning - Pose Estimation Confidence per Frame')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    download_and_extract_annotations(ANNOTATIONS_URL, EXTRACT_FOLDER)
    pose_pairs = [
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
        [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
        [5, 11], [6, 12]
    ]
    process_video(VIDEO_PATH, pose_pairs)