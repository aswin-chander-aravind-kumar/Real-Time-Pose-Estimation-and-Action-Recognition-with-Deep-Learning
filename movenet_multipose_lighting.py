import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import requests
import zipfile
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from tensorflow import keras
from scipy.spatial.distance import cosine

# Define the model directory and load the model correctly
MODEL_DIRECTORY = '/Users/aswinchanderaravindkumar/Desktop/Final Project - CV/multiposemovenet'
model = tf.saved_model.load(MODEL_DIRECTORY)
movenet = model.signatures['serving_default']

# Define COCO body parts and pairs for drawing
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
    "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "MidHip"], ["MidHip", "RHip"], ["RHip", "RKnee"],
    ["RKnee", "RAnkle"], ["MidHip", "LHip"], ["LHip", "LKnee"],
    ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

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

def process_frame(frame, high_confidence_threshold=0.3):
    img_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_data = tf.image.resize(img_data, [192, 192])
    img_data *= 255.0
    img_data = tf.cast(img_data, tf.int32)
    img_data = tf.expand_dims(img_data, axis=0)
    
    outputs = movenet(img_data)
    poses = outputs['output_0'].numpy().squeeze(0)
    
    keypoints_list = []
    confidences = []

    for pose in poses:
        keypoints = []
        for i in range(len(pose) // 3):
            x, y, conf = pose[i*3], pose[i*3+1], pose[i*3+2]
            if conf >= high_confidence_threshold:
                keypoints.append(np.array([x, y, conf]))
        if keypoints:
            keypoints_list.append(np.array(keypoints))
            confidences.extend([kp[2] for kp in keypoints])
    
    avg_confidence = np.mean(confidences) if confidences else 0
    print("Detected keypoints:", keypoints_list)  # Debug print
    return keypoints_list, avg_confidence



def draw_skeletons(frame, keypoints_list, confidence_threshold=0.1):
    h, w, _ = frame.shape
    print("Frame dimensions:", h, w)

    for keypoints in keypoints_list:
        points = [None] * len(BODY_PARTS)  # Ensure all points have a place
        for x, y, conf in keypoints:
            if conf > confidence_threshold:
                x, y = int(x * w), int(y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                for part_name, idx in BODY_PARTS.items():
                    if keypoints.shape[0] > idx:  # Check if the keypoint index exists
                        points[idx] = (x, y)
                print("Drawing point at:", x, y, "with confidence:", conf)

        # Check and draw lines
        for pair in POSE_PAIRS:
            partFrom = BODY_PARTS[pair[0]]
            partTo = BODY_PARTS[pair[1]]
            if points[partFrom] is not None and points[partTo] is not None:
                cv2.line(frame, points[partFrom], points[partTo], (255, 0, 0), 2)
                print(f"Drawing line between {pair[0]} and {pair[1]} from {points[partFrom]} to {points[partTo]}")

    return frame


def keypoints_csv(keypoints_list, csv_path, frame_number, label):
    # Header includes the label now
    if frame_number == 0:
        header = ['frame', 'label']
        for i in range(len(keypoints_list[0])):
            header.extend([f'x{i}', f'y{i}', f'conf{i}'])
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    # Write the keypoints data with the label
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        for keypoints in keypoints_list:
            row = [frame_number, label]
            for point in keypoints:
                row.extend(point)
            writer.writerow(row)

def process_video(video_path, csv_path):
    label = input("Enter the label for the action in the video: ")
    cap = cv2.VideoCapture(video_path)
    all_confidences = []
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Ensure process_frame returns two values: keypoints_list and avg_confidence
        keypoints_list, avg_confidence = process_frame(frame, high_confidence_threshold=0.5)
        all_confidences.append(avg_confidence)

        if keypoints_list:
            frame = draw_skeletons(frame, keypoints_list, confidence_threshold=0.1)
            cv2.imshow('Multipose - Pose Estimation', frame)
            keypoints_csv(keypoints_list, csv_path, frame_number, label)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

    # Print overall average confidence
    if all_confidences:
        overall_avg_confidence = np.mean([conf for conf in all_confidences if conf > 0])
        print(f"Overall Average Confidence: {overall_avg_confidence:.2f}")
        plt.figure(figsize=(10, 5))
        plt.plot(all_confidences, label='Confidence per frame')
        plt.axhline(y=overall_avg_confidence, color='r', linestyle='--', label=f'Average Confidence: {overall_avg_confidence:.2f}')
        plt.xlabel('Frame Number')
        plt.ylabel('Average Confidence')
        plt.title('Multipose-Average Confidence per Frame')
        plt.legend()
        plt.show()

def load_and_prepare_data(csv_path):
    """Load keypoints data from a CSV file and prepare it for the classifier."""
    data = pd.read_csv(csv_path)
    features = data.iloc[:, 2:]  # Assuming first two columns are frame and label
    labels = data['label']
    return features, labels

def classify_actions(csv_path, model_path):
    """Classify actions based on keypoints stored in CSV using a pre-trained model."""
    # Load and prepare data
    features, labels = load_and_prepare_data(csv_path)
    
    # Scale features if the model expects standardized input
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Load the trained model
    model = keras.models.load_model(model_path)

    # Make predictions
    predictions = model.predict(features)
    predicted_labels = np.argmax(predictions, axis=1)  # if the output is one-hot encoded

    # Evaluate the model (optional, if ground truth labels are available)
    accuracy = np.mean(predicted_labels == labels)
    print(f"Classification accuracy: {accuracy:.2%}")

    return predicted_labels, accuracy

def keypoints_to_features(all_keypoints):
    # Ensure all_keypoints contains valid keypoints
    filtered_keypoints = []
    for kp in all_keypoints:
        # Check if kp is a 2D array with at least 3 columns
        if isinstance(kp, np.ndarray) and kp.ndim == 2 and kp.shape[1] >= 3:
            if np.all(kp[:, 2] > 0.3):  # High-confidence keypoints
                filtered_keypoints.append(kp)

    if not filtered_keypoints:
        print("No valid keypoints found.")
        return np.zeros(54)  # Return zeros if no valid keypoints found

    # Flatten and average keypoints
    mean_features = np.mean(filtered_keypoints, axis=0).flatten()
    if len(mean_features) != 54:
        print("Invalid number of features after flattening:", len(mean_features))
        return np.zeros(54)  # Return zeros if invalid number of features

    return mean_features




# Load your labeled data
def load_labeled_data():
    # This should load your labeled keypoints data
    # For simplicity, let's assume it returns a dictionary of {'label': np.array(features)}
    return {
        'walking': np.random.rand(54),  # Example data
        'running': np.random.rand(54)
    }

labeled_data = load_labeled_data()

def classify_action(current_features, labeled_features):
    min_distance = float('inf')
    action_label = None

    # Ensure that the feature vectors are comparable
    current_features = np.array(current_features)
    if not np.any(current_features):  # Check if `current_features` is a zero vector
        print("Current features vector is zero, cannot classify action.")
        return action_label  # Could return a default action or None

    if current_features.shape[0] != 54:  # Ensure there are exactly 54 features
        print("Incorrect number of features. Expected 54, got ", current_features.shape[0])
        return action_label  # Could return a default action or None

    for label, features in labeled_features.items():
        if not np.any(features):  # Check if the labeled feature is a zero vector
            print(f"Labeled features for '{label}' is zero, cannot use for comparison.")
            continue
        distance = cosine(current_features, features)
        if distance < min_distance:
            min_distance = distance
            action_label = label

    return action_label


def process_and_classify_video(video_path, labeled_features):
    cap = cv2.VideoCapture(video_path)
    all_keypoints = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints_list, _ = process_frame(frame)
        if keypoints_list:
            all_keypoints.extend(keypoints_list)  # Ensure this is a list of NumPy arrays

        # Show frame with drawn skeletons
        frame_with_skeletons = draw_skeletons(frame.copy(), keypoints_list, confidence_threshold=0.1)
        cv2.imshow('Frame', frame_with_skeletons)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Convert all_keypoints to 2D NumPy arrays
    # This step helps ensure you're working with consistent data structures
    keypoint_arrays = [np.array(kp) for sublist in all_keypoints for kp in sublist]

    # Convert to features
    features = keypoints_to_features(keypoint_arrays)
    
    if features is None or len(features) != 54:
        print("Invalid feature length, expected 54. Cannot classify.")
        return None
    
    action = classify_action(features, labeled_features)
    print(f"Action classified as: {action}")
    return action


if __name__ == "__main__":
    ANNOTATIONS_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    EXTRACT_FOLDER = '/Users/aswinchanderaravindkumar/Desktop/Final Project - CV/coco_annotations'
    CSV_PATH = '/Users/aswinchanderaravindkumar/Desktop/Final Project - CV/keypoints.csv'
    MODEL_PATH= '/Users/aswinchanderaravindkumar/Desktop/Final Project - CV/multiposemovenet/saved_model.pb'

    download_and_extract_annotations(ANNOTATIONS_URL, EXTRACT_FOLDER)

    # Number of videos to process
    number_of_videos = int(input("Enter the number of videos to process: "))

    for i in range(number_of_videos):
        VIDEO_PATH = input(f"Enter the path for video {i+1}: ")
        process_video(VIDEO_PATH, CSV_PATH)
    
    print("Completed processing all videos.")

    video_path = input("Enter the path for the testing video: ")
    labeled_features = load_labeled_data()
    action = process_and_classify_video(video_path, labeled_features)
    print(f"Action classified as: {action}")
