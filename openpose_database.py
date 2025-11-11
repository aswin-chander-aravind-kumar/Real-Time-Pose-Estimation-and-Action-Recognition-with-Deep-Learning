import os
import cv2
import numpy as np
import requests
import zipfile
import matplotlib.pyplot as plt

# Define paths and URLs
MODEL_PATH = 'C:/Users/shrut/OneDrive/Desktop/Final Project - CV/graph_opt.pb'
ANNOTATIONS_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
EXTRACT_FOLDER = 'C:/Users/shrut/OneDrive/Desktop/Final Project - CV/coco_annotations'
VIDEO_PATH = 'C:/Users/shrut/OneDrive/Desktop/Final Project - CV/video.mp4'

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

def load_model(model_path):
    """Load the TensorFlow model from a file."""
    net = cv2.dnn.readNetFromTensorflow(model_path)
    return net

def process_frame(frame, net, threshold=0.1):
    """Process a single frame using OpenPose model."""
    blob = cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    points = []
    confidences = []  # List to store all confidences
    for i in range(len(BODY_PARTS)):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = int((point[0] * frame.shape[1]) / W)
        y = int((point[1] * frame.shape[0]) / H)
        
        if prob > threshold:
            cv2.circle(frame, (x, y), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            points.append((x, y))
            confidences.append(prob)  # Add the confidence score
        else:
            points.append(None)

    for pair in POSE_PAIRS:
        partFrom = BODY_PARTS[pair[0]]
        partTo = BODY_PARTS[pair[1]]
        if points[partFrom] and points[partTo]:
            cv2.line(frame, points[partFrom], points[partTo], (0, 255, 0), 3)

    return frame, confidences

def process_video(video_path, model_path):
    """Process the video and apply pose estimation on each frame."""
    net = load_model(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    all_confidences = []  # List to store confidence scores of all frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, confidences = process_frame(frame, net)
        all_confidences.extend(confidences)  # Add confidences of this frame
        cv2.imshow('Pose Estimation', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Plot the confidence scores
    plot_confidences(all_confidences)

    # Calculate and print the average confidence score
    average_confidence = np.mean(all_confidences) if all_confidences else 0
    print(f"Average Confidence Score: {average_confidence:.2f}")
    
    return all_confidences

def plot_confidences(confidences):
    average_confidence = np.mean(confidences) if confidences else 0
    plt.figure(figsize=(10, 5))
    plt.plot(confidences, label='Confidence per Frame')
    plt.axhline(y=average_confidence, color='r', linestyle='--', label=f'Average Confidence: {average_confidence:.2f}')
    plt.xlabel('Frame Number')
    plt.ylabel('Confidence Value')
    plt.title('OpenPose - Pose Estimation Confidence per Frame')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Download and extract COCO annotations if not already done
    download_and_extract_annotations(ANNOTATIONS_URL, EXTRACT_FOLDER)

    # Process the video and plot confidence scores
    process_video(VIDEO_PATH, MODEL_PATH)
