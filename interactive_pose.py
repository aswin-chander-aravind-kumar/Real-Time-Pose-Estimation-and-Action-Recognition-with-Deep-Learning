import cv2
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine
import numpy as np
from tensorflow.keras.layers import Input

# Load keypoints data from CSV
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    labels = df.pop('label').values
    features = df.iloc[:, 2:4].values  # Assuming the first two columns are not features

    # Convert labels to categorical or binary (depending on the data)
    unique_labels = np.unique(labels)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    if len(unique_labels) == 2:
        labels = np.array([label_to_index[label] for label in labels])  # For binary classification
    else:
        labels = to_categorical([label_to_index[label] for label in labels])  # For multi-class classification

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    num_samples = features.shape[0]
    num_features = features.shape[1]

    # Reshape for LSTM: Assuming each sample is one time step for simplicity
    features = features.reshape(num_samples, 1, num_features)

    return features, labels, unique_labels

def build_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Define the input shape explicitly here
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))  # Assume binary classification as implied
    return model


# Train the LSTM model
def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=10):
    # Select the correct loss function based on the number of classes
    loss = 'binary_crossentropy' if y_train.shape[1] == 1 else 'categorical_crossentropy'
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
    return history

# Evaluate and classify using the LSTM model
def evaluate_and_classify(model, features, unique_labels):
    predictions = model.predict(features)
    print("Raw predictions:", predictions)  # Debug: Show raw predictions

    # Manually setting the labels for binary classification
    # Since we know there's only one label from the CSV, we assume the other
    if len(unique_labels) == 1:
        unique_labels = ['not_' + unique_labels[0], unique_labels[0]]  # Create a fictitious opposite label

    # Binary classification with sigmoid output
    predicted_labels = [int(pred >= 0.5) for pred in predictions.flatten()]

    predicted_actions = [unique_labels[idx] for idx in predicted_labels]
    return predicted_actions








# Extract keypoints from the test video
def extract_keypoints_from_video(video_path, movenet):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame to extract keypoints
        img_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_data = tf.image.resize(img_data, [192, 192])
        img_data = tf.cast(img_data * 255, tf.int32)  # Ensure correct data type
        img_data = tf.expand_dims(img_data, 0)  # Add batch dimension

        outputs = movenet(img_data)
        poses = outputs['output_0'].numpy().squeeze(0)

        for pose in poses:
            keypoints = []
            for i in range(len(pose) // 3):
                x, y, conf = pose[i * 3], pose[i * 3 + 1], pose[i * 3 + 2]
                if conf >= 0.3:  # Confidence threshold
                    keypoints.append((x, y, conf))
            if keypoints:
                keypoints_list.append(keypoints)

    cap.release()

    return keypoints_list


# Ensure proper flattening of keypoints
# Flatten keypoints from the test video
def flatten_keypoints(test_keypoints):
    flattened_keypoints = []
    for frame in test_keypoints:
        for keypoint in frame:
            # Assuming each keypoint consists of x, y, and confidence
            flattened_keypoints.extend(keypoint[:3])
    return flattened_keypoints



def adjust_reshape(flattened_keypoints, expected_size):
    if len(flattened_keypoints) > expected_size:
        # Reduce to the expected number of features by selecting the first `expected_size` features
        reshaped_features = np.array(flattened_keypoints[:expected_size]).reshape(1, expected_size)
    else:
        # Pad with zeros to meet the expected feature count
        padding = [0] * (expected_size - len(flattened_keypoints))
        reshaped_features = np.array(flattened_keypoints + padding).reshape(1, expected_size)
    return reshaped_features


def check_input_dimensions(features, expected_input_shape):
    print("Actual feature shape:", features.shape)
    print("Expected feature shape:", expected_input_shape)

    if features.shape != expected_input_shape:
        raise ValueError("Feature dimension mismatch. Expected {}, got {}".format(expected_input_shape, features.shape))






# Main function to build, train, and classify with LSTM
if __name__ == "__main__":
    CSV_PATH = '/Users/aswinchanderaravindkumar/Desktop/Final Project - CV/keypoints.csv'  # Adjusted to use the uploaded CSV
    TEST_VIDEO_PATH = '/Users/aswinchanderaravindkumar/Desktop/Final Project - CV/Walking/testing_10.avi'

    # Load data and unique labels
    features, labels, unique_labels = load_data(CSV_PATH)
    print("Unique labels:", unique_labels)

    # Split into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Build and train the LSTM model
    input_shape = (1, 2)  # Correct input shape
    expected_input_shape = (1, 1, 2)

    lstm_model = build_lstm_model(input_shape, labels.shape[1])
    if lstm_model is None:
        print("Model was not created successfully.")
    else:
        history = train_lstm_model(lstm_model, X_train, y_train, X_test, y_test, epochs=10)
        print("Model training completed.")

        # Load the pose estimation model
        MODEL_DIRECTORY = '/Users/aswinchanderaravindkumar/Desktop/Final Project - CV/multiposemovenet'
        movenet = tf.saved_model.load(MODEL_DIRECTORY).signatures['serving_default']

        # Extract keypoints from the test video
        test_keypoints = extract_keypoints_from_video(TEST_VIDEO_PATH, movenet)

        if test_keypoints:
            # Flatten keypoints
            flattened_keypoints = [item for sublist in test_keypoints for item in sublist]  # Flatten keypoints
            expected_size = 2
            flattened_keypoints = flatten_keypoints(test_keypoints)
            print("Flattened keypoints size:", len(flattened_keypoints))
            print("Expected size:", expected_size)

            test_features = adjust_reshape(flattened_keypoints, expected_size)
            print(test_features)
            check_input_dimensions(test_features, input_shape)
            if test_features.ndim == 2:
                test_features = test_features.reshape(-1, 1, test_features.shape[1])
            predicted_actions = evaluate_and_classify(lstm_model, test_features, unique_labels)
            print("Predicted Actions:", predicted_actions)
        else:
            print("No keypoints detected in the video.")
