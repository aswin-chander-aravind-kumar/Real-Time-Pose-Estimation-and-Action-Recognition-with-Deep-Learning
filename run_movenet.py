import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import time

# Define connections between keypoints for the body
POSE_PAIRS = [
    # Upper Body
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    # Lower Body
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    # Body
    [5, 11], [6, 12]
]

# Load the TensorFlow Hub model outside of main to avoid reloading for every call
model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
movenet = model.signatures['serving_default']

def load_img(path_to_img):
    """Load and preprocess an image."""
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [192, 192])
    img = tf.cast(img, dtype=tf.int32)  # Directly cast to int32 without normalization
    img = img[tf.newaxis, :]
    return img

def run_pose_estimation(image_path):
    """Run pose estimation model to detect keypoints."""
    start_time = time.time()
    img = load_img(image_path)
    outputs = movenet(img)
    keypoints = outputs['output_0'].numpy()[0][0]
    elapsed_time = time.time() - start_time
    print(f"Time taken to compute pose: {elapsed_time:.3f} seconds")
    return keypoints

def draw_skeleton(image_path, keypoints, POSE_PAIRS, elapsed_time, confidence_threshold=0.1):
    """Draw skeleton on an image by connecting keypoints and print execution time."""
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    points = []
    
    # Draw keypoints
    for i, keypoint in enumerate(keypoints):
        ky, kx, conf = keypoint
        if conf > confidence_threshold:
            x = int(kx * w)
            y = int(ky * h)
            points.append((x, y))
            cv2.circle(img, (x, y), 6, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
        else:
            points.append(None)
    
    # Draw skeleton
    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        if points[partFrom] and points[partTo]:
            cv2.line(img, points[partFrom], points[partTo], (255, 0, 0), 3)
    
    # Print execution time on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Time: {elapsed_time:.3f}s", (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Pose", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def draw_skeleton_on_frame(frame, keypoints, POSE_PAIRS, confidence_threshold=0.1):
    """Draw skeleton on a video frame by connecting keypoints."""
    h, w, _ = frame.shape
    points = []
    
    # Draw keypoints
    for i, keypoint in enumerate(keypoints):
        ky, kx, conf = keypoint
        if conf > confidence_threshold:
            x = int(kx * w)
            y = int(ky * h)
            points.append((x, y))
            cv2.circle(frame, (x, y), 6, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
        else:
            points.append(None)
    
    # Draw skeleton
    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        if points[partFrom] and points[partTo]:
            cv2.line(frame, points[partFrom], points[partTo], (255, 0, 0), 3)
    
    return frame

# Function for pose estimation on video frames
def run_pose_estimation_video(frame):
    """Run pose estimation on a video frame."""
    start_time = time.time()
    # Convert colors from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize and cast the frame to match model expectations
    img_data = tf.image.resize(frame_rgb, [192, 192])
    img_data = tf.cast(img_data, dtype=tf.int32)
    img_data = img_data[tf.newaxis, :]  # Add batch dimension

    # Run model
    outputs = movenet(img_data)
    keypoints = outputs['output_0'].numpy()[0][0]
    elapsed_time = time.time() - start_time
    print(f"Time taken to compute pose: {elapsed_time:.3f} seconds")
    return keypoints, elapsed_time

def process_video(video_path):
    """Process a video file frame by frame to perform pose estimation."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, elapsed_time = run_pose_estimation_video(frame)
        frame = draw_skeleton_on_frame(frame, keypoints, POSE_PAIRS, confidence_threshold=0.1)
        
        # Show the elapsed time on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Time: {elapsed_time:.3f}s", (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Pose Estimation', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    image_path = 'C:/Users/shrut/OneDrive/Desktop/Final Project - CV/img.jpg'  # Replace with your image path
    start_time = time.time()
    keypoints = run_pose_estimation(image_path)
    elapsed_time = time.time() - start_time
    print(f"Time taken to compute pose: {elapsed_time:.3f} seconds")
    draw_skeleton(image_path, keypoints, POSE_PAIRS, elapsed_time)
    video_path = 'C:/Users/shrut/OneDrive/Desktop/Final Project - CV/video.mp4'  # Replace with your video path
    process_video(video_path)
    

if __name__ == "__main__":
    main()