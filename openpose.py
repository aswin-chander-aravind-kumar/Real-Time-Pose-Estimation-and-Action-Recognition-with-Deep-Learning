import cv2 as cv
import numpy as np
import time

BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

image_width = 600
image_height = 600

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

threshold = 0.2

img = cv.imread('C:/Users/shrut/OneDrive/Desktop/Final Project - CV/img.jpg')

photo_height, photo_width, _ = img.shape

# Prepare the frame to be fed to the network
inpBlob = cv.dnn.blobFromImage(img, 1.0, (image_width, image_height),
                               (127.5, 127.5, 127.5), swapRB=True, crop=False)

# Set the prepared object as the input blob of the network
net.setInput(inpBlob)

start_time = cv.getTickCount()  # Get the current Tick Count
out = net.forward()
end_time = cv.getTickCount()

# Compute the time taken for the forward pass (inference)
time_taken = (end_time - start_time) / cv.getTickFrequency()

out = out[:, :19, :, :]

assert(len(BODY_PARTS) == out.shape[1])

points = []
for i in range(len(BODY_PARTS)):
    heatMap = out[0, i, :, :]
    _, conf, _, point = cv.minMaxLoc(heatMap)
    x = (photo_width * point[0]) / out.shape[3]
    y = (photo_height * point[1]) / out.shape[2]
    points.append((int(x), int(y)) if conf > threshold else None)

for pair in POSE_PAIRS:
    partFrom = BODY_PARTS[pair[0]]
    partTo = BODY_PARTS[pair[1]]
    if points[partFrom] and points[partTo]:
        cv.line(img, points[partFrom], points[partTo], (0, 255, 0), 3)
        cv.ellipse(img, points[partFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        cv.ellipse(img, points[partTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

# Display the time taken for pose estimation on the image
text = f"Inference time: {time_taken:.2f} seconds"
cv.putText(img, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

cv.imshow("Pose", img)

def process_video(video_path, model_path):
    # Load the neural network
    net = cv.dnn.readNetFromTensorflow(model_path)

    # Open video file
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare the frame to be fed to the network
        inpBlob = cv.dnn.blobFromImage(frame, 1.0, (368, 368),
                                       (127.5, 127.5, 127.5), swapRB=True, crop=False)
        net.setInput(inpBlob)

        # Measure inference time
        start_time = time.time()
        out = net.forward()
        time_taken = time.time() - start_time

        # Assuming output shape and BODY_PARTS index are matching
        points = []
        for i in range(len(BODY_PARTS)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = int((frame.shape[1] * point[0]) / out.shape[3])
            y = int((frame.shape[0] * point[1]) / out.shape[2])
            points.append((x, y) if conf > threshold else None)

        # Draw detected points and lines
        for pair in POSE_PAIRS:
            partFrom = BODY_PARTS[pair[0]]
            partTo = BODY_PARTS[pair[1]]
            if points[partFrom] and points[partTo]:
                cv.line(frame, points[partFrom], points[partTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[partFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[partTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        # Display the frame with the drawn pose
        cv.putText(frame, f"Inference time: {time_taken:.2f} seconds", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        cv.imshow('Pose Estimation Video', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

# Usage
video_path = 'C:/Users/shrut/OneDrive/Desktop/Final Project - CV/video.mp4'  # Path to your video file
model_path = 'C:/Users/shrut/OneDrive/Desktop/Final Project - CV/graph_opt.pb'  # Path to your OpenPose model file
process_video(video_path, model_path)

# Wait for a key press and close all windows
cv.waitKey(0)
cv.destroyAllWindows()
