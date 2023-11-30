import cv2
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns

def process_frame_with_tensorflow(frame, background, model, color_threshold=50):
    frame = cv2.resize(frame, (640, 480))
    background = cv2.resize(background, (640, 480))

   
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    # Perform color segmentation using KMeans clustering
    pixels = frame_rgb.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1).fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]

    # Define upper and lower bounds based on the dominant color
    lower_bound = np.array(dominant_color - color_threshold, dtype=np.uint8)
    upper_bound = np.array(dominant_color + color_threshold, dtype=np.uint8)

    # Create a mask for the dominant color
    mask = cv2.inRange(frame_rgb, lower_bound, upper_bound)

    # Apply the mask to get the foreground object
    result = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask)

    difference = frame_rgb - result
    processed_frame = np.where(mask[:, :, np.newaxis] != 0, background_rgb, difference)

    return cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

def main():
    
    root = tk.Tk()
    root.withdraw()
    background_path = filedialog.askopenfilename(title="Select Background Image")

    video_capture = cv2.VideoCapture(0)
    background_image = cv2.imread(background_path)

    # model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

    while True:
        ret, current_frame = video_capture.read()

        if not ret:
            break

        processed_frame = process_frame_with_tensorflow(current_frame, background_image, model=None)

        cv2.imshow("Video", current_frame)
        cv2.imshow("Processed Frame", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
