import cv2
import numpy as np
import os
import time

frame_folder = "./"
frames = sorted([f for f in os.listdir(frame_folder) if f.endswith(".png")])

timings = []

# Define color bounds for object detection
lower_bound = np.array([200, 200, 200])
upper_bound = np.array([255, 255, 255])

# Minimum area to be considered a valid object
min_area = 500

total_objects_detected = 0

# Initialize variables for video writing
video_output_path = "output_video.mp4"
frame_width, frame_height = None, None
video_writer = None

for frame_name in frames:
    frame_path = os.path.join(frame_folder, frame_name)
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"Failed to load {frame_name}")
        continue

    # Get frame dimensions and initialize video writer
    if video_writer is None:
        frame_height, frame_width = frame.shape[:2]
        video_writer = cv2.VideoWriter(
            video_output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),  # Codec for MP4 format
            30,  # Frame rate
            (frame_width, frame_height)
        )

    print(f"\nProcessing {frame_name} (Resolution: {frame_width}x{frame_height})")

    start_time = time.time()

    # Create mask for objects in the specified color range
    mask = cv2.inRange(frame, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count valid objects based on minimum area
    object_count = sum(1 for contour in contours if cv2.contourArea(contour) > min_area)

    end_time = time.time()
    process_time = end_time - start_time
    timings.append(process_time)
    total_objects_detected += object_count

    print(f"{frame_name} processed in {process_time:.4f} seconds")
    print(f"Detected {object_count} significant objects in {frame_name}")

    # Write the frame to the video
    video_writer.write(frame)

# Release the video writer
if video_writer:
    video_writer.release()

# Calculate and print average processing time
avg_time = sum(timings) / len(timings) if timings else 0
print(f"\nTotal objects detected across all frames: {total_objects_detected}")
print(f"Average processing time per frame: {avg_time:.4f} seconds")
print(f"Video saved to {video_output_path}")
