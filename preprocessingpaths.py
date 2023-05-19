import skimage.io
import skimage.color
import skimage.filters
import skimage.feature
import numpy as np
from collections import deque
from tqdm import tqdm

# Load video
video = skimage.io.imread(r'C:\Users\Dashe\Documents\physics 128L\03CloudChamber\CF4day1.tif')

# Convert to grayscale
video_gray = skimage.color.rgb2gray(video)

# Apply background subtraction
window_size = 60  # Window size for running average
video_subtracted = np.empty_like(video_gray)  # Initialize an array to hold the background-subtracted video

# Create a queue to hold the frames for the running average
queue = deque(maxlen=window_size)

for i in tqdm(range(video_gray.shape[0]), desc="Background Subtraction"):
    queue.append(video_gray[i])  # Add the current frame to the queue
    
    # Calculate the running average and subtract it from the current frame
    running_average = np.mean(queue, axis=0)
    video_subtracted[i] = np.abs(video_gray[i] - running_average)

# Apply Gaussian blur
video_blurred = skimage.filters.gaussian(video_subtracted, sigma=1)

# Perform edge detection on each frame
video_edges = np.array([skimage.feature.canny(frame) for frame in tqdm(video_blurred, desc="Edge Detection")])

# Perform thresholding
thresh = skimage.filters.threshold_otsu(video_edges)
video_thresholded = video_edges > thresh

# Save the processed video
skimage.io.imsave(r'C:\Users\Dashe\Documents\physics 128L\03CloudChamber\processed_CF4.tif', video_thresholded)
