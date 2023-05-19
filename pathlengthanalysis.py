import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import skimage.color
import trackpy as tp
import pandas as pd
from tqdm import tqdm
from scipy.stats import poisson

# Load video
video = skimage.io.imread(r'C:\Users\Dashe\Documents\physics 128L\03CloudChamber\processed_CF4.tif')

# Get the center of the video
center = np.array([video.shape[1], video.shape[2]]) / 2

# Perform Hough transform
print("Performing Hough transform...")
lines = [skimage.transform.probabilistic_hough_line(frame) for frame in tqdm(video, desc="Hough transform")]

# Measure lines
print("Measuring lines...")
lengths = []
for frame in tqdm(lines, desc="Measuring lines"):
    for line in frame:
        # Calculate the distance from the center of the video to the edge of the path
        # Here line[0] and line[1] are the endpoints of the line, we calculate the distance to each endpoint and take the minimum
        length = min(np.linalg.norm(np.array(line[0]) - center), np.linalg.norm(np.array(line[1]) - center))
        lengths.append(length)

# Convert pixels to mm
lengths_mm = np.array(lengths) / 7.2

# Identify points for each frame and combine all frames into a single DataFrame
print("Identifying points...")
points = []
for frame_num in tqdm(range(len(lines)), desc="Identifying points"):
    for line in lines[frame_num]:
        for point in line:
            points.append([frame_num] + list(point))
            
# Link trajectories
print("Linking trajectories...")
tracks = tp.link_df(pd.DataFrame(points, columns=['frame', 'y', 'x']), search_range=3, memory=3)

# Save tracks to csv
tracks.to_csv('tracks.csv', index=False)

# Calculate and print the average length
average_length = np.mean(lengths_mm)
print(f'The average length of the paths is {average_length} mm')

# Calculate histogram without plotting
bins = np.arange(0, max(lengths_mm)+0.5, 0.5)
counts, bin_edges = np.histogram(lengths_mm, bins=bins, density=True)

# Get the maximum y-value
max_y = max(counts)

# Plot histogram of lengths
plt.hist(lengths_mm, bins=bins, density=True, label='Data')

# Fit a Poisson distribution to the rounded path lengths
mu = np.round(average_length)
x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))
plt.plot(x, poisson.pmf(x, mu) * max_y / max(poisson.pmf(x, mu)), 'r-', label=f'Poisson fit (mu = {mu})')

plt.xlabel('Path Length (mm)')
plt.ylabel('Density')
plt.title('Histogram of Path Lengths')
plt.legend()
plt.show()

print('complete')
