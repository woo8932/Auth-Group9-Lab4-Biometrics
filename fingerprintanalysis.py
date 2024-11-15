import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Base directory
base = 'png_txt'

# Prepare train and test lists
train_f = []
train_s = []
test_f = []
test_s = []

# Separate into TRAIN and TEST based on filename
for fig in os.listdir(base):
    fig_path = os.path.join(base, fig)
    all_files = sorted(os.listdir(fig_path))
    
    for f in all_files:
        path = os.path.join(fig_path, f)
        if f.endswith('.png'):
            if f.startswith('f'):
                index = int(f[1:5])  # Get the image index, e.g., 0001
                if index <= 1500:
                    train_f.append(path)
                else:
                    test_f.append(path)
            elif f.startswith('s'):
                index = int(f[1:5])
                if index <= 1500:
                    train_s.append(path)
                else:
                    test_s.append(path)

# Ensure that we have the correct number of pairs
train_pairs = list(zip(train_f, train_s))
test_pairs = list(zip(test_f, test_s))

# Feature extraction function using ORB
def extract_orb_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found or unreadable at {image_path}")
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

results = []
# Process TRAIN pairs and calculate distances
for (train_f_path, train_s_path) in train_pairs:
    kp_f, des_f = extract_orb_features(train_f_path)
    kp_s, des_s = extract_orb_features(train_s_path)

    # Method 1: Brute-Force Matcher with ORB descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_f, des_s)
    results.append(len(matches))

print(results)