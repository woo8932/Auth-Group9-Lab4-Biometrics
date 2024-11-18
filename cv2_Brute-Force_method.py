# By Craig Woolley
# 11/17/2024 

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Base directory
base = 'png_txt'

train_f = []
train_s = []
test_f = []
test_s = []

# Sort the files into the correct arrays
for fig in os.listdir(base):
    fig_path = os.path.join(base, fig)
    all_files = sorted(os.listdir(fig_path))
    
    for f in all_files:
        path = os.path.join(fig_path, f)
        if f.endswith('.png'):
            if f.startswith('f'):
                index = int(f[1:5]) 
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

# Combine the f and s sets 
train_pairs = list(zip(train_f, train_s))
test_pairs = list(zip(test_f, test_s))

# Find the features using ORB
def extract_orb_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found or unreadable at {image_path}")
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

# Arrays for calculating FFR and FAR 
genuine_scores = []
impostor_scores = []

#  find the key points and distances then use the brute force matcher and add the number of matches to genuine score
for (train_f_path, train_s_path) in train_pairs:
    kp_f, des_f = extract_orb_features(train_f_path)
    kp_s, des_s = extract_orb_features(train_s_path)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_f, des_s)

    genuine_scores.append(len(matches))

# Do the same as above but for the test set and store it in impostor scores
for (test_f_path, test_s_path) in zip(test_f, reversed(test_s)):
    kp_f, des_f = extract_orb_features(test_f_path)
    kp_s, des_s = extract_orb_features(test_s_path)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_f, des_s)

    impostor_scores.append(len(matches))

# Calculates FAR FRR based on a threshold
def calculate_rates(genuine_scores, impostor_scores, threshold):
    # FRR
    fr = sum(1 for score in genuine_scores if score < threshold)
    frr = fr / len(genuine_scores) if genuine_scores else 0

    # FAR
    fa = sum(1 for score in impostor_scores if score >= threshold)
    far = fa / len(impostor_scores) if impostor_scores else 0

    return frr, far

# Gets the minimum and maximum scores based on all of the scores from test and train (genuince and impostor)
all_scores = genuine_scores + impostor_scores
min_score = min(all_scores)
max_score = max(all_scores)
thresholds = np.linspace(min_score, max_score, 100)

# Save the FRR and FAR values
frrs = []
fars = []

# Go through the thresholds to prepare to find the EER
for threshold in thresholds:
    frr, far = calculate_rates(genuine_scores, impostor_scores, threshold)
    frrs.append(frr)
    fars.append(far)

# Finds EER
eer_index = np.argmin(np.abs(np.array(frrs) - np.array(fars)))
eer = frrs[eer_index]
eer_threshold = thresholds[eer_index]

# Results
print(f"EER: {eer:.4f} at threshold {eer_threshold:.4f}")
print(f"Max FRR: {max(frrs):.4f}, Min FRR: {min(frrs):.4f}, Avg FRR: {np.mean(frrs):.4f}")
print(f"Max FAR: {max(fars):.4f}, Min FAR: {min(fars):.4f}, Avg FAR: {np.mean(fars):.4f}")

# Plot the results 
plt.figure(figsize=(8, 6))
plt.plot(thresholds, frrs, label='False Reject Rate (FRR)')
plt.plot(thresholds, fars, label='False Accept Rate (FAR)')
plt.axvline(eer_threshold, color='red', linestyle='--', label=f"EER Threshold: {eer_threshold:.4f}")
plt.xlabel('Threshold')
plt.ylabel('Rate')
plt.legend()
plt.title('FRR and FAR vs Threshold')
plt.show()
