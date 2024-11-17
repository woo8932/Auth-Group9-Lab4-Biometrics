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

   
# Initialize lists for genuine and impostor scores
genuine_scores = []
impostor_scores = []

# Calculate similarity scores for TRAIN and TEST pairs
for (train_f_path, train_s_path) in train_pairs:
    kp_f, des_f = extract_orb_features(train_f_path)
    kp_s, des_s = extract_orb_features(train_s_path)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_f, des_s)

    # Similarity score is the number of matches (normalized)
    genuine_scores.append(len(matches))

# Generate impostor pairs (cross-pairing fingerprints)
for (test_f_path, test_s_path) in zip(test_f, reversed(test_s)):
    kp_f, des_f = extract_orb_features(test_f_path)
    kp_s, des_s = extract_orb_features(test_s_path)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_f, des_s)

    impostor_scores.append(len(matches))

def calculate_rates(genuine_scores, impostor_scores, threshold):
    # Calculate FRR (False Reject Rate)
    fr = sum(1 for score in genuine_scores if score < threshold)
    frr = fr / len(genuine_scores) if genuine_scores else 0

    # Calculate FAR (False Accept Rate)
    fa = sum(1 for score in impostor_scores if score >= threshold)
    far = fa / len(impostor_scores) if impostor_scores else 0

    return frr, far




all_scores = genuine_scores + impostor_scores
min_score = min(all_scores)
max_score = max(all_scores)
thresholds = np.linspace(min_score, max_score, 100)

print(f"Thresholds: Min={min(thresholds)}, Max={max(thresholds)}")

frrs = []
fars = []
for threshold in thresholds:
    frr, far = calculate_rates(genuine_scores, impostor_scores, threshold)
    frrs.append(frr)
    fars.append(far)

# Find EER (point where FRR = FAR)
eer_index = np.argmin(np.abs(np.array(frrs) - np.array(fars)))
eer = frrs[eer_index]
eer_threshold = thresholds[eer_index]

# Output results
print(f"EER: {eer:.4f} at threshold {eer_threshold:.4f}")
print(f"Max FRR: {max(frrs):.4f}, Min FRR: {min(frrs):.4f}, Avg FRR: {np.mean(frrs):.4f}")
print(f"Max FAR: {max(fars):.4f}, Min FAR: {min(fars):.4f}, Avg FAR: {np.mean(fars):.4f}")

# Plot the FRR and FAR
plt.figure(figsize=(8, 6))
plt.plot(thresholds, frrs, label='False Reject Rate (FRR)')
plt.plot(thresholds, fars, label='False Accept Rate (FAR)')
plt.axvline(eer_threshold, color='red', linestyle='--', label=f"EER Threshold: {eer_threshold:.4f}")
plt.xlabel('Threshold')
plt.ylabel('Rate')
plt.legend()
plt.title('FRR and FAR vs Threshold')
plt.show()
