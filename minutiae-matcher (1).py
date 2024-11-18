import os
import cv2
import numpy as np
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Define root path
root_path = os.path.join(os.path.dirname(__file__), 'png_txt')

# Helper functions
def load_image_pairs(folders):
    """Load reference and subject image pairs."""
    image_pairs = []
    for folder in folders:
        folder_path = os.path.join(root_path, folder)
        for filename in os.listdir(folder_path):
            if filename.startswith("f") and filename.endswith(".png"):
                pair_id = filename[1:]
                reference_path = os.path.join(folder_path, filename)
                subject_path = os.path.join(folder_path, f"s{pair_id}")
                if os.path.exists(subject_path):
                    image_pairs.append((reference_path, subject_path))
    return image_pairs


def preprocess_image(image):
    """Preprocess the fingerprint image."""
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image


def get_minutiae_points_from_grayscale(image, min_distance=7, edge_margin=10):
    """Extract minutiae points from a grayscale image."""
    if image is None:
        print("Warning: Image is None.")
        return []
    minutiae_points = []
    rows, cols = image.shape

    # Threshold to binary image
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Identify minutiae
    for x in range(edge_margin, rows - edge_margin):
        for y in range(edge_margin, cols - edge_margin):
            if binary_image[x, y] == 255:
                neighborhood = binary_image[x - 1:x + 2, y - 1:y + 2]
                white_pixels = np.sum(neighborhood == 255)

                # Classify as ending or bifurcation
                if white_pixels in [2, 3]:
                    minutiae_points.append((x, y, 'ending'))
                elif white_pixels == 4:
                    minutiae_points.append((x, y, 'bifurcation'))

    # Filter out close minutiae using KD-tree for better performance
    if minutiae_points:
        points_array = np.array([(p[0], p[1]) for p in minutiae_points])
        tree = cKDTree(points_array)
        filtered_points = []
        for idx, point in enumerate(points_array):
            distances, _ = tree.query(point, k=10, distance_upper_bound=min_distance)
            if len(distances[distances < min_distance]) <= 1:
                filtered_points.append(minutiae_points[idx])
        minutiae_points = filtered_points

    return minutiae_points


def match_minutiae(ref_minutiae, subj_minutiae, threshold=10):
    """Match minutiae points between reference and subject images."""
    matched_points = 0
    for rx, ry, rtype in ref_minutiae:
        for sx, sy, stype in subj_minutiae:
            if rtype == stype:  # Match type
                distance = np.hypot(rx - sx, ry - sy)  # Use np.hypot for efficiency
                if distance <= threshold:
                    matched_points += 1
                    break
    return matched_points


def process_pair(ref_path, subj_path, min_distance, match_threshold):
    """Process a single image pair for minutiae extraction and matching."""
    ref_image = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    subj_image = cv2.imread(subj_path, cv2.IMREAD_GRAYSCALE)

    if ref_image is None or subj_image is None:
        print(f"Error loading images: {ref_path}, {subj_path}")
        return None

    ref_image = preprocess_image(ref_image)
    subj_image = preprocess_image(subj_image)

    ref_minutiae = get_minutiae_points_from_grayscale(ref_image, min_distance)
    subj_minutiae = get_minutiae_points_from_grayscale(subj_image, min_distance)

    matched_points = match_minutiae(ref_minutiae, subj_minutiae, match_threshold)
    return ref_path, subj_path, matched_points, len(ref_minutiae), len(subj_minutiae)


def evaluate_test_set_parallel(image_pairs, min_distance, match_threshold, max_workers=4):
    """Evaluate a set of image pairs in parallel."""
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pair, ref, subj, min_distance, match_threshold): (ref, subj) for ref, subj in image_pairs}
        for future in as_completed(futures):
            ref, subj = futures[future]
            try:
                print(f"Processing pair: Reference = {ref}, Subject = {subj}")
                result = future.result()
                if result:  # Skip None results (error loading images)
                    print(f"Completed pair: Reference = {ref}, Subject = {subj}, Matched Points = {result[2]}")
                    results.append(result)
                else:
                    print(f"Failed to process pair: Reference = {ref}, Subject = {subj}")
            except Exception as e:
                print(f"Error processing pair: Reference = {ref}, Subject = {subj}, Error = {e}")
    return results


def calculate_error_rates(results, impostor_results, similarity_threshold):
    """Calculate FRR, FAR, max, min, average error rates, and EER."""
    false_rejects = 0
    total_matches = len(results)
    frr_values = []

    for _, _, matched_points, ref_count, subj_count in results:
        similarity = matched_points / max(ref_count, subj_count) if max(ref_count, subj_count) > 0 else 0
        frr_values.append(1 - similarity)  # FRR per pair
        if similarity < similarity_threshold:
            false_rejects += 1

    frr = false_rejects / total_matches if total_matches > 0 else 0
    frr_max = max(frr_values) if frr_values else 0
    frr_min = min(frr_values) if frr_values else 0
    frr_avg = sum(frr_values) / len(frr_values) if frr_values else 0

    false_accepts = 0
    total_impostors = len(impostor_results)
    far_values = []

    for _, _, matched_points, ref_count, subj_count in impostor_results:
        similarity = matched_points / max(ref_count, subj_count) if max(ref_count, subj_count) > 0 else 0
        far_values.append(similarity)  # FAR per pair
        if similarity >= similarity_threshold:
            false_accepts += 1

    far = false_accepts / total_impostors if total_impostors > 0 else 0
    far_max = max(far_values) if far_values else 0
    far_min = min(far_values) if far_values else 0
    far_avg = sum(far_values) / len(far_values) if far_values else 0

    eer = calculate_eer(frr_values, far_values)

    return {
        "frr": frr,
        "frr_max": frr_max,
        "frr_min": frr_min,
        "frr_avg": frr_avg,
        "far": far,
        "far_max": far_max,
        "far_min": far_min,
        "far_avg": far_avg,
        "eer": eer,
    }


def calculate_eer(frr_values, far_values):
    """Calculate the Equal Error Rate (EER)."""
    frr_values = sorted(frr_values)
    far_values = sorted(far_values)

    min_diff = float('inf')
    eer = 0
    for frr, far in zip(frr_values, far_values):
        diff = abs(frr - far)
        if diff < min_diff:
            min_diff = diff
            eer = (frr + far) / 2
    return eer


def plot_roc_curve(frr_values, far_values):
    """Plot ROC curve."""
    plt.figure()
    plt.plot(far_values, frr_values, color='blue', lw=2, label='ROC curve')
    plt.xlabel('False Accept Rate (FAR)')
    plt.ylabel('False Reject Rate (FRR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


def create_impostor_pairs(image_pairs):
    """Generate impostor pairs by shuffling subject images."""
    reference_images = [pair[0] for pair in image_pairs]
    subject_images = [pair[1] for pair in image_pairs]
    random.shuffle(subject_images)
    impostor_pairs = list(zip(reference_images, subject_images))

    # Ensure no reference image matches its true subject
    for ref, subj in impostor_pairs:
        while ref.split('/')[-1][1:] == subj.split('/')[-1][1:]:
            random.shuffle(subject_images)
            impostor_pairs = list(zip(reference_images, subject_images))

    return impostor_pairs


def tune_parameters_parallel(train_image_pairs, param_grid, max_workers=4):
    """Tune parameters in parallel."""
    tasks = []
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for min_distance in param_grid['min_distance']:
            for match_threshold in param_grid['match_threshold']:
                for similarity_threshold in param_grid['similarity_threshold']:
                    tasks.append(
                        executor.submit(
                            evaluate_and_report,
                            train_image_pairs,
                            min_distance,
                            match_threshold,
                            similarity_threshold
                        )
                    )
        for future in as_completed(tasks):
            results.append(future.result())

    best_config = min(results, key=lambda x: x[3])  # Based on FRR
    return best_config[:3], best_config[3], results


def evaluate_and_report(train_image_pairs, min_distance, match_threshold, similarity_threshold):
    """Evaluate a single parameter combination."""
    results = evaluate_test_set_parallel(train_image_pairs, min_distance, match_threshold)
    frr = calculate_error_rates(results, [], similarity_threshold)['frr']
    return min_distance, match_threshold, similarity_threshold, frr


def main():
    train_folders = [f"figs_{i}" for i in range(6)]
    test_folders = [f"figs_{i}" for i in range(6, 8)]

    train_image_pairs = load_image_pairs(train_folders)
    test_image_pairs = load_image_pairs(test_folders)

    print(f"Loaded {len(train_image_pairs)} TRAIN pairs and {len(test_image_pairs)} TEST pairs.")

    param_grid = {
        'min_distance': [5, 7, 9],
        'match_threshold': [8, 10, 12],
        'similarity_threshold': [0.1, 0.2]
    }

    #print("\nTuning parameters...")
    #best_config, best_frr, _ = tune_parameters_parallel(
    #    train_image_pairs, param_grid, max_workers=16
    #)

    #print(f"\nBest Parameters from TRAIN:")
    #print(f"  min_distance: {best_config[0]}")
    #print(f"  match_threshold: {best_config[1]}")
    #print(f"  similarity_threshold: {best_config[2]}")
    #print(f"  Best FRR: {best_frr:.2%}")

    impostor_pairs = create_impostor_pairs(test_image_pairs)

    best_config = [5, 8]

    print("\nEvaluating TEST set...")
    genuine_results = evaluate_test_set_parallel(
        test_image_pairs,
        best_config[0],
        best_config[1]
    )
    impostor_results = evaluate_test_set_parallel(
        impostor_pairs,
        best_config[0],
        best_config[1]
    )

    error_rates = calculate_error_rates(
        genuine_results, impostor_results, best_config[2]
    )

    print("\nTEST Set Results:")
    print(f"  False Reject Rate (FRR): {error_rates['frr']:.2%}")
    print(f"  False Accept Rate (FAR): {error_rates['far']:.2%}")
    print(f"  Equal Error Rate (EER): {error_rates['eer']:.2%}")

    plot_roc_curve(sorted(error_rates['frr_max']), sorted(error_rates['far_max']))


if __name__ == "__main__":
    main()
