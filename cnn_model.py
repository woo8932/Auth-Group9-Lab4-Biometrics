import numpy as np
import os
import cv2
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

"""
    CNN Method for biometric lab using.  
    
    By: Juliet Meza

    Returns:
        Test Loss
        Test Accuracy
        ROC AUC Score
        EER per Class 
        FRR per Class (Min, Max, Avg)
        FRR per Class (Min, Max, Avg)
        Overall Average FRR
        Overall Average FAR
"""

# Constants
IMAGE_SIZE = 512  # Original size of the fingerprint images (512x512)
TRIMMED_SIZE = 480  # Removes the bottom 32 rows of white space from each image
NUM_CLASSES = 5  # Five classes (A=Arch, L=Left Loop, R=Right Loop, T=Tented Arch, W=Whorl)
CLASS_MAP = {'A': 0, 'L': 1, 'R': 2, 'T': 3, 'W': 4}  # Maps class labels to numeric values

# Data Loading Function
def load_nist_data(base_dir):
    """
    Loads fingerprint images and corresponding labels from the NIST dataset.
    - Normalizes images to values between 0 and 1 for neural network training.
    - Extracts labels based on 'Class' information in the .txt files.

    Returns:
        - images: Array of processed images
        - labels: Corresponding labels
        - filenames: Names of the image files
    """
    images = []
    labels = []
    filenames = []

    # Recursively load images and labels
    for subdir, _, files in os.walk(base_dir):
        for file in sorted(files):
            if file.endswith(".png"):
                image_path = os.path.join(subdir, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                trimmed_image = image[:TRIMMED_SIZE, :]  # Removes white space
                normalized_image = trimmed_image / 255.0  # Normalizes pixel values
                images.append(normalized_image)

                txt_path = os.path.join(subdir, file.replace(".png", ".txt"))
                try:
                    with open(txt_path, 'r') as f:
                        for line in f:
                            if line.startswith("Class:"):
                                class_label = line.split(":")[1].strip()  # Extracts label
                                labels.append(CLASS_MAP[class_label])
                                break
                except (IndexError, FileNotFoundError, KeyError):
                    print(f"Skipping file due to error: {txt_path}")
                    continue  # Skips problematic files

                filenames.append(file)

    return np.array(images), np.array(labels), filenames

def split_data(images, labels, filenames):
    """
    Splits the dataset into TRAIN and TEST sets based on file numbering.
    - TRAIN: Files numbered f0001-f1499
    - TEST: Files numbered f1500-f2000
    """
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for i, filename in enumerate(filenames):
        file_number = int(filename[1:5])  # Extracts numeric part of filename
        if file_number <= 1499:  # Train set
            train_images.append(images[i])
            train_labels.append(labels[i])
        elif 1500 <= file_number <= 2000:  # Test set
            test_images.append(images[i])
            test_labels.append(labels[i])

    return (
        np.array(train_images),
        np.array(train_labels),
        np.array(test_images),
        np.array(test_labels),
    )

def calculate_frr_far_eer(y_true, y_pred_probs):
    """
    Calculates False Reject Rate (FRR), False Accept Rate (FAR), and Equal Error Rate (EER).
    - y_true: Ground truth labels (one-hot encoded).
    - y_pred_probs: Predicted probabilities from the model.
    """
    frr_list = []  # To store FRR values
    far_list = []  # To store FAR values
    
    # Convert one-hot encoded true labels to integers
    y_true_int = np.argmax(y_true, axis=1)
    y_pred_int = np.argmax(y_pred_probs, axis=1)

    # Calculate ROC curve for each class
    for class_id in range(NUM_CLASSES):
        # Binary labels: 1 if the class matches, 0 otherwise
        binary_y_true = (y_true_int == class_id).astype(int)
        binary_y_pred = y_pred_probs[:, class_id]
        
        # Compute FPR (False Positive Rate), TPR (True Positive Rate), and thresholds
        fpr, tpr, thresholds = roc_curve(binary_y_true, binary_y_pred)
        
        # Calculate FRR = 1 - TPR
        fnr = 1 - tpr  # False Negative Rate = 1 - TPR
        
        # Find the threshold where FPR = FNR (approximation for EER)
        eer_threshold_index = np.argmin(np.abs(fpr - fnr))
        eer = fpr[eer_threshold_index]  # FPR and FNR are equal here
        
        # Store max, min, and average values of FRR and FAR
        frr_list.append((fnr.min(), fnr.max(), fnr.mean()))
        far_list.append((fpr.min(), fpr.max(), fpr.mean()))
        
        # Print EER for each class
        print(f"Class {class_id}: EER = {eer:.4f}, Threshold = {thresholds[eer_threshold_index]:.4f}")
    
    # Calculate overall average FRR and FAR across all classes
    avg_frr = np.mean([frr[2] for frr in frr_list])  # Mean of average FRR for all classes
    avg_far = np.mean([far[2] for far in far_list])  # Mean of average FAR for all classes

    return frr_list, far_list, avg_frr, avg_far

# Load Data
base_dir = "./NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt" # Make sure to point to correct directory
images, labels, filenames = load_nist_data(base_dir)
images = images[..., np.newaxis]  # Adds channel dimension for grayscale images
x_train, y_train, x_test, y_test = split_data(images, labels, filenames)

# One-hot encode the labels for neural network training
y_train = to_categorical(y_train, num_classes=len(CLASS_MAP))
y_test = to_categorical(y_test, num_classes=len(CLASS_MAP))

# CNN Model Definition
def create_cnn(input_shape, num_classes):
    """
    Defines a Convolutional Neural Network (CNN) for fingerprint classification.
    - Three convolutional blocks for feature extraction.
    - A dense layer with dropout for classification.
        - Dropout (60%) ensures regularization, reducing the risk of overfitting.
    Returns:
        - Compiled CNN model
    """
    input_layer = Input(shape=input_shape)
    # First convolutional block
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D((2, 2))(conv1)  # Reduces spatial dimensions by half
    # Second convolutional block
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    # Third convolutional block
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)
    # Fully connected layers
    flatten = Flatten()(pool3)
    dense1 = Dense(128, activation='relu')(flatten)  # 128 units for compact representation
    dropout = Dropout(0.6)(dense1)  # Prevents overfitting by randomly dropping nodes
    output_layer = Dense(num_classes, activation='softmax')(dropout)  # Output layer for classification
    return Model(inputs=input_layer, outputs=output_layer)


model = create_cnn(input_shape=(TRIMMED_SIZE, IMAGE_SIZE, 1), num_classes=NUM_CLASSES)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Adam optimizer for adaptive learning
model.summary()

"""
Adam Optimizer:
- Combines the benefits of momentum and adaptive learning rates.
- Handles noisy gradients well and converges faster.

Resource: 
https://www.geeksforgeeks.org/adam-optimizer/
https://keras.io/api/optimizers/adam/
"""


# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=20,  # Optimal balance between training time and generalization
    batch_size=32,  # Small batch size improves generalization
    callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Calculate ROC AUC Score for Multiclass
# https://www.evidentlyai.com/classification-metrics/explain-roc-curve#:~:text=The%20ROC%20AUC%20score%20is%20the%20area%20under%20the%20ROC,and%201%20indicates%20perfect%20performance.
def calculate_eer_multiclass(y_true, y_pred):
    """
    Calculates the ROC AUC score for multiclass classification.
    """
    roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')
    print(f"ROC AUC Score (Macro-Averaged): {roc_auc}")
    return roc_auc

# Get model predictions
y_pred = model.predict(x_test)

# Calculate and print ROC AUC Score
roc_auc = calculate_eer_multiclass(y_test, y_pred)
print(f"ROC AUC Score: {roc_auc}")

# Get FRR, FAR, and EER values
frr_list, far_list, avg_frr, avg_far = calculate_frr_far_eer(y_test, y_pred)

# Print results
print("\nFalse Reject Rates (FRR) for each class (min, max, avg):")
for i, frr in enumerate(frr_list):
    print(f"Class {i}: Min = {frr[0]:.4f}, Max = {frr[1]:.4f}, Avg = {frr[2]:.4f}")

print("\nFalse Accept Rates (FAR) for each class (min, max, avg):")
for i, far in enumerate(far_list):
    print(f"Class {i}: Min = {far[0]:.4f}, Max = {far[1]:.4f}, Avg = {far[2]:.4f}")

print(f"\nOverall Average FRR: {avg_frr:.4f}")
print(f"Overall Average FAR: {avg_far:.4f}")

# Save the model
model.save('fingerprint_cnn_model.h5')
print("Model saved as fingerprint_cnn_model.h5")
