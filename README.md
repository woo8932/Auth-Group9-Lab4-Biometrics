# **CNN Fingerprint Classification Script**
By: Juliet Meza

This is one method for the biometrics project. Its a single Convolutional Neural Network (CNN) that clasifying images on 5 different categories. 

---

## **Table of Contents**
1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Configuration](#configuration)
   - [Setting the Dataset Path](#setting-the-dataset-path)
4. [Running the Script](#running-the-script)
5. [Results](#results)

---

## **Requirements**

- Python 3.10
- TensorFlow 2.0+
- NumPy
- OpenCV
- Scikit-learn
- Conda for environment management

---

## **Setup**

### **Importing Conda Environment**
To set up the environment from the `environment.yml` file:

1. Create the environment:
   ```bash
   conda env create -f CNN_environment.yml
   ```

2. Activate the environment:
   ```bash
   conda activate csec472_lab4
   ```

## **Configuration**

### **Setting the Dataset Path**
The script requires the path to the dataset. Update the `base_dir` variable in the script:

```python
base_dir = "./NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt"
```

Replace the value of `base_dir` with the path to your dataset folder.

---

## **Running the Script**

1. Activate the Conda environment:
   ```bash
   conda activate csec472_lab4
   ```

2. Run the script:
   ```bash
   python3 cnn_model.py
   ```

3. Outputs:
   - Training and validation accuracy/loss are displayed during training.
   - Test accuracy, loss, and metrics (ROC AUC, FRR, FAR, EER) are printed in the terminal.

---

## **Results**

The script generates:
1. A trained model saved as `fingerprint_cnn_model.h5`.
2. Metrics printed in the terminal:
   - False Reject Rate (FRR)
   - False Accept Rate (FAR)
   - Equal Error Rate (EER)

