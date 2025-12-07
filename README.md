Here is the `README.md` file code for your GitHub repository. It includes sections for both parts of your pipeline (SAR Restoration and YOLO Detection) with specific placeholders left for your output screenshots.


# 🛰️ Military Asset Detection & SAR Image Restoration Pipeline

## 🚀 Overview
This repository hosts a comprehensive computer vision pipeline tailored for military intelligence and satellite imagery analysis. The solution is divided into two core modules:

1.  **SAR Image Restoration:** A hybrid signal processing and deep learning approach to remove speckle noise and enhance Synthetic Aperture Radar (SAR) images.
2.  **Military Object Detection:** A custom-trained YOLOv8 model optimized to detect and classify specific military assets (e.g., Tanks, Jets, Submarines) with high precision.

---

## 🛠️ Part 1: SAR Image Restoration (Denoising)

SAR images provide critical data in all weather conditions but often suffer from **speckle noise**, which degrades image quality. This module implements a restoration pipeline to clean these images for better interpretability.

### 🧠 Methodology
* **Noise Simulation:** Synthetic speckle noise (following a Gamma distribution) is introduced to training images to simulate real-world interference.
* **Matched Filter (MF):** A traditional Gaussian-based matched filter is applied first to smooth the noisy data.
* **Deep Learning Refinement (DnCNN):** A Denoising Convolutional Neural Network (DnCNN) with residual learning is trained on the MF outputs to subtract the remaining noise and recover fine details.

### 📷 Visual Results

**1. Noisy Input SAR Image:**
<img width="615" height="609" alt="Screenshot 2025-11-28 185549" src="https://github.com/user-attachments/assets/4a124b97-77d4-4beb-aca3-c6a9f2fcbd56" />


**2. Intermediate Matched Filter Output:**

<img width="610" height="606" alt="Screenshot 2025-11-28 185621" src="https://github.com/user-attachments/assets/80f4c624-d024-447c-a7bc-d649a9df632b" />

**3. Final Denoised Output (DnCNN):**


<img width="600" height="606" alt="Screenshot 2025-11-28 185633" src="https://github.com/user-attachments/assets/115ad018-8b07-4ea3-9dda-45526cedf661" />

---

## 🎯 Part 2: Military Object Detection

Once the imagery is clean (or using optical satellite data), the next step is identifying assets. This module uses **Ultralytics YOLOv8**, fine-tuned on a custom military dataset.

### ⚙️ Model Configuration
* **Model Architecture:** YOLOv8s (Small)
* **Epochs:** 100
* **Image Size:** 640x640
* **Classes (10):** `Ship`, `Helicopter`, `Tank`, `Fighter_jet`, `Submarine`, `Jeep`, `Truck`, `Bridge`, `Bunker`, `Helipad`.

### 📊 Performance Metrics
The model demonstrates strong performance on distinct assets, achieving an overall **mAP50 of 0.555**.

| Class         | Precision | Recall | mAP50  |
|---------------|-----------|--------|--------|
| **Helicopter**| 0.901     | 1.00   | 0.995  |
| **Tank** | 0.414     | 1.00   | 0.995  |
| **Fighter Jet**| 0.737    | 0.929  | 0.877  |
| **All Classes**| **0.724**| **0.448**| **0.555**|

### 📷 Detection Outputs

**Detection Result 1 (e.g., Ships & Bridges):**

<img width="1011" height="568" alt="Screenshot 2025-11-28 191259" src="https://github.com/user-attachments/assets/65a09648-f3dd-4125-ba41-002baae0c5f3" />


**Detection Result 2 (e.g., Tanks & Helicopters):**

<img width="486" height="454" alt="Screenshot 2025-11-16 174616" src="https://github.com/user-attachments/assets/b818a85c-0c75-418e-be58-44a10c948d43" />


---

## 🆚 Comparison: Pre-trained vs. Custom Model

Standard pre-trained models (trained on COCO) often fail to detect specific military objects due to domain shift.

* **Pre-trained YOLOv8:** Failed to detect assets in test samples.
* **Custom Military YOLOv8:** Successfully detected **11 Ships and 1 Bridge** in the same sample.

**Side-by-Side Comparison:**

<img width="990" height="560" alt="Screenshot 2025-11-28 191047" src="https://github.com/user-attachments/assets/03ce6089-c1d0-45c9-b00e-3eb285b9dbde" />

<img width="1011" height="568" alt="Screenshot 2025-11-28 191259" src="https://github.com/user-attachments/assets/4a3e9e0e-96cf-42ec-bcf1-5f123a7ed0f1" />
---


## 💻 Installation & Usage

### Dependencies
Ensure you have Python 3.x installed along with the following libraries:

pip install torch torchvision ultralytics opencv-python scikit-image matplotlib scipy


### Running the Restoration Pipeline

1.  Navigate to `SAR_IMAGE_RESTORATION.ipynb`.
2.  Input your raw SAR image path.
3.  Run the training loop to denoise and visualize the output.

### Running Object Detection

1.  Navigate to `Ship_Detection_in_Satellite_Image_with_YOLO.ipynb`.
2.  Load the custom weights:
    ```python
    from ultralytics import YOLO
    model = YOLO('runs/detect/train3/weights/best.pt')
    ```
3.  Run inference on new satellite images:
    ```python
    results = model.predict(source='path/to/your/image.jpg', save=True)
    ```


-----

## 🔮 Future Improvements

  * **Dataset Balancing:** Collect more data for underrepresented classes like `Jeep` and `Truck` to improve recall.
  * **Real-Time Optimization:** Optimize the pipeline for real-time video inference on edge devices (e.g., drones).
  * **Model Upscaling:** Experiment with YOLOv8-Large or Transformer-based backbones for higher accuracy on small objects.

<!-- end list -->


