# 😷 SmartMask Detector – Real-Time Face Mask Detection System

A computer vision-based system that detects if a person is wearing a face mask using a webcam or Raspberry Pi camera module. Useful for entry screening, public health monitoring, and automated alerts.

---

## 📦 Project Structure
```
SmartMaskDetector/
├── mask_detector.py            # Real-time webcam detection
├── train_model.py              # CNN training script
├── dataset/                    # Mask / No-mask images
├── models/
│   └── mask_model.h5           # Pre-trained CNN model
├── utils/
│   └── dataset_loader.py       # Dataset processing helpers
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 Features
- Real-time webcam detection
- Accurate CNN classifier trained on face mask datasets
- Buzzer / LED alert option via GPIO (for Raspberry Pi)
- Optionally logs detection timestamp and result

---

## 🛠 Hardware (Optional for Raspberry Pi version)
- Raspberry Pi 3/4
- Pi Camera Module or USB webcam
- Buzzer or LED (optional alert output)

---

## 🐍 Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```
Main libraries:
- OpenCV
- TensorFlow / Keras
- NumPy

---

## 🚀 Run Detection
```bash
python mask_detector.py
```
This will open webcam and display bounding boxes with labels `Mask` or `No Mask`.

---

## 🧪 Train Your Own Model (Optional)
```bash
python train_model.py
```
Make sure your `dataset/` folder contains `with_mask/` and `without_mask/` image folders.

---

## 📊 Output Example
On detection:
- Green label for `Mask`
- Red label for `No Mask`
- Optional logging to `detections_log.csv`

---

## 📄 License
MIT License – free for educational and commercial use.

---

## 🙋‍♂️ Author
Built by [Your Name] with TensorFlow, OpenCV, and Raspberry Pi (optional).

