# 🤟 ASL Alphabet Recognition

Real-time American Sign Language (ASL) alphabet recognition using deep learning and computer vision.

## Features

- **Real-time Detection**: Live webcam feed with instant ASL letter recognition
- **Automatic Hand Detection**: Uses MediaPipe for precise hand tracking
- **Deep Learning Model**: MobileNetV2-based CNN for accurate classification
- **Web Interface**: Clean Streamlit UI for easy interaction
- **29 Classes**: Recognizes A-Z letters plus 'del', 'space', and 'nothing'


## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd asl-recognition
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup

1. Download the ASL Alphabet dataset
2. Place training data in `data/asl_alphabet_train/asl_alphabet_train/`
3. Place test data in `data/asl_alphabet_test/asl_alphabet_test/`

Expected structure:
```
data/
├── asl_alphabet_train/asl_alphabet_train/
│   ├── A/
│   ├── B/
│   ├── C/
│   └── ...
└── asl_alphabet_test/asl_alphabet_test/
    ├── A/
    ├── B/
    └── ...
```

## Usage

### 1. Preprocess Dataset (if needed)
```bash
python preprocess_dataset.py
```

### 2. Train the Model
```bash
python train_model.py
```

### 3. Run the Application
```bash
streamlit run app.py
```

## Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 224x224x3
- **Architecture**: 
  - MobileNetV2 backbone (frozen)
  - Global Average Pooling
  - Batch Normalization
  - Dense layer (256 units, ReLU)
  - Dropout (0.5)
  - Output layer (29 classes, Softmax)

## Files Description

- `app.py` - Main Streamlit application
- `train_model.py` - Model training script
- `preprocess_dataset.py` - Dataset preprocessing utilities
- `requirements.txt` - Python dependencies
- `asl_model.keras` - Trained model file
- `class_labels.json` - Class label mappings

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- OpenCV
- MediaPipe
- Streamlit
- NumPy

## Performance

- **Training Accuracy**: ~95%+
- **Validation Accuracy**: ~90%+
- **Real-time FPS**: 30+ fps
- **Inference Time**: <50ms per frame

## Usage Tips

1. **Good Lighting**: Ensure adequate lighting for better detection
2. **Clear Background**: Use a plain background for optimal results
3. **Steady Gestures**: Hold the sign steady for 1-2 seconds

## Troubleshooting

### Model Not Loading
- Ensure `asl_model.keras` and `class_labels.json` exist
- Retrain the model if files are corrupted

### Camera Issues
- Check camera permissions
- Try different camera indices (0, 1, 2...)
- Restart the application

### Poor Accuracy
- Retrain with more epochs
- Improve lighting conditions
- Use consistent hand positioning
