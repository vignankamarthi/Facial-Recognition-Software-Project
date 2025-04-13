# Facial Recognition Software Project
A facial recognition system demonstrating technical capabilities and ethical considerations in AI-based recognition technologies.

## Quick Start Guide

### Installation
```bash
git clone https://github.com/vignankamarthi/Facial-Recognition-Software-Project.git
cd Facial-Recognition-Software-Project
pip install -r requirements.txt
python src/utilities/quick_setup.py
```

### Running the Demo
```bash
python run_demo.py
```

### Features Overview

| Feature | Webcam Required | Description |
|---------|:--------------:|-------------|
| Face Detection | ✅ | Detect faces in real-time |
| Face Anonymization | ✅ | Blur, pixelate, or mask detected faces |
| Face Matching | ✅ | Match detected faces against reference images |
| Static Image Processing | ❌ | Analyze saved images instead of webcam feed |
| Dataset Management | ❌ | Download and prepare datasets for testing |
| Bias Testing | ❌ | Test recognition accuracy across demographic groups |

## Running Without a Webcam

If you don't have a webcam, you can still use these features:

1. **Set up sample data**:
   ```bash
   python run_demo.py --setup-dataset
   ```

2. **Process static images**:
   ```bash
   python run_demo.py --image data/test_images/sample.jpg
   ```

3. **Run bias testing**:
   ```bash
   python run_demo.py --bias
   ```

## Command Line Shortcuts

| Command | Description |
|---------|-------------|
| `--detect` | Run face detection demo |
| `--anonymize` | Run face anonymization demo |
| `--match` | Run face matching demo |
| `--bias` | Run bias testing demo |
| `--image PATH` | Process a single image |
| `--dir PATH` | Process all images in a directory |
| `--setup-dataset` | Download and prepare sample datasets |

## Project Structure

```
Facial-Recognition-Software-Project/
├── data/                   # Data directory
├── docs/                   # Documentation
│   ├── quick_guides/       # Feature-specific guides
│   └── ethical_discussion.md
├── src/                    # Source code
│   ├── facial_recognition_software/  # Core modules
│   ├── utilities/          # Utility modules
│   └── main.py             # Main application
├── run_demo.py             # Demo launcher
└── requirements.txt        # Dependencies
```

For detailed usage instructions, see [docs/quick_guides/](docs/quick_guides/).

For ethical considerations, see [docs/ethical_discussion.md](docs/ethical_discussion_draft.md).

## Troubleshooting

If you encounter issues:

1. Run the setup script: `python src/utilities/quick_setup.py`
2. Check dependencies: `pip install -r requirements.txt`
3. Clean up temporary files: `python src/utilities/cleanup.py`
4. Fix import issues: `python src/utilities/fix_imports.py`

## Project Goals

This project serves as both a functional demonstration and an ethical case study exploring:

- Privacy concerns in facial recognition
- Issues of consent when capturing biometric data
- Algorithmic bias in recognition systems
- Balancing security benefits with individual rights
