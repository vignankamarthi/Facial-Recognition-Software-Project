# Dataset Setup Guide

This guide explains how to set up and manage datasets for the facial recognition project.

## Quick Start
```bash
python run_demo.py --setup-dataset
```

## Available Operations

The dataset setup menu provides four options:

1. **Download LFW dataset sample**
2. **Prepare known faces from LFW**
3. **Prepare test dataset from LFW**
4. **Return to main menu**

## 1. Download LFW Dataset Sample

This option downloads a subset of the Labeled Faces in the Wild (LFW) dataset:

```bash
python run_demo.py --setup-dataset
# Select option 1
```

**Process**:
1. Prompts for number of people to include (recommended: 10-100)
2. Downloads the LFW dataset archive (~200MB)
3. Extracts files to data/datasets/lfw/
4. Creates a random sample in data/datasets/lfw/lfw_sample/

**Notes**:
- Downloads may take several minutes depending on internet speed
- The full LFW dataset contains over 13,000 images of 5,749 people
- The sample uses a random subset to save space

## 2. Prepare Known Faces from LFW

This option creates reference faces for face matching:

```bash
python run_demo.py --setup-dataset
# Select option 2
```

**Process**:
1. Prompts for number of people to include as known faces
2. Selects random people from the LFW dataset with multiple images
3. Copies one image per person to data/sample_faces/

**Result**:
- Creates a set of labeled face images for the face matching feature
- These images will be used as reference for identification

## 3. Prepare Test Dataset from LFW

This option creates test images for evaluating face recognition:

```bash
python run_demo.py --setup-dataset
# Select option 3
```

**Process**:
1. Prompts for:
   - Number of known people to include in test set
   - Number of test images per person
2. Copies images to:
   - data/test_images/known/: Additional images of people in the known set
   - data/test_images/unknown/: Images of people not in the known set

**Purpose**:
- Creates a structured test set for evaluating face matching accuracy
- Includes both known faces (should match) and unknown faces (should not match)

## Directory Structure

After setup, your data directory will contain:

```
data/
├── datasets/            # Raw datasets
│   └── lfw/             # Labeled Faces in the Wild dataset
│       └── lfw_sample/  # Sampled subset of LFW
├── sample_faces/        # Reference faces for matching
├── test_datasets/       # Datasets for bias testing
│   └── sample_dataset/  # Demographic groups for bias testing
└── test_images/         # Test images for static processing
    ├── known/           # Known people (should match)
    └── unknown/         # Unknown people (should not match)
```

## Using Your Own Images

You can also add your own images to the system:

1. For face matching references:
   - Add clear face images to data/sample_faces/
   - Name files with the person's name (e.g., john_smith.jpg)

2. For bias testing:
   - Add images to data/test_datasets/sample_dataset/ subfolders
   - Organize by demographic groups in separate folders

3. For general testing:
   - Add test images to data/test_images/

## Troubleshooting

- **Download failures**: Check your internet connection and try again
- **"Missing dataset directory"**: Run the quick setup script first
- **Permission errors**: Check file system permissions on the data directory

For more information on the technical implementation, see:
- `src/utilities/image_processing.py`
- `src/facial_recognition_software/bias_testing.py`
