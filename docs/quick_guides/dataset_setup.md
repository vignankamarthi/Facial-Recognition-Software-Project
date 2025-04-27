[ ] TODO: Update this file to reflect NO UTILITIES/ package, rather utils/

# Dataset Setup Guide

This guide explains how to set up and manage datasets for the facial recognition project, including the UTKFace dataset for ethical bias testing.

## Quick Start
```bash
python run_demo.py --setup-dataset
```

## Available Operations

The dataset setup menu provides several options:

1. **Download UTKFace dataset** - Main dataset with demographic labels
2. **Set up bias testing with UTKFace** - Prepare data for demographic analysis
3. **Prepare known faces from UTKFace** - Create reference faces for matching
4. **Prepare test dataset from UTKFace** - Create test images for evaluation
5. **Return to main menu**

## 1. Download UTKFace Dataset

This option downloads the UTKFace (University of Tennessee, Knoxville Face) dataset with demographic annotations:

```bash
python run_demo.py --setup-dataset
# Select option 1
```

**Process**:
1. Prompts for sample size (number of images to include)
2. Asks which ethnicities to include:
   - All ethnicities
   - White and Black only (for pronounced contrast)
   - White, Black, and Asian (most common groups)
   - Custom selection (choose specific ethnicities)
3. Downloads the UTKFace aligned dataset
4. Organizes images by demographic categories

**Dataset Information**:
- UTKFace contains over 20,000 face images with age, gender, and ethnicity annotations
- Each image filename follows the format: [age]_[gender]_[race]_[date&time].jpg
- Gender: 0 (male), 1 (female)
- Ethnicity: 0 (White), 1 (Black), 2 (Asian), 3 (Indian), 4 (Others)

## 2. Set Up Bias Testing with UTKFace

This option prepares the UTKFace dataset for bias testing:

```bash
python run_demo.py --setup-dataset
# Select option 2
```

**Process**:
1. Prompts for number of images per ethnicity group
2. Copies a balanced sample of images from each ethnicity
3. Organizes them into demographic directories:
   - white/
   - black/
   - asian/
   - indian/
   - others/

**Result**:
- Creates a structured dataset in data/test_datasets/demographic_split_set/
- This dataset is used by the bias testing feature to measure recognition accuracy across ethnicities

## 3. Prepare Known Faces from UTKFace

This option creates reference faces for face matching from the UTKFace dataset:

```bash
python run_demo.py --setup-dataset
# Select option 3
```

**Process**:
1. Prompts for number of people to include as known faces
2. Asks whether to balance faces across ethnic groups
3. Creates demographically diverse reference faces

**Features**:
- Option to balance ethnicity representation in the reference set
- Each face is labeled with ethnicity information
- Maintains diverse representation for more fair matching

## 4. Prepare Test Dataset

This option creates test images for evaluating face recognition:

```bash
python run_demo.py --setup-dataset
# Select option 4
```

**Process**:
1. Prompts for:
   - Number of known people to include in test set
   - Number of unknown people to include
2. Creates test images in:
   - data/test_images/known/: Additional images of people in the known set
   - data/test_images/unknown/: Images of people not in the known set

## Directory Structure

After setup, your data directory will contain:

```
data/
├── datasets/
│   └── utkface/             # UTKFace dataset
│       ├── utkface_aligned/ # Raw aligned images
│       └── demographic_split/ # Images organized by ethnicity
├── known_faces/             # Reference faces for matching
├── test_datasets/           # Datasets for bias testing
│   └── demographic_split_set/ # Ethnicity groups for bias testing
│       ├── white/           # White ethnicity group
│       ├── black/           # Black ethnicity group
│       ├── asian/           # Asian ethnicity group
│       ├── indian/          # Indian ethnicity group
│       └── others/          # Other ethnicities
└── test_images/             # Test images for static processing
    ├── known/               # Known people (should match)
    └── unknown/             # Unknown people (should not match)
```

## Using Your Own Images

You can also add your own images to the system:

1. For face matching references:
   - Add clear face images to data/known_faces/
   - Name files with the person's name (e.g., john_smith.jpg)

2. For bias testing:
   - Add images to data/test_datasets/demographic_split_set/ subfolders
   - Organize by demographic groups in appropriate folders

3. For general testing:
   - Add test images to data/test_images/

## Troubleshooting

- **Download failures**: The UTKFace dataset is hosted on Google Drive, which may cause download issues. Consider installing the 'gdown' package (`pip install gdown`) for better Google Drive support
- **"Missing dataset directory"**: Run the quick setup script first
- **Permission errors**: Check file system permissions on the data directory
- **Demographic categorization**: UTKFace ethnicities are based on the dataset's original categorization, which may have limitations. Use this for educational purposes about algorithmic bias.

For more information on the technical implementation, see:
- `src/utilities/image_processing.py`
- `src/facial_recognition_software/bias_testing.py`
