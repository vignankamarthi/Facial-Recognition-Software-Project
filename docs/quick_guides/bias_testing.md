# Bias Testing Guide
[ ] PLEASE GO OVER ALL QUICK GUIDES AND NEW DOCUMENTATION IN THE QUICK GUIDES FOLDER
The bias testing feature demonstrates how facial recognition systems can have varying accuracy across different demographic groups.

## Quick Start
```bash
python run_demo.py --bias
```

## What It Does

This feature:
1. Tests face detection accuracy across different demographic groups
2. Visualizes accuracy differences with a bar chart
3. Identifies potential bias in the recognition system

## Setup Process

The first time you run bias testing:
1. The system creates a sample dataset structure:
   ```
   data/test_datasets/sample_dataset/
   ├── group_a/  # First demographic group
   ├── group_b/  # Second demographic group
   └── group_c/  # Third demographic group
   ```

2. You need to add test images to each demographic group folder
   - Each image should contain one clearly visible face
   - Images should represent different demographics (e.g., by skin tone, gender, age)
   - At least 3-5 images per group is recommended for meaningful results

3. For quick demonstrations, the system can automatically populate groups with sample images from LFW dataset if it's installed

## Running the Test

After setup, running the bias test again will:
1. Process all images in each demographic group
2. Calculate detection accuracy for each group
3. Check for significant accuracy differences between groups
4. Generate a bar chart visualization
5. Save results to data/test_datasets/results/

## Interpreting Results

The test generates:
1. **Console output** showing:
   - Overall accuracy across all images
   - Accuracy breakdown by demographic group
   - Potential bias alert if significant difference exists

2. **Bar chart visualization** in `data/test_datasets/results/`:
   - Each bar represents a demographic group
   - Height shows detection accuracy percentage
   - Red line indicates overall average
   - Significant differences suggest potential bias

## Creating Realistic Test Data

For meaningful bias testing:
1. Use the LFW dataset setup:
   ```bash
   python run_demo.py --setup-dataset
   ```

2. Organize test images into demographic categories based on:
   - Skin tone (light, medium, dark)
   - Gender
   - Age groups
   - Different lighting conditions

3. Ensure similar image quality across groups

## Technical Implementation

The bias testing system:
1. Loads test images from demographic group folders
2. Runs face detection on each image
3. Records success/failure rates by group
4. Calculates accuracy percentages
5. Tests for statistically significant differences
6. Generates visualizations with matplotlib

For implementation details, see `src/facial_recognition_software/bias_testing.py`.
