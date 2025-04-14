# Bias Testing Guide
The bias testing feature demonstrates how facial recognition systems can have varying accuracy across different demographic groups, using a properly labeled dataset.

## Quick Start
```bash
python run_demo.py --bias
```

## What It Does

This feature:
1. Tests face detection accuracy across different demographic groups
2. Visualizes accuracy differences with colored bar charts
3. Identifies potential bias in recognition systems
4. Provides statistical analysis of accuracy disparities

## Setup Process

The first time you run bias testing:
1. The system creates a demographic split set structure based on ethnicities:
   ```
   data/test_datasets/demographic_split_set/
   ├── white/  # White ethnicity group
   ├── black/  # Black ethnicity group
   ├── asian/  # Asian ethnicity group
   ├── indian/ # Indian ethnicity group
   └── others/ # Other ethnicities
   ```

2. You need to download and set up the UTKFace dataset:
   ```bash
   python run_demo.py --setup-dataset
   ```
   Then select Option 1: "Download UTKFace dataset"
   Then select Option 2: "Set up bias testing with UTKFace"

3. The UTKFace dataset provides proper demographic labels:
   - Each image filename contains age, gender, and ethnicity information
   - Images are organized by ethnicity categories
   - This allows for meaningful bias testing based on real demographic data

## Running the Test

After setup, running the bias test again will:
1. Process images in each demographic group
2. Calculate detection accuracy for each ethnicity group
3. Check for significant accuracy differences between ethnicities
4. Generate a color-coded bar chart visualization
5. Provide statistical analysis of detected bias
6. Save results to data/test_datasets/results/

## Interpreting Results

The test generates:
1. **Console output** showing:
   - Overall accuracy across all ethnicities
   - Accuracy breakdown by ethnic group
   - Potential bias alert if significant difference exists
   - Statistical analysis including variance and standard deviation

2. **Bar chart visualization** in `data/test_datasets/results/`:
   - Each bar represents an ethnic group with color coding
   - Height shows detection accuracy percentage
   - Red line indicates overall average
   - Significant differences suggest potential algorithmic bias

## UTKFace Dataset Information

The UTKFace dataset:
1. Contains over 20,000 face images with age, gender, and ethnicity annotations
2. Age ranges from 0 to 116 years
3. Gender categories: 0 (male) and 1 (female)
4. Ethnicity categories:
   - 0: White
   - 1: Black
   - 2: Asian
   - 3: Indian
   - 4: Others (including Hispanic, Latino, Middle Eastern)

## Detailed Analysis Mode

For more in-depth bias investigation:
1. Select "Demographic Bias Testing (UTKFace)" from the main menu
2. When prompted, choose "Yes" to enable detailed statistical analysis
3. The system will calculate:
   - Standard deviation of accuracy across demographics
   - Variance in facial recognition performance
   - Mean absolute deviation from average accuracy
   - Bias level categorization (Low/Moderate/High)



## Technical Implementation

The bias testing system:
1. Loads labeled images from demographic-specific folders
2. Runs face detection on each image
3. Records success/failure rates by ethnicity
4. Calculates accuracy percentages
5. Tests for statistically significant differences
6. Generates colored visualizations with matplotlib

For implementation details, see `src/facial_recognition_software/bias_testing.py`.
