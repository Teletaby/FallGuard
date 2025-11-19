import numpy as np
import pandas as pd
import os

# --- Configuration ---
# Number of samples (rows) to generate in total
TOTAL_SAMPLES = 2000
# Sequence length (number of frames per sample)
SEQUENCE_LENGTH = 30
# Number of keypoint features (e.g., 33 keypoints * 3 coordinates (x, y, visibility) = 99)
FEATURE_SIZE = 99
# Data split ratios
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
# Directory to save files
DATA_DIR = 'data'
# Class labels: 0 for "Safe/No Fall", 1 for "Fall"
CLASS_LABELS = [0, 1]

def generate_keypoint_data(num_samples, seq_len, feature_size):
    """
    Generates synthetic normalized keypoint data.
    Data is normalized (values between 0 and 1) and includes a 'Label' column.
    """
    print(f"Generating {num_samples} samples...")

    # Create the column headers: 'k_0_x', 'k_0_y', 'k_0_v', ..., 'k_32_v'
    feature_names = []
    for i in range(feature_size // 3): # Assuming 3 features per keypoint (x, y, v)
        feature_names.extend([f'k_{i}_x', f'k_{i}_y', f'k_{i}_v'])

    # Create empty list to hold all sample data
    all_data = []

    for i in range(num_samples):
        # Generate random, continuous data for features (mimics sequential video data)
        # Use a slight variation on uniform random data
        features = np.random.rand(seq_len, feature_size) * 0.5 + 0.25 # Range [0.25, 0.75]

        # Determine the label: roughly 60% safe (0), 40% fall (1)
        label = np.random.choice(CLASS_LABELS, p=[0.6, 0.4])

        # If it's a 'Fall' (label=1), make the last few frames look like a fall
        # (e.g., keypoints suddenly dropping to near-zero normalized y-values)
        if label == 1:
            features[-5:, 1::3] = np.random.uniform(low=0.0, high=0.1, size=(5, feature_size // 3))
        # If it's 'Safe' (label=0), make sure the keypoints stay upright
        else:
             features[:, 1::3] = np.random.uniform(low=0.3, high=0.9, size=(seq_len, feature_size // 3))

        # Flatten the 3D data (SEQUENCE_LENGTH, FEATURE_SIZE) into a single row of size (SEQUENCE_LENGTH * FEATURE_SIZE)
        row = features.flatten().tolist()
        row.append(label)
        all_data.append(row)

    # Add headers for the flattened keypoint data and the final 'Label'
    header = [f'frame_{f}' for f in range(seq_len * feature_size)]
    header.append('Label')

    df = pd.DataFrame(all_data, columns=header)
    return df

def split_and_save_data(df):
    """Splits the DataFrame and saves it to train, validation, and test CSV files."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Shuffle the entire dataset
    df = df.sample(frac=1).reset_index(drop=True)

    # Calculate split sizes
    n_total = len(df)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VALIDATION_RATIO)

    # Split the DataFrame
    train_df = df[:n_train]
    val_df = df[n_train:n_train + n_val]
    test_df = df[n_train + n_val:]

    # Save to CSV files
    train_df.to_csv(os.path.join(DATA_DIR, 'train_data.csv'), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, 'validation_data.csv'), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, 'test_data.csv'), index=False)

    print("\n--- Data Generation Complete ---")
    print(f"Saved: {os.path.join(DATA_DIR, 'train_data.csv')} ({len(train_df)} samples)")
    print(f"Saved: {os.path.join(DATA_DIR, 'validation_data.csv')} ({len(val_df)} samples)")
    print(f"Saved: {os.path.join(DATA_DIR, 'test_data.csv')} ({len(test_df)} samples)")
    print("You can now run the prediction script!")


if __name__ == '__main__':
    # 1. Generate the combined dataset
    synthetic_df = generate_keypoint_data(TOTAL_SAMPLES, SEQUENCE_LENGTH, FEATURE_SIZE)

    # 2. Split and save the files
    split_and_save_data(synthetic_df)