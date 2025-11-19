import numpy as np
import pandas as pd
import os
from tqdm import tqdm

DATA_DIR = 'data'
NORMAL_FILENAME = 'urfall-cam0-adls.csv' # Activities of Daily Living (ADLs) -> Label 0
FALL_FILENAME = 'urfall-cam0-falls.csv' # Fall activities -> Label 1

# URFall feature columns + required metadata columns
URFALL_COLUMNS = ['sequence', 'frame', 'label', 'HeightWidthRatio', 'MajorMinorRatio', 
                  'BoundingBoxOccupancy', 'MaxStdXZ', 'HHmaxRatio', 'H', 'D', 'P40']
NUM_FEATURES = 8 # The last 8 columns are the actual features used

def generate_urfall_data(num_sequences, num_frames_per_sequence, is_fall_data):
    """Generates synthetic data mimicking the URFall dataset structure."""
    all_data = []
    
    for seq_id in tqdm(range(1, num_sequences + 1), desc=f"Generating {'Fall' if is_fall_data else 'ADL'} Sequences"):
        # The 'label' column is frame-level. URFall uses 1-5 for ADL/Fall categories, 
        # but the train script converts these to 0 (ADL) or 1 (Fall). We'll generate 
        # the original labels here to match the expected format.
        original_label = 5 if is_fall_data else 1
        
        for frame_id in range(1, num_frames_per_sequence + 1):
            
            # Generate the 8 numerical features (normalized between 0 and 1)
            features = np.random.rand(NUM_FEATURES).astype('float32')
            
            # Apply basic heuristics to make fall data look different from ADL data
            if is_fall_data:
                # Falls often have lower H (height) and higher ratios (as the bounding box collapses)
                features[5] = np.random.uniform(0.05, 0.4) # H (normalized height) is low
                features[0] = np.random.uniform(0.5, 0.95) # HeightWidthRatio is high
            else:
                # ADLs have higher H (upright) and lower ratios
                features[5] = np.random.uniform(0.6, 0.95) # H (normalized height) is high
                features[0] = np.random.uniform(0.1, 0.5) # HeightWidthRatio is low

            
            # Combine metadata and features
            row = [seq_id, frame_id, original_label] + features.tolist()
            all_data.append(row)
            
    return pd.DataFrame(all_data, columns=URFALL_COLUMNS)

if __name__ == '__main__':
    print("Starting URFall Synthetic Data Generation...")
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Generate ADL (Normal) Data (Label 0)
    # Total ~15000 frames
    adl_df = generate_urfall_data(num_sequences=150, num_frames_per_sequence=100, is_fall_data=False)
    
    # 2. Generate Fall Data (Label 1)
    # Total ~15000 frames
    fall_df = generate_urfall_data(num_sequences=150, num_frames_per_sequence=100, is_fall_data=True)

    # Save to CSV files (without header, to match the format expected by the URFallDataset class)
    adl_df.to_csv(os.path.join(DATA_DIR, NORMAL_FILENAME), index=False, header=False)
    fall_df.to_csv(os.path.join(DATA_DIR, FALL_FILENAME), index=False, header=False)

    print("\n--- URFall Data Generation Complete ---")
    print(f"Saved Normal Data: {os.path.join(DATA_DIR, NORMAL_FILENAME)} ({len(adl_df)} frames)")
    print(f"Saved Fall Data: {os.path.join(DATA_DIR, FALL_FILENAME)} ({len(fall_df)} frames)")
    print("You can now run the training script: 'python train_lstm.py'")