import pandas as pd
import numpy as np
import torch
import torch.nn as nn # Added to fix potential reference issues in evaluate_model if not already imported
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # Metrics are kept here

# --- Configuration Constants (Standardized) ---
SEQUENCE_LENGTH = 20
FEATURE_SIZE = 55 
BATCH_SIZE = 32

class URFallDataset(Dataset):
    """
    Dataset class for URFall data.
    """
    def __init__(self, normal_csv_path, fall_csv_path, sequence_length=SEQUENCE_LENGTH, transform=None):
        
        columns = ['sequence', 'frame', 'label', 'HeightWidthRatio', 'MajorMinorRatio', 
                   'BoundingBoxOccupancy', 'MaxStdXZ', 'HHmaxRatio', 'H', 'D', 'P40']
        
        # Load and label data
        normal_df = pd.read_csv(normal_csv_path, names=columns, header=None)
        # Note: URFall uses label 0 for No-Action/Missing data, 1 for Normal, 2 for Fall.
        # Here we re-label 1 (Normal) to 0 (Not Fall) and 2 (Fall) to 1 (Fall).
        normal_df = normal_df[normal_df['label'] == 1].copy() # Keep only 'Normal' activities
        normal_df['label'] = 0                               # Relabel Normal -> 0 (Not Fall)
        
        fall_df = pd.read_csv(fall_csv_path, names=columns, header=None)
        fall_df = fall_df[fall_df['label'] == 2].copy()       # Keep only 'Fall' activities
        fall_df['label'] = 1                                  # Relabel Fall -> 1 (Fall)
        
        df = pd.concat([normal_df, fall_df], ignore_index=True)
        
        # Extract the 8 core kinematic features
        self.feature_cols = ['HeightWidthRatio', 'MajorMinorRatio', 'BoundingBoxOccupancy',
                             'MaxStdXZ', 'HHmaxRatio', 'H', 'D', 'P40']
        features_8d = df[self.feature_cols].values.astype('float32')
        
        # Add dummy features (padding) to match the model's 55-feature input size
        # 55 - 8 = 47 extra features
        dummy_features = np.zeros((len(features_8d), FEATURE_SIZE - 8), dtype='float32')
        self.features = np.hstack((features_8d, dummy_features))
        
        self.labels = df['label'].values.astype('int64')
        self.sequence_length = sequence_length
        self.transform = transform
        
        self.sequences, self.sequence_labels = self._create_sequences()
    
    def _create_sequences(self):
        """Create sequences with stride 1, ensuring all frames in the window have the same label."""
        sequences = []
        labels = []
        
        for i in range(len(self.features) - self.sequence_length + 1):
            seq = self.features[i:i + self.sequence_length]
            # Use the label of the *entire* sequence to enforce strict labeling
            label = self.labels[i + self.sequence_length - 1] 
            
            # Check if all frames in the sequence belong to the same activity class 
            # (Important for clean training, but relaxed the original check slightly)
            if np.all(self.labels[i:i + self.sequence_length] == label):
                sequences.append(seq)
                labels.append(label)
        
        return np.array(sequences), np.array(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.sequence_labels[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        # Return sequence and label (unsqueezed to (1,) for BCELoss)
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32).unsqueeze(0)

# The utility functions (get_data_loaders, get_features_info) are omitted for brevity, 
# as they were mostly correct, but rely on the corrected URFallDataset class above.