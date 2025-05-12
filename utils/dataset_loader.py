import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class URFallDataset(Dataset):
    """
    Dataset class for URFall data.
    
    This class handles loading, preprocessing, and sequence creation for the URFall dataset.
    """
    def __init__(self, normal_csv_path, fall_csv_path, sequence_length=10, transform=None):
        """
        Initialize the URFall dataset.
        
        Args:
            normal_csv_path (str): Path to CSV file containing normal activities
            fall_csv_path (str): Path to CSV file containing fall activities
            sequence_length (int): Number of frames in each sequence
            transform (callable, optional): Optional transform to be applied to the features
        """
        # Column names for the dataset
        columns = ['sequence', 'frame', 'label', 'HeightWidthRatio', 'MajorMinorRatio', 
                   'BoundingBoxOccupancy', 'MaxStdXZ', 'HHmaxRatio', 'H', 'D', 'P40']
        
        # Load normal activities data
        normal_df = pd.read_csv(normal_csv_path, names=columns, header=None)
        normal_df = normal_df[normal_df['label'] != 0]  # Filter out rows with label 0
        normal_df['label'] = 0  # Set all normal activities to label 0
        
        # Load fall activities data
        fall_df = pd.read_csv(fall_csv_path, names=columns, header=None)
        fall_df = fall_df[fall_df['label'] != 0]  # Filter out rows with label 0
        fall_df['label'] = 1  # Set all fall activities to label 1
        
        # Combine normal and fall data
        df = pd.concat([normal_df, fall_df], ignore_index=True)
        
        # Extract features
        self.features = df[['HeightWidthRatio', 'MajorMinorRatio', 'BoundingBoxOccupancy',
                             'MaxStdXZ', 'HHmaxRatio', 'H', 'D', 'P40']].values.astype('float32')
        
        # Add dummy features to match MediaPipe output
        # This ensures compatibility with the model when used with real-time video
        dummy_features = np.zeros((len(self.features), 47), dtype='float32')  # 55 - 8 = 47 extra features
        self.features = np.hstack((self.features, dummy_features))
        
        # Extract labels
        self.labels = df['label'].values.astype('int64')
        
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Create sequences
        self.sequences, self.sequence_labels = self._create_sequences()
    
    def _create_sequences(self):
        """
        Create sequences from individual frames.
        
        Returns:
            tuple: (sequences, labels) where sequences is a numpy array of shape 
                  (num_sequences, sequence_length, num_features) and labels is a 
                  numpy array of shape (num_sequences,)
        """
        sequences = []
        labels = []
        
        # Create sequences with stride 1 (overlapping)
        for i in range(0, len(self.features) - self.sequence_length, 1):
            seq = self.features[i:i + self.sequence_length]
            # Use the label of the last frame in the sequence
            label = self.labels[i + self.sequence_length - 1]
            
            # Only add sequence if all frames have the same label
            if np.all(self.labels[i:i + self.sequence_length] == label):
                sequences.append(seq)
                labels.append(label)
        
        return np.array(sequences), np.array(labels)
    
    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a sequence and its label by index.
        
        Args:
            idx (int): Index of the sequence to retrieve
            
        Returns:
            tuple: (sequence, label) where sequence is a tensor of shape 
                  (sequence_length, num_features) and label is a scalar tensor
        """
        sequence = self.sequences[idx]
        label = self.sequence_labels[idx]
        
        # Apply transforms if specified
        if self.transform:
            sequence = self.transform(sequence)
        
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def get_data_loaders(normal_csv_path, fall_csv_path, sequence_length=10, 
                     batch_size=32, test_size=0.2, random_state=42):
    """
    Create training and test data loaders.
    
    Args:
        normal_csv_path (str): Path to CSV file containing normal activities
        fall_csv_path (str): Path to CSV file containing fall activities
        sequence_length (int): Number of frames in each sequence
        batch_size (int): Batch size for data loaders
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, test_loader) containing DataLoader objects for 
               training and testing
    """
    # Create dataset
    dataset = URFallDataset(normal_csv_path, fall_csv_path, sequence_length)
    
    # Split into train and test sets
    train_indices, test_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.sequence_labels
    )
    
    # Create data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices)
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_indices)
    )
    
    return train_loader, test_loader


def get_features_info(normal_csv_path, fall_csv_path):
    """
    Get information about the dataset features.
    
    Args:
        normal_csv_path (str): Path to CSV file containing normal activities
        fall_csv_path (str): Path to CSV file containing fall activities
        
    Returns:
        dict: Dictionary containing information about the dataset features
    """
    # Column names for the dataset
    columns = ['sequence', 'frame', 'label', 'HeightWidthRatio', 'MajorMinorRatio', 
               'BoundingBoxOccupancy', 'MaxStdXZ', 'HHmaxRatio', 'H', 'D', 'P40']
    
    # Load normal activities data
    normal_df = pd.read_csv(normal_csv_path, names=columns, header=None)
    
    # Load fall activities data
    fall_df = pd.read_csv(fall_csv_path, names=columns, header=None)
    
    # Combine normal and fall data
    df = pd.concat([normal_df, fall_df], ignore_index=True)
    
    # Get feature columns
    feature_cols = ['HeightWidthRatio', 'MajorMinorRatio', 'BoundingBoxOccupancy',
                    'MaxStdXZ', 'HHmaxRatio', 'H', 'D', 'P40']
    
    # Calculate statistics for each feature
    feature_stats = {}
    for col in feature_cols:
        feature_stats[col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std()
        }
    
    # Additional dataset info
    info = {
        'num_samples': len(df),
        'num_normal': len(normal_df),
        'num_falls': len(fall_df),
        'feature_names': feature_cols,
        'feature_stats': feature_stats
    }
    
    return info