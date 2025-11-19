import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import the LSTM model definition
from app.skeleton_lstm import LSTMModel, evaluate_model

class URFallDataset:
    def __init__(self, normal_csv_path, fall_csv_path, sequence_length=10):
        """
        Load and prepare URFall dataset
        
        Args:
            normal_csv_path: Path to the CSV file with normal activities
            fall_csv_path: Path to the CSV file with fall activities
            sequence_length: Number of frames to include in each sequence
        """
        print(f"Loading data from {normal_csv_path} and {fall_csv_path}")
        
        # Column names for the URFall dataset
        columns = ['sequence', 'frame', 'label', 'HeightWidthRatio', 'MajorMinorRatio', 
                   'BoundingBoxOccupancy', 'MaxStdXZ', 'HHmaxRatio', 'H', 'D', 'P40']
        
        # Load normal activities
        normal_df = pd.read_csv(normal_csv_path, names=columns, header=None)
        normal_df = normal_df[normal_df['label'] != 0]  # Filter out rows with label 0
        normal_df['label'] = 0  # Set all normal activities to label 0
        
        # Load fall activities
        fall_df = pd.read_csv(fall_csv_path, names=columns, header=None)
        fall_df = fall_df[fall_df['label'] != 0]  # Filter out rows with label 0
        fall_df['label'] = 1  # Set all fall activities to label 1

        # Combine the datasets
        df = pd.concat([normal_df, fall_df], ignore_index=True)
        print(f"Total samples: {len(df)}, Normal: {len(normal_df)}, Falls: {len(fall_df)}")

        # Extract features
        self.features = df[['HeightWidthRatio', 'MajorMinorRatio', 'BoundingBoxOccupancy',
                            'MaxStdXZ', 'HHmaxRatio', 'H', 'D', 'P40']].values.astype('float32')
        
        # Add extra features to match MediaPipe pose output (dummy values for now)
        # This will make the model compatible with the 55 features from MediaPipe (54) + aspect ratio
        dummy_features = np.zeros((len(self.features), 47), dtype='float32')  # 55 - 8 = 47 extra features
        self.features = np.hstack((self.features, dummy_features))
        
        # Extract labels
        self.labels = df['label'].values.astype('int64')
        
        self.sequence_length = sequence_length
        self.sequences, self.labels = self.create_sequences()
        
        print(f"Feature shape: {self.features.shape}, Sequences shape: {self.sequences.shape}")

    def create_sequences(self):
        """Create sequences of frames for LSTM processing"""
        sequences = []
        labels = []
        
        print("Creating sequences...")
        for i in range(0, len(self.features) - self.sequence_length, 1):  # Overlapping sequences with stride 1
            seq = self.features[i:i + self.sequence_length]
            label = self.labels[i + self.sequence_length - 1]  # Label of last frame in sequence
            
            # Only add sequence if all frames have the same label (either all normal or all fall)
            if np.all(self.labels[i:i + self.sequence_length] == label):
                sequences.append(seq)
                labels.append(label)
        
        return np.array(sequences), np.array(labels)

    def get_data(self):
        """Return the prepared sequences and labels"""
        return self.sequences, self.labels


def train_lstm(normal_csv_path, fall_csv_path, save_path='models', sequence_length=10):
    """
    Train an LSTM model on the URFall dataset
    
    Args:
        normal_csv_path: Path to normal activities CSV
        fall_csv_path: Path to fall activities CSV
        save_path: Directory to save the model
        sequence_length: Number of frames in each sequence
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Load and prepare dataset
    dataset = URFallDataset(normal_csv_path, fall_csv_path, sequence_length)
    X, y = dataset.get_data()
    
    # Print dataset statistics
    print(f"Dataset loaded with {len(X)} sequences")
    print(f"Sequence shape: {X.shape}, Label shape: {y.shape}")
    print(f"Number of fall sequences: {np.sum(y == 1)}")
    print(f"Number of normal sequences: {np.sum(y == 0)}")
    
    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} sequences")
    print(f"Test set: {X_test.shape[0]} sequences")
    
    # Define model parameters
    input_size = X.shape[2]  # Number of features per time step
    hidden_size = 128  # Size of LSTM hidden state
    output_size = 1  # Binary classification: fall or no fall
    num_layers = 2  # Number of LSTM layers
    
    print(f"Model configuration: input_size={input_size}, hidden_size={hidden_size}, "
          f"output_size={output_size}, num_layers={num_layers}")
    
    # Initialize model
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with logits
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    epochs = 30
    batch_size = 32
    
    # Track best model
    best_accuracy = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        
        # Generate random indices for batches
        indices = torch.randperm(X_train.shape[0])
        total_loss = 0
        
        # Process in batches
        for start_idx in range(0, X_train.shape[0], batch_size):
            # Clear gradients
            optimizer.zero_grad()
            
            # Get batch indices
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            # Get batch data
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate on test set
        metrics = evaluate_model(model, X_test, y_test)
        
        # Print epoch results
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, "
              f"Accuracy: {metrics['accuracy']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, "
              f"F1: {metrics['f1_score']:.4f}")
        
        # Save best model
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_model_state = model.state_dict()
            print(f"New best model with accuracy: {best_accuracy:.4f}")
    
    # Save the best model
    model_save_path = os.path.join(save_path, 'skeleton_lstm_pytorch_model.pth')
    torch.save(best_model_state, model_save_path)
    print(f"Best model saved to {model_save_path}")
    
    # Save model config for reference
    config_path = os.path.join(save_path, 'model_config.txt')
    with open(config_path, 'w') as f:
        f.write(f"input_size={input_size}\n")
        f.write(f"hidden_size={hidden_size}\n")
        f.write(f"output_size={output_size}\n")
        f.write(f"num_layers={num_layers}\n")
        f.write(f"sequence_length={sequence_length}\n")
    
    print(f"Model configuration saved to {config_path}")
    
    # Final evaluation
    print("\nFinal model evaluation:")
    model.load_state_dict(best_model_state)
    final_metrics = evaluate_model(model, X_test, y_test)
    
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1 Score: {final_metrics['f1_score']:.4f}")
    
    return model, final_metrics


if __name__ == '__main__':
    data_dir = "data"
    
    normal_csv_path = os.path.join(data_dir, 'urfall-cam0-adls.csv')
    fall_csv_path = os.path.join(data_dir, 'urfall-cam0-falls.csv')
    
    if not os.path.exists(normal_csv_path) or not os.path.exists(fall_csv_path):
        print(f"Data files not found! Please ensure the following files exist:")
        print(f"  - {normal_csv_path}")
        print(f"  - {fall_csv_path}")
    else:
        model, metrics = train_lstm(normal_csv_path, fall_csv_path)
        print("Training complete!")