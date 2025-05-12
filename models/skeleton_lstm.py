import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class LSTMModel(nn.Module):
    """
    LSTM model for skeleton-based fall detection
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        """
        Initialize the LSTM model
        
        Args:
            input_size: Number of expected features in the input x
            hidden_size: Number of features in the hidden state h
            output_size: Number of output features (1 for binary classification)
            num_layers: Number of recurrent layers
            dropout: Dropout probability (0 means no dropout)
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # We only need the output from the last time step
        out = self.fc(out[:, -1, :])
        
        return out


def evaluate_model(model, X_test, y_test, threshold=0.5, batch_size=32):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained LSTM model
        X_test: Test features tensor
        y_test: Test labels tensor
        threshold: Classification threshold for binary prediction
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary containing evaluation metrics (accuracy, precision, recall, f1_score)
    """
    model.eval()  # Set the model to evaluation mode
    
    all_outputs = []
    
    with torch.no_grad():  # No need to track gradients
        # Process in batches
        for start_idx in range(0, X_test.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, X_test.shape[0])
            batch_X = X_test[start_idx:end_idx]
            
            # Forward pass
            outputs = model(batch_X)
            all_outputs.append(outputs)
    
    # Concatenate all batch outputs
    all_outputs = torch.cat(all_outputs, dim=0)
    
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(all_outputs.squeeze()).cpu().numpy()
    
    # Convert to binary predictions
    y_pred = (probs >= threshold).astype(int)
    
    # Convert labels to numpy if they're not already
    if isinstance(y_test, torch.Tensor):
        y_true = y_test.cpu().numpy()
    else:
        y_true = y_test
        
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return metrics


def predict(model, sequence, threshold=0.5):
    """
    Make prediction on a single sequence
    
    Args:
        model: Trained LSTM model
        sequence: Input sequence tensor of shape (1, seq_length, input_size)
        threshold: Classification threshold
        
    Returns:
        Tuple of (prediction, probability)
    """
    model.eval()
    
    with torch.no_grad():
        output = model(sequence)
        prob = torch.sigmoid(output.squeeze()).item()
        pred = 1 if prob >= threshold else 0
        
    return pred, prob