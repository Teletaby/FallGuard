# File: app/skeleton_lstm.py

import torch
import torch.nn as nn

# --- Configuration Constants (Required for main.py imports) ---
SEQUENCE_LENGTH = 30  # Look-back window of frames (e.g., 1 second @ 30 FPS)
FEATURE_SIZE = 55     # Number of features per frame (8 kinematic features + padding)
HIDDEN_SIZE = 128     # LSTM hidden state size
NUM_LAYERS = 2        # Number of stacked LSTM layers
OUTPUT_SIZE = 2       # Output classes: 0 (Normal), 1 (Fall)

# --- LSTM Model Definition ---
class LSTMModel(nn.Module):
    """
    A PyTorch LSTM-based model for sequence classification (Fall Detection).
    It processes a sequence of features and outputs raw logits for two classes.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=0.2 
        )
        
        # Final fully connected layer outputs raw logits (size 2: Normal/Fall)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass through the network, returning raw logits.
        """
        # Pass input through LSTM
        # out shape: (batch_size, sequence_length, hidden_size)
        out, (hn, cn) = self.lstm(x)
        
        # Take the output of the last time step for sequence classification
        final_output = out[:, -1, :]

        # Pass the final time step output through the fully connected layer to get logits
        logits = self.fc(final_output)
        
        return logits