import torch.nn as nn

class RNNFallDetector(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=1):
        super(RNNFallDetector, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])