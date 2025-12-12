import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=2, output_size=1, dropout=0.1):
        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (h, c) = self.lstm(x)

        output = self.fc(h[-1])

        return output
