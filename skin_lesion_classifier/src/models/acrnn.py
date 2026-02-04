
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """
    Computes attention weights and context vector for temporal data.
    """
    def __init__(self, input_dim: int, units: int = 128):
        super(AttentionLayer, self).__init__()
        self.units = units
        
        # W -> attention_score_vec (Dense)
        # Input dim is the feature size of the hidden state
        self.attention_score_vec = nn.Linear(input_dim, input_dim, bias=False)
        
        # V -> attention_vector (Dense)
        # Input dim is concatenation: context_vector + h_t
        self.attention_vector = nn.Linear(input_dim + input_dim, units, bias=False)
        
    def forward(self, x):
        # x shape: (Batch, Time, Features)
        # h_t is the last hidden state: (Batch, Features)
        h_t = x[:, -1, :] 
        
        # score_first_part: (Batch, Time, Features)
        score_first_part = self.attention_score_vec(x)
        
        # Calculate attention scores
        # h_t expanded: (Batch, 1, Features)
        # score = dot(x @ W, h_t)
        # (Batch, Time, Features) @ (Batch, Features, 1) -> (Batch, Time, 1)
        score = torch.bmm(score_first_part, h_t.unsqueeze(-1)).squeeze(-1)
        
        # attention_weights: (Batch, Time)
        attention_weights = F.softmax(score, dim=1)
        
        # context_vector = dot(weights, x)
        # (Batch, 1, Time) @ (Batch, Time, Features) -> (Batch, 1, Features)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)
        
        # pre_activation: Concatenate([context_vector, h_t]) -> (Batch, 2*Features)
        pre_activation = torch.cat([context_vector, h_t], dim=1)
        
        # attention_vector: Dense(units)
        attention_vector = torch.tanh(self.attention_vector(pre_activation))
        
        return attention_vector

class ACRNN(nn.Module):
    """
    Attention Convolutional Recurrent Neural Network (ACRNN)
    Acts as the classification head to be placed on top of a backbone.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super(ACRNN, self).__init__()
        
        # Conv1D configuration matching Keras architecture:
        # Keras: Conv1D(filters=128, kernel_size=3, padding='same')
        # Input to Keras was (Batch, 1, 1280) -> Channels=1280, Length=1.
        # PyTorch Conv1d expects (Batch, Channels, Length).
        
        self.conv1d = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=128, 
            kernel_size=3, 
            padding=1 # 'same' padding for kernel 3
        )
        self.relu = nn.ReLU()
        
        # LSTM configuration:
        # Keras: LSTM(64, return_sequences=True)
        # Input to LSTM is output of Conv1D: (Batch, 1, 128) -> (Batch, Length, Features)
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=64, 
            batch_first=True
        )
        
        # AttentionLayer
        # Input features to attention is hidden_size of LSTM (64)
        self.attention = AttentionLayer(input_dim=64, units=16)
        
        # Final classification layer
        # Output of AttentionLayer is (Batch, 16)
        self.fc = nn.Linear(16, num_classes)
        
    def forward(self, x):
        # x: (Batch, InputDim) e.g. (Batch, 1280)
        
        # Reshape for Conv1D: (Batch, InputDim, 1)
        # Treating features as channels, and time/sequence length as 1.
        x = x.unsqueeze(-1) 
        
        # Conv1D
        x = self.conv1d(x)
        x = self.relu(x)
        
        # Prepare for LSTM
        # LSTM expects (Batch, Length, Features)
        # Current x: (Batch, 128, 1) -> (Batch, 1, 128)
        x = x.permute(0, 2, 1)
        
        # LSTM
        # output: (Batch, Length, HiddenSize)
        x, _ = self.lstm(x)
        
        # Attention
        # input: (Batch, Length, HiddenSize)
        x = self.attention(x)
        
        # Classification
        x = self.fc(x)
        
        return x
