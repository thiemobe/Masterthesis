from torchvision.models import ResNet50_Weights, resnet50, ResNet101_Weights, resnet101, ResNet152_Weights, resnet152
import pandas as pd
import torch
from torchvision import models
from torch import nn

# ResNet modified to return the features before the fully connected layer
class Identity(nn.Module):
    def forward(self, x):
        return x

# Attention Mechanism for the LSTM output
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context

# Combined Model
class CombinedModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, fc1_neurons, fc2_neurons, use_attention, dropout_value):
        super(CombinedModel, self).__init__()  # Initialize the parent class
        
        # Initialize ResNet50 for image feature extraction
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self.resnet50.parameters():
            param.requires_grad = False  # Freeze ResNet50 parameters to prevent them from being updated during training
        
        # Update only layer4 for high-level features
        for param in self.resnet50.layer4.parameters():
            param.requires_grad = True

        # Replace the fully connected layer with an identity function
        self.resnet50.fc = Identity()

        # Initialize Attention Mechanism if specified
        self.use_attention = use_attention
        if use_attention:
            self.attention = Attention(hidden_size)        
        
        # Initialize LSTM for sequence data
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Initialize fully connected layers
        fc1_input_size = 2048 * 2 + hidden_size
        self.fc1 = nn.Linear(fc1_input_size, fc1_neurons)  # Adjusted to match the combined features size
        self.fc2 = nn.Linear(fc1_neurons, fc2_neurons)  # Adjusted to match the number of classes
        self.fc3 = nn.Linear(fc2_neurons, num_classes)
        
        # Initialize ReLU and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_value)  # Dropout layer for uncertainty estimation

    def forward(self, image_before, image_after, heat_treatment_parameters):
        # Process images through ResNet50
        if image_before is not None and image_after is not None: 
            with torch.no_grad():
                features_before = self.resnet50(image_before)
                features_after = self.resnet50(image_after)
            features_before = torch.flatten(features_before, start_dim=1)
            features_after = torch.flatten(features_after, start_dim=1)
        else:
            # Skip image processing, initialize features to zeros
            features_before = features_after = torch.zeros(heat_treatment_parameters.size(0), 2048)  # Example zero initialization
        
        # Ensure heat_treatment_parameters has three dimensions
        if len(heat_treatment_parameters.shape) == 2:
            heat_treatment_parameters = heat_treatment_parameters.unsqueeze(-1)  # Add the input_size dimension
            heat_treatment_parameters = heat_treatment_parameters.permute(0, 2, 1)  # Reshape to [batch_size, sequence_length, input_size]
        elif len(heat_treatment_parameters.shape) == 3:
            if heat_treatment_parameters.size(-1) == self.lstm.input_size:
                print("Tensor is already in the correct shape:", heat_treatment_parameters.shape)
            elif heat_treatment_parameters.size(-2) == self.lstm.input_size:
                heat_treatment_parameters = heat_treatment_parameters.permute(0, 2, 1)
            else:
                raise ValueError(f"Expected input size {self.lstm.input_size}, got {heat_treatment_parameters.size(-1)}")
        else:
            raise ValueError(f"Expected 2 or 3 dimensions, got {len(heat_treatment_parameters.shape)}")

        # Process sequences through LSTM
        heat_treatment_parameters = heat_treatment_parameters.to(torch.float32)
        lstm_out, (hn, cn) = self.lstm(heat_treatment_parameters)

        # Apply Attention Mechanism if specified
        if self.use_attention:
            lstm_out = self.attention(lstm_out)
        else:
            lstm_out = hn[-1]  # Use the last hidden state
        
        # Ensure lstm_out is 2-dimensional
        if len(lstm_out.shape) == 3:
            lstm_out = lstm_out.squeeze(0)  # Remove the batch dimension if it exists

        # Concatenate all features
        combined_features = torch.cat((features_before, features_after, lstm_out), dim=1)

        # Pass through dense layers
        x = self.relu(self.fc1(combined_features))
        x = self.fc2(x)
        x = self.fc3(x)

        return x