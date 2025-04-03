from torchvision.models import ResNet50_Weights, resnet50, ResNet101_Weights, resnet101, ResNet152_Weights, resnet152
import pandas as pd
import torch
from torchvision import models
from torch import nn

# ResNet modified to return the features before the fully connected layer --> outputs high-level features instead of class probabilities
class Identity(nn.Module):
    def forward(self, x):
        return x

# Initialize ResNet50
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Replace the fully connected layer with an identity function
resnet50.fc = Identity()
    
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
    def __init__(self, input_size, hidden_size, num_classes, fc1_neurons, fc2_neurons, use_attention, dropout_value, activation_function):
        super(CombinedModel, self).__init__()  # Initialize the parent class
        # Initialize ResNet50 for image feature extraction
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self.resnet50.parameters():
            param.requires_grad = False  # Freeze ResNet50 parameters to prevent them from being updated during training
        
        # update only layer4 for high-level features
        for param in self.resnet50.layer4.parameters():
            param.requires_grad = True

        # Replace the fully connected layer with an identity function
        self.resnet50.fc = Identity()

        self.use_attention = use_attention
        if use_attention:
            self.attention = Attention(hidden_size)        
        
        # Initialize LSTM for sequence data
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Initialize fully connected layers
        fc1_input_size = 2048 + hidden_size
        self.fc1 = nn.Linear(fc1_input_size, fc1_neurons)  # Adjusted to match the combined features size
        self.fc2 = nn.Linear(fc1_neurons, fc2_neurons)  # Adjusted to match the number of classes
        self.fc3 = nn.Linear(fc2_neurons, num_classes)
        # Initialize ReLU and Dropout
        self.activation_function = activation_function
        self.dropout = nn.Dropout(p=dropout_value) # Dropout layer for uncertainty estimation

    def forward(self, image_after, heat_treatment_parameters):
        # Move inputs to the same device as the model
        device = next(self.parameters()).device
        if image_after is not None:
            image_after = image_after.to(device) 
            heat_treatment_parameters = heat_treatment_parameters.to(device)
         # Process images through ResNet50
            with torch.no_grad():
                #features_before = self.resnet50(image_before)
                features_after = self.resnet50(image_after)
                #print("Features before shape:", features_before.shape)
            features_after = torch.flatten(features_after, start_dim=1)
            #print("Features before shape flattened:", features_before.shape)
            #print("Features after shape flattened:", features_after.shape)
        else:
            # Skip image processing, initialize features to zeros or handle accordingly
            # check size!
            features_after = torch.zeros(heat_treatment_parameters.size(0), 2048, device=device)  # Example zero initialization
        
        # Debug print statements
        #print(f"features_before shape: {features_before.shape}")
        #print(f"features_after shape: {features_after.shape}")

        #Ensure heat_treatment_parameters has three dimensions
        if len(heat_treatment_parameters.shape) == 2:
            heat_treatment_parameters = heat_treatment_parameters.unsqueeze(-1)  # Add the input_size dimension
            #Reshape to [batch_size, sequence_length, input_size] = [10, 12, 1]
            heat_treatment_parameters = heat_treatment_parameters.permute(0, 2, 1)
            #("Shape after permute:", heat_treatment_parameters.shape)
        elif len(heat_treatment_parameters.shape) == 3:
            if heat_treatment_parameters.size(-1) == self.lstm.input_size:
                # The tensor is already in the correct shape
                print("Tensor is already in the correct shape:", heat_treatment_parameters.shape)
            elif heat_treatment_parameters.size(-2) == self.lstm.input_size:
                # The tensor has the second and third dimensions in the wrong order
                heat_treatment_parameters = heat_treatment_parameters.permute(0, 2, 1)
                #print("Shape after permute:", heat_treatment_parameters.shape)
            else:
                raise ValueError(f"Expected input size {self.lstm.input_size}, got {heat_treatment_parameters.size(-1)}")
        else:
            raise ValueError(f"Expected 2 or 3 dimensions, got {len(heat_treatment_parameters.shape)}")


        # Process sequences through LSTM
        # Ensure heat_treatment_parameters is of type torch.float32
        heat_treatment_parameters = heat_treatment_parameters.to(torch.float32)

        # Debug prints
        #print(f"heat_treatment_parameters shape: {heat_treatment_parameters.shape}")

        # Pack the sequences for handeling different batch sizes
        #packed_input = pack_padded_sequence(heat_treatment_parameters, sequence_lengths.cpu(), batch_first=True, enforce_sorted=True)
        #print("packed input shape", packed_input.data.shape) 

        #packed_output, (hn, cn) = self.lstm(packed_input)
        # Unpack the sequences
        #lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        lstm_out, (hn, cn) = self.lstm(heat_treatment_parameters)
        #print('LSTM layer output',lstm_out)
        #print("LSTM output shape:", lstm_out.shape)
        #print("LSTM hidden state shape:", hn.shape)
        #print("LSTM cell state shape:", cn.shape)

        # Use the last hidden state (not done yet)       
        if self.use_attention:
            lstm_out = self.attention(lstm_out)        #print('Attention layer output',lstm_out)
        else:
            # Use the last hidden state
            lstm_out = hn[-1] 
        
        # Ensure lstm_out is 2-dimensional
        if len(lstm_out.shape) == 3:
            lstm_out = lstm_out.squeeze(0)  # Remove the batch dimension if it exists

        # Concatenate all features
        combined_features = torch.cat((features_after, lstm_out), dim=1)
        #print("Combined features shape:", combined_features.shape)

        # Pass through dense layers
        x = self.activation_function(self.fc1(combined_features))
        x = self.dropout(x)
        x = self.activation_function(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        #print(f"Final output shape: {x.shape}")


        #print(f"Forward pass dropout active: {self.training}, dropout output: {x}\n")  # Print whether dropout is active and the output of the dropout layer

        return x
