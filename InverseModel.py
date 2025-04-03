import torch
from torch import nn
from torchvision import models

# Reusing the ResNet50 and Attention classes
class Identity(nn.Module):
    def forward(self, x):
        return x

# Initialize ResNet50
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Replace the fully connected layer with an identity function
resnet50.fc = Identity()


# Inverse Model
class InverseModel(nn.Module):
    def __init__(self, input_size, output_size, fc1_neurons, fc2_neurons, dropout_value, activation_function):
        super(InverseModel, self).__init__()

        # Initialize ResNet50 for image feature extraction
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in self.resnet50.parameters():
            param.requires_grad = False  # Freeze ResNet50 parameters to prevent updating during training

        # Unfreeze only layer4 parameters for high-level features
        for param in self.resnet50.layer4.parameters():
            param.requires_grad = True

        # Replace the fully connected layer with an identity function
        self.resnet50.fc = Identity()

        # Initialize Attention
        #self.attention = Attention(input_size)

        # Fully connected layers
        fc1_input_size = 2048 + input_size  
        self.fc1 = nn.Linear(fc1_input_size, fc1_neurons)  # Adjusted to match the combined features size
        self.fc2 = nn.Linear(fc1_neurons, fc2_neurons)  # Adjusted to match the number of classes
        self.fc3 = nn.Linear(fc2_neurons, output_size)

        self.activation_function = activation_function
        self.dropout = nn.Dropout(p=dropout_value) # Dropout layer for uncertainty estimation


    def forward(self, image_after, measured_characteristics):
        if image_after is not None: 
        # Extract features from images
            with torch.no_grad():
                features_after = self.resnet50(image_after)
            # Flatten image features
            #features_before = torch.flatten(features_before, start_dim=1)
            features_after = torch.flatten(features_after, start_dim=1)
            #print("Features before shape flattened:", features_before.shape)
            #print("Features after shape flattened:", features_after.shape)
            # Debug print statements
            #print(f"features_before shape: {features_before.shape}")
            #print(f"features_after shape: {features_after.shape}")
        else:
            # Skip image processing, initialize features to zeros or handle accordingly
            # check size!
            #DOUBLE CHECK
            features_after = torch.zeros(measured_characteristics.size(0), 2048)  # Example zero initialization
    
        # Ensure heat_treatment_parameters is of type torch.float32
        measured_characteristics = measured_characteristics.to(torch.float32)
        # Debug prints
        #print(f"measured characteristics shape: {measured_characteristics.shape}")

        # Concatenate features
        combined_features = torch.cat((features_after, measured_characteristics), dim=1)
        #print("Combined features shape:", combined_features.shape)

        # Fully connected layers
        x = self.activation_function(self.fc1(combined_features))
        x = self.dropout(x)
        x = self.activation_function(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        #print(f"Final output shape: {x.shape}")

        return x
