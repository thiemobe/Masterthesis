import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ForwardPreprocessing1ImageInput import DataPreprocessor
from ForwardLoader1ImageInput import HeatTreatmentDataset
from ForwardModel1ImageInput import CombinedModel
import numpy as np

#np.random.seed(42)
#torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#test
# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'neue_aug_training/trained_model.pth')
scaler_path = os.path.join(base_dir, 'scalers')
predictions_samples_folder = os.path.join(base_dir,'prediction')

# Model
model = CombinedModel(input_size=5, hidden_size=40, num_classes=4, fc1_neurons=83, fc2_neurons=41, use_attention=False, dropout_value=0.2, activation_function=nn.Tanh()).to(device) #4 --> alpha, circularity, major, minor

# Check if model file exists
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Verify dataset directory
assert os.path.exists(predictions_samples_folder), "Dataset directory does not exist."
assert len(os.listdir(predictions_samples_folder)) > 0, "Dataset directory is empty."

# Load and Preprocess Training Data and Prediction Data
preprocessor = DataPreprocessor(predictions_samples_folder)
preprocessor.load_and_preprocess_data()

# Print debug information
#print(f"Number of heat treatment parameters: {len(preprocessor.heat_treatment_parameters)}")
# Load fitted scalers
preprocessor.load_scalers(scaler_path)

# Ensure scaler for heat treatment parameters is fitted
assert hasattr(preprocessor.scaler_ht, 'mean_'), "Scaler for heat treatment parameters is not fitted."
# Ensure scaler for measured characteristics is fitted
assert hasattr(preprocessor.scaler_mc, 'mean_'), "Scaler for measured characteristics is not fitted."

# Dataloader
# define a custom collate_fn if necessary
dataset = HeatTreatmentDataset(predictions_samples_folder, transform=None)
#print(f"Number of samples in dataset: {len(dataset)}")  # Debug information
data_loader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=None)  # Adjust batch_size as needed --> could cause problems, e.g. when the amount of samples (14) it not divisible by 4
#print("size of variables before batching",len(heat_treatment_parameters), len(image_before), len(image_after), len(measured_characteristics)) 

prediction_preprocessor = DataPreprocessor(predictions_samples_folder)
prediction_preprocessor.load_and_preprocess_data()
#print("Data Loaded and Preprocessed")

# Instantiate Prediction Dataset and DataLoader
prediction_dataset = HeatTreatmentDataset(predictions_samples_folder, transform=None)  # Use the same transform as for training if applicable
prediction_loader = DataLoader(prediction_dataset, batch_size=10, shuffle=False)  


#print(f"Prediction Dataset Size: {len(prediction_dataset)}")
#print(f"Prediction DataLoader Batches: {len(prediction_dataset)}")

# Access the processed data from the preprocessor object
image_after = preprocessor.image_after
heat_treatment_parameters = preprocessor.heat_treatment_parameters
#measured_characteristics = preprocessor.measured_characteristics

assert len(prediction_loader) > 0, "Prediction DataLoader is not empty."

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
num_epochs = 20  

 # Prediction and Uncertainty Estimation
def mc_dropout(model, heat_treatment_parameters, num_passes=20):
    model.train()  # Ensure dropout is enabled
    predictions = []
    for i in range(num_passes):
        with torch.no_grad():
            outputs = model(image_after=None, heat_treatment_parameters=heat_treatment_parameters)
            heat_treatment_parameters = heat_treatment_parameters.to(device)
            #sequence_lengths = sequence_lengths.to(device)
            #print(f"Forward pass {i+1}, outputs:\n{outputs}\n")  # Print the outputs of each forward pass
            outputs_inverse_transformed = preprocessor.inverse_transform_measured_characteristics(outputs.cpu().numpy())  # Move to CPU before converting to NumPy
            outputs_inverse_transformed_tensor = torch.tensor(outputs_inverse_transformed).to(device) # Convert to tensor
            predictions.append(outputs_inverse_transformed_tensor.unsqueeze(0))
            #print(f"Forward pass {i+1}, outputs:\n{outputs_inverse_transformed_tensor}\n")  # Print the outputs of each forward pass
    predictions = torch.cat(predictions, 0)
    predictions = predictions.cpu() if predictions.is_cuda else predictions
    return predictions

model.eval()  # Set model to evaluation mode for the initial prediction

num_high_variance_samples = 10  # Define the number of samples with the highest variance to select

all_predictions = []
sample_paths = []

for batch_idx, batch in enumerate(prediction_loader):
    print(f"Batch {batch_idx}: {len(batch[1])} samples")
    batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
    print(batch)
    samples_path, heat_treatment_parameters = batch
    model.train()  # Ensure model is in training mode for dropout
    predictions = mc_dropout(model, heat_treatment_parameters)
    all_predictions.append(predictions)
    sample_paths.extend(samples_path)
#print(f"these are the predictions: {all_predictions}")

# Calculate variance of the predictions
all_predictions = torch.cat(all_predictions, dim=0)
print(f"All predictions shape: {all_predictions.shape}")

# Reshape all_predictions to match the number of sample paths
all_predictions = all_predictions.view(len(sample_paths), -1, all_predictions.size(-1))  # Added reshaping step
print(f"Reshaped all predictions shape: {all_predictions.shape}")


feature_variances = torch.var(all_predictions, dim=0, unbiased=False)
print(f"Feature variances shape: {feature_variances.shape}")
# Check alignment between all_predictions and sample_paths

# Extracting sample names from file paths
def extract_sample_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

assert len(sample_paths) == all_predictions.shape[0], \
    f"Mismatch: {len(sample_paths)} sample paths vs {all_predictions.shape[0]} predictions"
#assert len(feature_variances.shape) == 2, f"Expected 2D tensor for feature variances, but got shape {feature_variances.shape}"

# Calculate the mean and standard deviation for each of the 4 values
means = torch.mean(all_predictions, dim=1)  # Calculate along the correct dimension
stds = torch.std(all_predictions, dim=1, unbiased=False)  # Calculate along the correct dimension
print(f"Means shape: {means.shape}, Stds shape: {stds.shape}")

# Calculate the coefficient of variation (CV) for each of the 4 values
epsilon = 1e-8
cv = stds / (means + epsilon)
print(f"CV shape: {cv.shape}")

# Calculate the mean CV across the 4 values for each sample
mean_cv = torch.mean(cv, dim=1)
print(f"Mean CV shape: {mean_cv.shape}")

# Get the highest CV indices
highest_cv_indices = torch.topk(mean_cv, num_high_variance_samples, dim=0).indices
highest_cv_sample_names = [extract_sample_name(sample_paths[idx]) for idx in highest_cv_indices.flatten().tolist()]
print(f"Highest CV sample names: {highest_cv_sample_names}")

#print(len(sample_paths), len(set(sample_paths)))

# Output the sample names and data with the highest CV
for idx in highest_cv_indices.flatten().tolist():
    assert idx < len(sample_paths), f"Index {idx} out of bounds for sample_paths with length {len(sample_paths)}"
    sample_path = sample_paths[idx]
    sample_data = prediction_loader.dataset[idx]
    coefficient_of_variation = cv[idx].mean().item()
    extracted_sample_name = extract_sample_name(sample_path)
    #output = f'Sample: {extracted_sample_name}, Data: {sample_data}, Coefficient of Variation: {coefficient_of_variation}'
    output = f'Sample: {extracted_sample_name}, Coefficient of Variation: {coefficient_of_variation}'
    print(output)
#print(f"Dataset size: {len(prediction_loader.dataset)}, Sample Path Size: {len(sample_paths)}")
#print(f"Top indices: {highest_cv_indices.tolist()}")
#print(f"Corresponding sample paths: {[sample_paths[idx] for idx in highest_cv_indices.tolist()]}")
# Check if indices in highest_cv_indices map correctly to the dataset
sampled_dataset_indices = [prediction_loader.dataset.samples.index(p) for p in sample_paths]
#print(f"Dataset indices from sample_paths: {sampled_dataset_indices}")
#print("Coefficient of variation estimation complete.")
print(f"Sample Path: {sample_paths[:10]}")
#print(f"Predictions: {all_predictions[:5]}")
#print(len(sample_paths))
print(all_predictions.shape)


# Function to print all CVs and corresponding sample names
#def print_all_cvs(cv, sample_paths):
#    for idx in range(len(sample_paths)):
#        sample_path = sample_paths[idx]
#        coefficient_of_variation = cv[idx].mean().item()
#        extracted_sample_name = extract_sample_name(sample_path)
#        output = f'Sample: {extracted_sample_name}, Coefficient of Variation: {coefficient_of_variation}'
#        print(output)

# Print all CVs and corresponding sample names
#print_all_cvs(cv, sample_paths)

#debugging
#for sample_path in sample_paths:
    #print(f"Processing sample: {sample_path}")
    
    # Load data for the sample (adjust based on your data loading logic)
    #sample_data = prediction_loader.dataset[sample_paths.index(sample_path)]
    #print(f"Data for {sample_path}: {sample_data}")

    # Calculate mean and standard deviation
    #means = torch.mean(all_predictions, dim=1)  # Calculate along the correct dimension
    #stds = torch.std(all_predictions, dim=1, unbiased=False)  # Calculate along the correct dimension

    #print(f"Mean: {means}, Std Dev: {stds}")

    # Calculate Coefficient of Variation
    #if means != 0:
    #    cv = stds / means
    #else:
    #    cv = float('inf')  # Handle division by zero case explicitly
   # print(f"Coefficient of Variation: {cv}")