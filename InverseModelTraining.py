import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, LeaveOneOut
from InverseModelLoader import HeatTreatmentInverseDataset 
from InverseModelpreprocessing import InverseModelPreprocessor
from InverseModel import InverseModel
from sklearn.metrics import root_mean_squared_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


base_dir = os.path.dirname(os.path.abspath(__file__))
samples_folder = os.path.join(base_dir, 'neue_aug_training')
model_path = os.path.join(base_dir,'InverseLogs') 
scaler_path = os.path.join(base_dir, 'scalers')

# Verify dataset directory
assert os.path.exists(samples_folder), "Dataset directory does not exist."
assert len(os.listdir(samples_folder)) > 0, "Dataset directory is empty."

# Load and Preprocess Training Data and Prediction Data
preprocessor = InverseModelPreprocessor(samples_folder)
preprocessor.load_and_preprocess_data()

# Save scalers
os.makedirs(scaler_path, exist_ok=True)
preprocessor.save_scalers(scaler_path)

measured_characteristics = preprocessor.measured_characteristics
heat_treatment_parameters = preprocessor.heat_treatment_parameters

#print(f"Measured characteristics: {len(measured_characteristics)} samples")
#print(f"Heat treatment parameters: {len(heat_treatment_parameters)} samples")
########################################################################################################################################
# Dataloader
dataset = HeatTreatmentInverseDataset(samples_folder, transform=None)
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold Cross-Validation

# Initialize TensorBoard SummaryWriter for overall mean and std
overall_name = f"overall_{datetime.now().strftime('%Y%m%d_%H%M%S')}_INVERSE_optuna_Parameter"
overall_log_dir = os.path.join(base_dir, 'InverseLogs', overall_name)

os.makedirs(overall_log_dir, exist_ok=True)
overall_writer = SummaryWriter(overall_log_dir)

# Store losses for mean and std calculation
train_losses = []
val_losses = []
train_r2_scores = []
val_r2_scores = []

# Convert lists of numpy arrays to tensors (if they are not already tensors)
measured_characteristics = torch.tensor(np.vstack(measured_characteristics), dtype=torch.float32)
heat_treatment_parameters = torch.tensor(np.vstack(heat_treatment_parameters), dtype=torch.float32)

# Debug print statements
print(f"Measured characteristics tensor shape: {measured_characteristics.shape}")
print(f"Heat treatment parameters tensor shape: {heat_treatment_parameters.shape}")
########################################################################################################################################
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f'Fold {fold+1}')

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=10, shuffle=True)
    # hier muss doch eine 1 beim batch_size stehen, oder?
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False) # Use batch size equal to validation set size

    # Initialize the model
    input_size = measured_characteristics.shape[1]  # Number of measured characteristics (output of the forward model) should be 4  # Number of neurons in the hidden layers
    output_size = heat_treatment_parameters.shape[1]  # Number of heat treatment parameters should be 7

    inverse_model = InverseModel(input_size=input_size, output_size=5, fc1_neurons=99, fc2_neurons=53, dropout_value=0.13212065672719586, activation_function=nn.Tanh()).to(device)
    #optuna parames wth neue_aug_training
    #fc1_neurons 99
    #fc2_neurons 53
    #lr 0.004880901799077941
    #dropout 0.13212065672719586
    #activation_function"Tanh"

    #optuna parames with training
    #fc1_neurons 62
    #fc2_neurons 17
    #lr0.003099089003148454
    #dropout 0.11445553057668431
    #activation_function"Tanh"

    # Define loss function and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(inverse_model.parameters(), lr=0.004880901799077941)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    num_epochs = 40  
    clip_value = 1.0  # Maximum allowed value for gradients

    fold_train_losses = []
    fold_val_losses = []
    fold_train_r2 = []
    fold_val_r2 = []
########################################################################################################################################
    # Training
    for epoch in range(num_epochs):
        inverse_model.train()
        train_true_values = []
        train_predictions = []
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            samples_folder, image_after, heat_treatment_parameters, measured_characteristics = batch
            image_after = image_after.to(device)
            heat_treatment_parameters = heat_treatment_parameters.to(device)
            measured_characteristics = measured_characteristics.to(device)
            optimizer.zero_grad()
            outputs = inverse_model(image_after, measured_characteristics)
            loss = criterion(outputs, heat_treatment_parameters)
            loss.backward()
            #ggf verwenden torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value) #Gradient clipping
            optimizer.step()
            train_loss += loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}')
            fold_train_losses.append(loss.item())

            #Accumulate true values and predictions
            train_true_values.extend(heat_treatment_parameters.detach().cpu().numpy())
            train_predictions.extend(outputs.detach().cpu().numpy())

        # Calculate R² for training
        r2 = r2_score(train_true_values, train_predictions)
        fold_train_r2.append(r2)
########################################################################################################################################
        # Validation
        inverse_model.eval()
        val_loss = 0.0
        all_measured_characteristics = []
        all_predicted_parameters = []
        all_true_parameters = []
        val_true_values = []
        val_predictions = []
        with torch.no_grad():
            for batch_idx, batch in enumerate (val_loader):
                samples_folder, image_after, heat_treatment_parameters, measured_characteristics = batch
                image_after = image_after.to(device)
                heat_treatment_parameters = heat_treatment_parameters.to(device)
                measured_characteristics = measured_characteristics.to(device)
                outputs = inverse_model(image_after, measured_characteristics)
                loss = criterion(outputs, heat_treatment_parameters)
                val_loss += loss.item()

                # Collect data for inverse transformation and evaluation
                all_measured_characteristics.append(measured_characteristics.cpu().numpy())
                all_predicted_parameters.append(outputs.cpu().numpy())
                all_true_parameters.append(heat_treatment_parameters.cpu().numpy())
                #print('this is validation', batch_idx, outputs)

                 #Accumulate true values and predictions
                val_true_values.extend(heat_treatment_parameters.detach().cpu().numpy())
                val_predictions.extend(outputs.detach().cpu().numpy())

        val_loss /= len(val_loader)
        fold_val_losses.append(val_loss)

        # Calculate R² for validation
        if len(val_true_values) > 1:  # Ensure there are at least two samples
            r2 = r2_score(val_true_values, val_predictions)
        else:
            r2 = float('nan')  # Handle the case where R² cannot be calculated
        fold_val_r2.append(r2)

        # Step the scheduler
        scheduler.step(val_loss)
########################################################################################################################################
    # Append fold losses to overall lists
    train_losses.append(fold_train_losses)
    val_losses.append(fold_val_losses)
    train_r2_scores.append(fold_train_r2)
    val_r2_scores.append(fold_val_r2)

# Calculate mean and std for each epoch
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)
mean_train_losses = np.mean(train_losses, axis=0)
std_train_losses = np.std(train_losses, axis=0)
mean_val_losses = np.mean(val_losses, axis=0)
std_val_losses = np.std(val_losses, axis=0)

train_r2_scores = np.array(train_r2_scores)
val_r2_scores = np.array(val_r2_scores)
mean_train_r2 = np.mean(train_r2_scores, axis=0)
std_train_r2 = np.std(train_r2_scores, axis=0)
mean_val_r2 = np.mean(val_r2_scores, axis=0)
std_val_r2 = np.std(val_r2_scores, axis=0)

# Log mean and std to TensorBoard
for epoch in range(num_epochs):
    overall_writer.add_scalar('Loss/train_mean', mean_train_losses[epoch], epoch)
    overall_writer.add_scalar('Loss/train_std', std_train_losses[epoch], epoch)
    overall_writer.add_scalar('Loss/val_mean', mean_val_losses[epoch], epoch)
    overall_writer.add_scalar('Loss/val_std', std_val_losses[epoch], epoch)
    overall_writer.add_scalar('R2/train_mean', mean_train_r2[epoch], epoch)
    overall_writer.add_scalar('R2/train_std', std_train_r2[epoch], epoch)
    overall_writer.add_scalar('R2/val_mean', mean_val_r2[epoch], epoch)
    overall_writer.add_scalar('R2/val_std', std_val_r2[epoch], epoch)


# Close the overall writer
overall_writer.close()

# Save the trained model
filename = 'trained_inverse_model.pth'
# Combine the directory path and filename to create the full model path
model_path = os.path.join(model_path, filename)
torch.save(inverse_model.state_dict(), model_path)
print(f"Model saved to {model_path}")