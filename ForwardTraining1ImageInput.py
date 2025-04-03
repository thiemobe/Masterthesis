import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold, LeaveOneOut
from ForwardPreprocessing1ImageInput import DataPreprocessor
from ForwardLoader1ImageInput import HeatTreatmentDataset
from ForwardModel1ImageInput import CombinedModel
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the Preprocessor
base_dir = os.path.dirname(os.path.abspath(__file__))
samples_folder = os.path.join(base_dir, 'neue_aug_training')
model_path = os.path.join(base_dir, 'neue_aug_training')
scaler_path = os.path.join(base_dir, 'scalers')

# Verify dataset directory
assert os.path.exists(samples_folder), "Dataset directory does not exist."
assert len(os.listdir(samples_folder)) > 0, "Dataset directory is empty."

# Load and Preprocess Training Data and Prediction Data
preprocessor = DataPreprocessor(samples_folder)
preprocessor.load_and_preprocess_data()

# Save scalers
os.makedirs(scaler_path, exist_ok=True)
preprocessor.save_scalers(scaler_path)
########################################################################################################################################
# Dataloader
# define a custom collate_fn if necessary
dataset = HeatTreatmentDataset(samples_folder, transform=None)
#loo = LeaveOneOut()  # Leave-One-Out Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold Cross-Validation

# Initialize TensorBoard SummaryWriter for overall mean and std
overall_name = f"overall_{datetime.now().strftime('%Y%m%d_%H%M%S')}_FORWARD_optuna_Parameter"
overall_log_dir = os.path.join('logs', overall_name)

os.makedirs(overall_log_dir, exist_ok=True)
print(f"Initializing SummaryWriter at: {overall_log_dir}")
overall_writer = SummaryWriter(overall_log_dir)

# Store losses for mean and std calculation
train_losses = []
val_losses = []
train_r2_scores = []
val_r2_scores = []
########################################################################################################################################
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f'Fold {fold+1}')

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=10, shuffle=True)
    # hier muss doch eine 1 beim batch_size stehen, oder?
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False) # Use batch size equal to validation set size
    
    model = CombinedModel(input_size=5, hidden_size=32, num_classes=4, fc1_neurons=88, fc2_neurons=53, use_attention=True, dropout_value=0.10538036256855553, activation_function=nn.Tanh()).to(device) #4 --> alpha, circularity, major, minor
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004295507418817361) #0.005 seems to be a good value
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    #early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    num_epochs = 40
    clip_value = 1.0  # Maximum allowed value for gradients


    #optimal parameter for neue_aug_training
    #hidden_size: 32
    #fc1_neurons: 88
    #fc2_neurons: 53
    #use_attention: true
    #lr: 0.004295507418817361
    #dropout: 0.10538036256855553
    #activation_function: Tanh

    #optimal parameter for training
    #hidden_size: 96
    #fc1_neurons: 94
    #fc2_neurons: 43
    #use_attention: false
    #lr: 0.0017562261113064522
    #dropout: 0.20983600832862556
    #activation_function: ELU

    fold_train_losses = []
    fold_val_losses = []
    fold_train_r2 = []
    fold_val_r2 = []
########################################################################################################################################
    # Training
    for epoch in range(num_epochs):
        model.train()
        train_true_values = []
        train_predictions = []        
        for batch_idx, batch in enumerate(train_loader):
            print(f"Batch {batch_idx}: {batch}")
            samples_folder, image_after, heat_treatment_parameters, measured_characteristics = batch
            image_after = image_after.to(device)
            heat_treatment_parameters = heat_treatment_parameters.to(device)
            measured_characteristics = measured_characteristics.to(device)
            optimizer.zero_grad()
            outputs = model(image_after, heat_treatment_parameters)
            loss = criterion(outputs, measured_characteristics)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value) #Gradient clipping
            optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}')
            fold_train_losses.append(loss.item())

            # Accumulate true values and predictions
            train_true_values.extend(measured_characteristics.detach().cpu().numpy())  # Move to CPU before converting to NumPy
            train_predictions.extend(outputs.detach().cpu().numpy())  # Move to CPU before converting to NumPy

        # Calculate R² for training
        r2 = r2_score(train_true_values, train_predictions)
        fold_train_r2.append(r2)
########################################################################################################################################
        # Validation
        model.eval()
        val_loss = 0
        val_true_values = []
        val_predictions = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                samples_folder, image_after, heat_treatment_parameters, measured_characteristics = batch
                image_after = image_after.to(device)
                heat_treatment_parameters = heat_treatment_parameters.to(device)
                measured_characteristics = measured_characteristics.to(device)
                outputs = model(image_after, heat_treatment_parameters)
                loss = criterion(outputs, measured_characteristics)
                val_loss += loss.item()

                # Accumulate true values and predictions
                val_true_values.extend(measured_characteristics.detach().cpu().numpy())  # Move to CPU before converting to NumPy
                val_predictions.extend(outputs.detach().cpu().numpy())  # Move to CPU before converting to NumPy

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
filename = 'trained_model.pth'
# Combine the directory path and filename to create the full model path
model_path = os.path.join(model_path, filename)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")