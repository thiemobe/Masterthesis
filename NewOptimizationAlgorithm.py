import torch
import torch.nn as nn
import GPyOpt
import json
import joblib
from InverseModel import InverseModel
from ForwardModel1ImageInput import CombinedModel
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = os.path.dirname(os.path.abspath(__file__))

# Load pre-trained forward and inverse models
forward_model = CombinedModel(input_size=7, hidden_size=40, num_classes=4, fc1_neurons=83, fc2_neurons=41, use_attention=False, dropout_value=0.11929701823940043, activation_function=nn.Tanh()).to(device)
forward_model.load_state_dict(torch.load(os.path.join(base_dir, 'neue_aug_training/trained_model.pth')))
forward_model.eval()

inverse_model = InverseModel(input_size=4, output_size=7, fc1_neurons=104, fc2_neurons=46, dropout_value=0.148349194081185, activation_function=nn.Tanh()).to(device)
inverse_model.load_state_dict(torch.load(os.path.join(base_dir, 'InverseLogs/trained_inverse_model.pth')))
inverse_model.eval()

# Load the scalers
scaler_path = os.path.join(base_dir, 'scalers')
scaler_ht_forward = joblib.load(os.path.join(scaler_path, 'scaler_ht_forward.pkl'))
scaler_mc_forward = joblib.load(os.path.join(scaler_path, 'scaler_mc_forward.pkl'))
scaler_ht_inverse = joblib.load(os.path.join(scaler_path, 'scaler_ht_inverse.pkl'))
scaler_mc_inverse = joblib.load(os.path.join(scaler_path, 'scaler_mc_inverse.pkl'))

# Define the loss function (Mean Squared Error)
criterion = nn.MSELoss()

# Function to get desired measured characteristics from the user
def get_desired_measured_characteristics():
    print("Please enter the desired measured characteristics:")
    alpha_fraction = float(input("Alpha fraction: "))
    circularity = float(input("Circularity: "))
    length = float(input("Length: "))
    width = float(input("Width: "))
    return [alpha_fraction, circularity, length, width]

# Define the objective function for GPyOpt
def objective_function(heat_treatment_params):
    print(f"Evaluating heat treatment parameters: {heat_treatment_params}")
    # Convert optimization variables to tensor
    heat_treatment_params_tensor = torch.tensor(heat_treatment_params, dtype=torch.float32).unsqueeze(0).to(device)
    heat_treatment_params_tensor = heat_treatment_params_tensor.view(1, -1)

    # Normalize heat treatment parameters using the scaler for heat treatment parameters of the forward model
    heat_treatment_params_normalized = scaler_ht_forward.transform(heat_treatment_params_tensor.cpu().numpy())
    heat_treatment_params_tensor = torch.tensor(heat_treatment_params_normalized, dtype=torch.float32).to(device)

    # this is for handling the missing image data
    batch_size = heat_treatment_params_tensor.size(0)
    image_placeholder = torch.zeros(batch_size, 3, 224, 224).to(device)

    # Forward model prediction
    with torch.no_grad():
        predicted_characteristics = forward_model(image_placeholder, heat_treatment_params_tensor)

    # Move target_properties_tensor to the same device as predicted_characteristics
    target_properties_tensor_device = target_properties_tensor.to(device)

    # Calculate error between predicted and target characteristics
    loss = criterion(predicted_characteristics, target_properties_tensor_device)
    print(f"Loss: {loss.item()}")
    return loss.item()

# Main loop
if __name__ == "__main__":
    # Define heat treatment parameter bounds
    bounds = [
        {'name': 'param_0', 'type': 'continuous', 'domain': (800, 875)},  # Parameter 1: Temperature
        {'name': 'param_1', 'type': 'continuous', 'domain': (-1, 400)},  # Parameter 2: Temperature
        {'name': 'param_2', 'type': 'continuous', 'domain': (-1, 120)},  # Parameter 3: Time
        {'name': 'param_3', 'type': 'continuous', 'domain': (-1, 10)},   # Parameter 4: Heating rate
        {'name': 'param_4', 'type': 'continuous', 'domain': (400, 800)},  # Parameter 5: Temperature
        {'name': 'param_5', 'type': 'continuous', 'domain': (90, 290)},  # Parameter 6: Time
        {'name': 'param_6', 'type': 'continuous', 'domain': (10, 40)}    # Parameter 7: Cooling rate
    ]

    # Define constraints
    constraints = [
        {'name': 'constraint_1', 'constraint': "(x[:, 1] != -1) | ((x[:, 2] == -1) & (x[:, 3] == -1))"},
        {'name': 'constraint_2', 'constraint': "(x[:, 2] != -1) | ((x[:, 1] == -1) & (x[:, 3] == -1))"},
        {'name': 'constraint_3', 'constraint': "(x[:, 3] != -1) | ((x[:, 1] == -1) & (x[:, 2] == -1))"},
        {'name': 'constraint_4', 'constraint': "(x[:, 1] == -1) | ((x[:, 2] == 120) & (x[:, 3] == 10))"},
        {'name': 'constraint_5', 'constraint': "(x[:, 2] == -1) | (x[:, 3] == 10)"},
        {'name': 'constraint_6', 'constraint': "(x[:, 3] == -1) | (x[:, 2] == 120)"}
]

    # Get desired characteristics from the user
    desired_characteristics = get_desired_measured_characteristics()

    # Normalize target properties using the scaler for measured characteristics of the forward model
    target_properties_normalized = scaler_mc_inverse.transform([desired_characteristics])
    target_properties_tensor = torch.tensor(target_properties_normalized, dtype=torch.float32).squeeze(1)

    # Define the feasible region
    feasible_region = GPyOpt.Design_space(space=bounds, constraints=constraints)

    # Generate initial design
    initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 10)

    # Define the objective
    objective = GPyOpt.core.task.SingleObjective(objective_function)

    # Define the model
    model = GPyOpt.models.GPModel(exact_feval=True, optimize_restarts=10, verbose=False)

    # Define the acquisition optimizer
    acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

    # Define the acquisition function
    acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=acquisition_optimizer)

    # Define the evaluator
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

    # Create the Bayesian Optimization object
    bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, initial_design, verbose=True)

    # Get initial guess from the inverse model
    with torch.no_grad():
        batch_size = target_properties_tensor.size(0)
        image_placeholder = torch.zeros(batch_size, 3, 224, 224).to(device)
        initial_guess = inverse_model(image_placeholder, target_properties_tensor.to(device)).squeeze(0).cpu().numpy()
        print("Initial guess from inverse model:", initial_guess)

    # Run the optimization
    bo.run_optimization(max_iter=50, eps=1e-4)

    # Get the best parameters and loss
    optimized_params = bo.x_opt
    final_loss = bo.fx_opt

    # Display and save results
    print(f"Optimized Heat Treatment Parameters: {optimized_params}")
    print(f"Final Loss: {final_loss}")

    # Save results to file
    with open('optimized_parameters.json', 'w') as f:
        json.dump({"optimized_params": optimized_params.tolist(), "final_loss": final_loss}, f)
    print("Results saved to 'optimized_parameters.json'")