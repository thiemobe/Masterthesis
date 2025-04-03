# This loop ties together the forward model, inverse model, and Bayesian optimization to iteratively minimize the error in measured characteristics by adjusting the heat treatment parameters.
import torch
import torch.nn as nn
import GPyOpt
from GPyOpt.core.task.space import Design_space
import json
import joblib
from InverseModel import InverseModel
from ForwardModel1ImageInput import CombinedModel
import os
import numpy as np
import time
import multiprocessing

# Set the start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

# Zur Darstellung der Models und der Aquirierungsfunktion sowie der Konvergenz
#GPyOpt.plotting.plots_bo.plot_acquisition(bounds, input_dim, model, Xdata, Ydata, acquisition_function, suggested_sample, filename=None)
#GPyOpt.plotting.plots_bo.plot_convergence(Xdata, best_Y, filename=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = os.path.dirname(os.path.abspath(__file__))

# Load pre-trained forward and inverse models
forward_model = CombinedModel(input_size=5, hidden_size=32, num_classes=4, fc1_neurons=88, fc2_neurons=53, use_attention=True, dropout_value=0.10538036256855553, activation_function=nn.Tanh()).to(device)
forward_model.load_state_dict(torch.load(os.path.join(base_dir, 'neue_aug_training/trained_model.pth')))
forward_model.eval()
#optimal parameter for neue_aug_training
    #hidden_size: 32
    #fc1_neurons: 88
    #fc2_neurons: 53
    #use_attention: true
    #lr: 0.004295507418817361
    #dropout: 0.10538036256855553
    #activation_function: Tanh
inverse_model = InverseModel(input_size=4, output_size=5, fc1_neurons=99, fc2_neurons=53, dropout_value=0.13212065672719586, activation_function=nn.Tanh()).to(device)
inverse_model.load_state_dict(torch.load(os.path.join(base_dir, 'InverseLogs/trained_inverse_model.pth')))
inverse_model.eval()
 #optuna parames wth neue_aug_training
    #fc1_neurons 99
    #fc2_neurons 53
    #lr 0.004880901799077941
    #dropout 0.13212065672719586
    #activation_function"Tanh"
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

# Define heat treatment parameter bounds
#space = [
 #   (800, 875),  # Parameter 1: Temperature
  #  (-2, 400),  # Parameter 2: Temperature
   # (-2, 120),   # Parameter 3: Time
   # (-2, 10),     # Parameter 4: Heating rate
   # (400, 800),  # Parameter 5: Temperature
   # (90, 290),   # Parameter 6: Time
   # (10, 40)      # Parameter 7: Cooling rate
#]

space = [
    {'name': 'x1', 'type': 'continuous', 'domain': (800, 875)},
    {'name': 'x2', 'type': 'continuous', 'domain': [-1, 400]},
    {'name': 'x5', 'type': 'continuous', 'domain': (400, 800)},
    {'name': 'x6', 'type': 'continuous', 'domain': [90, 290]},
    {'name': 'x7', 'type': 'continuous', 'domain': [10, 40]}
    
]

# Extract bounds
lower_bounds = [dim['domain'][0] if dim['type'] == 'continuous' else min(dim['domain']) for dim in space]
upper_bounds = [dim['domain'][1] if dim['type'] == 'continuous' else max(dim['domain']) for dim in space]

#constraints = [
    # If x2 == -1, then x3 == -1 and x4 == -1
    #{'name': 'const_1', 'constraint': "np.maximum(0, x[:,1] + 1) * (x[:,2] + 1)"},
    # Ensure x2 == -1 if x2 is between -1 and 10
    # {'name': 'const_x2', 'constraint': "np.maximum(0, (x[:,1] - (-1)) * (200 - x[:,1]))"}
    #{
     #   'name': 'const_x2',
      #  'constraint': "np.where((x[:,1] > -1) & (x[:,1] <= 200), x[:,1] - (-1), 0)"
    #}
#]

#design_space = Design_space(space=space, constraints=constraints)


# Convert the constraint into a format GPyOpt understands
#constraints = [{'name': 'constraint', 'constraint': constraints}]
# Optimization function
def optimize_heat_treatment(target_properties, lower_bounds, upper_bounds, max_iter=5, threshold=1e-4):
    """
    Perform Bayesian optimization to minimize error between measured and predicted characteristics.
    """
    bounds = [{'name': f'param_{i}', 'type': 'continuous', 'domain': (lb, ub)}
              for i, (lb, ub) in enumerate(zip(lower_bounds, upper_bounds))]
    
    # Normalize target properties using the scaler for measured characteristics of the forward model
    target_properties_normalized = scaler_mc_inverse.transform([target_properties])
    target_properties_tensor = torch.tensor(target_properties_normalized, dtype=torch.float32).squeeze(1)
    #print("Target properties tensor shape:", target_properties_tensor.shape)

    def objective_function(heat_treatment_params):
        #start_time = time.time()
        # Convert optimization variables to tensor
        heat_treatment_params_tensor = torch.tensor(heat_treatment_params, dtype=torch.float32).unsqueeze(0).to(device)      
        #print("Heat treatment parameters tensor shape:", heat_treatment_params_tensor.shape)
        # Ensure heat_treatment_params_tensor has the correct shape [batch_size, sequence_length, input_size]
        #heat_treatment_params_tensor = heat_treatment_params_tensor.view(1, 7, 1)
        #print("Heat treatment parameters tensor shape:", heat_treatment_params_tensor.shape)
        heat_treatment_params_tensor = heat_treatment_params_tensor.view(1, -1)
        #print("Heat treatment parameters tensor shape:", heat_treatment_params_tensor.shape)

        # Apply the constraint on the second index
        print(f"Original value at index 2: {heat_treatment_params_tensor[0, 1]}")
        heat_treatment_params_tensor[0, 1] = -1 if -1 <= heat_treatment_params_tensor[0, 1] <= 300 else heat_treatment_params_tensor[0, 1]
        print(f"Adjusted value at index 2: {heat_treatment_params_tensor[0, 1]}")


        # Normalize heat treatment parameters using the scaler for heat treatment parameters of the forward model
        heat_treatment_params_normalized = scaler_ht_forward.transform(heat_treatment_params_tensor.cpu().numpy())
        heat_treatment_params_tensor = torch.tensor(heat_treatment_params_normalized, dtype=torch.float32).to(device)

        # this is for handeling the missing image data
        batch_size = heat_treatment_params_tensor.size(0)
        image_placeholder = torch.zeros(batch_size, 3, 224, 224).to(device)

        # Forward model prediction
        with torch.no_grad():
            predicted_characteristics = forward_model(image_placeholder, heat_treatment_params_tensor)

        # Move target_properties_tensor to the same device as predicted_characteristics
        #target_properties_tensor_device = target_properties_tensor.to(device)

        # Calculate error between predicted and target characteristics
        loss = criterion(predicted_characteristics, target_properties_tensor.to(device))
        print(f"Loss at current iteration: {loss.item()}")
        #end_time = time.time()
        #print(f"Function evaluation time: {end_time - start_time:.4f} seconds | Loss: {loss.item()}")

        return loss.item()

    # Initialize Bayesian optimization
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=objective_function,
        domain=bounds,
        #constraints=constraints,
        acquisition_type='EI',
        exact_feval=True,
        #batch_size=4  # Number of parallel evaluations

    )

    # Get initial guess from the inverse model
    with torch.no_grad():
        batch_size = target_properties_tensor.size(0)
        image_placeholder = torch.zeros(batch_size, 3, 224, 224).to(device) 
        initial_guess = inverse_model(image_placeholder, target_properties_tensor.to(device)).squeeze(0).cpu().numpy()
        print("Initial guess from inverse model:", initial_guess)

    # Start optimization
    optimizer.run_optimization(max_iter=max_iter, eps=threshold)

    return optimizer.x_opt, optimizer.fx_opt

# Main loop
if __name__ == "__main__":

    # Get desired characteristics from the user
    desired_characteristics = get_desired_measured_characteristics()

    # Perform optimization
    print("Starting optimization...")
    optimized_params, final_loss = optimize_heat_treatment(
        desired_characteristics,
        lower_bounds,
        upper_bounds,        
        max_iter=50,
        threshold=1e-4
    )

    # After optimization loop
    optimized_params_tensor = torch.tensor(optimized_params, dtype=torch.float32).unsqueeze(0)

    # Apply the adjustment to the second index
    optimized_params_tensor[0, 1] = -1 if -1 <= optimized_params_tensor[0, 1] <= 300 else optimized_params_tensor[0, 1]

    # Print the adjusted optimized parameters
    print(f"Adjusted Optimized Heat Treatment Parameters: {optimized_params_tensor}")
    print(f"Final Loss: {final_loss}")

    # Save results to file
    with open('optimized_parameters.json', 'w') as f:
        json.dump({"optimized_params": optimized_params.tolist(), "final_loss": final_loss}, f)
    print("Results saved to 'optimized_parameters.json'")
