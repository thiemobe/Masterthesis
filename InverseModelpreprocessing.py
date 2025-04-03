import os
from PIL import Image
import torch
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib  # For saving and loading the scaler (necceassary for the testing phase)

# Load and preprocess data; create tensors and handle missing data
class InverseModelPreprocessor:
    def __init__(self, samples_folder):
        self.samples_folder = samples_folder
        #self.image_before = []
        self.image_after = []
        self.heat_treatment_parameters = []
        self.measured_characteristics = []
        self.measured_characteristics_lengths = [] # New list to store original lengths of measured_characteristics
        self.scaler_ht = StandardScaler()  # Scaler for heat treatment parameters
        self.scaler_mc = StandardScaler()  # Scaler for measured characteristics
        self.transform = transforms.Compose([    # Define image transformation
            transforms.Resize(256),  # Resize to 256 pixels on the shortest side
            transforms.CenterCrop(224),  # Center crop to 224x224
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize using mean and std for ResNet50
        ])
    
    def load_and_preprocess_data(self):
        heat_treatment_parameters_list = []
        measured_characteristics_list = []
        if os.path.exists(self.samples_folder):
            for sample_folder in os.listdir(self.samples_folder):
                sample_path = os.path.join(self.samples_folder, sample_folder)
                if os.path.isdir(sample_path):
                    # Process heat treatment parameters 
                    #image_before_path = os.path.join(sample_path, 'before.png')
                    image_after_path = os.path.join(sample_path, 'after.png')
                    if os.path.exists(image_after_path):
                        self._process_images(sample_path)
                        heat_treatment_parameters, measured_characteristics = self._process_parameters(sample_path)
                        heat_treatment_parameters_list.append(heat_treatment_parameters)
                        measured_characteristics_list.append(measured_characteristics)

        if heat_treatment_parameters_list:
            #print("Heat treatment parameters list before normilization:", heat_treatment_parameters_list)
            all_heat_treatment_parameters = np.vstack(heat_treatment_parameters_list)
            # Fit the scaler
            self.scaler_ht.fit(all_heat_treatment_parameters)
            # Transform the data
            normalized_htp = self.scaler_ht.transform(all_heat_treatment_parameters)
            #print("After normalization (heat treatment parameters):", normalized_htp)
            self.heat_treatment_parameters = [torch.from_numpy(ht) for ht in normalized_htp]
        if measured_characteristics_list:
            #print("Measured characteristics list before normilization:", measured_characteristics_list)
            all_measured_characteristics = np.vstack(measured_characteristics_list)
            # Fit the scaler
            self.scaler_mc.fit(all_measured_characteristics)
            # Transform the data
            normalized_mc = self.scaler_mc.transform(all_measured_characteristics)
            #print("After normalization (measured characteristics):", normalized_mc)
            self.measured_characteristics = [torch.from_numpy(mc) for mc in normalized_mc]
        # Debug print statements
        print(f"Number of heat treatment parameters: {len(self.heat_treatment_parameters)}")
        print(f"Number of measured characteristics: {len(self.measured_characteristics)}")
        
    def save_scalers(self, path):
        joblib.dump(self.scaler_ht, os.path.join(path, 'scaler_ht_inverse.pkl'))
        joblib.dump(self.scaler_mc, os.path.join(path, 'scaler_mc_inverse.pkl'))

    def load_scalers(self, path):
        self.scaler_ht = joblib.load(os.path.join(path, 'scaler_ht_inverse.pkl'))
        self.scaler_mc = joblib.load(os.path.join(path, 'scaler_mc_inverse.pkl'))
         
         # Load and preprocess images        
    def _process_images(self, sample_path):
        #image_before_paths = os.path.join(sample_path, 'before.png') 
        image_after_path = os.path.join(sample_path, 'after.png')
        if os.path.exists(image_after_path):
            #image_before = self.transform(Image.open(image_before_paths))
            image_after = self.transform(Image.open(image_after_path))
            #self.image_before.append(image_before) #transformed images asre stored in a list
            self.image_after.append(image_after) #transformed images asre stored in a list
        else: 
            self._handle_missing_images()

    def _handle_missing_images(self):
        # Initialize with zero tensors if images are missing
        #self.image_before.append(torch.zeros((3, 224, 224)))  # Assuming 3 channels and 224x224 resolution
        self.image_after.append(torch.zeros((3, 224, 224)))

    def get_heat_treatment_parameters(self):
        return self.heat_treatment_parameters

    def get_measured_characteristics(self):
        return self.measured_characteristics
    
    def _process_parameters(self, sample_path):
        heat_treatment_parameters_path = os.path.join(sample_path, 'heat treatment sequence.txt')
        measured_characteristics_path = os.path.join(sample_path, 'features.txt')
        
        heat_treatment_parameters_sample = None
        measured_characteristics_sample = None

        if os.path.exists(heat_treatment_parameters_path): #Numpy array is loaded and converted to tensor
            heat_treatment_parameters_sample = np.loadtxt(heat_treatment_parameters_path, dtype=np.float32)
            self.heat_treatment_parameters.append(torch.from_numpy(heat_treatment_parameters_sample)) 
        if os.path.exists(measured_characteristics_path):
            measured_characteristics_sample = np.loadtxt(measured_characteristics_path, dtype=np.float32)
            self.measured_characteristics_lengths.append(len(measured_characteristics_sample))
            self.measured_characteristics.append(torch.from_numpy(measured_characteristics_sample))

        return heat_treatment_parameters_sample, measured_characteristics_sample
    
    def inverse_transform_measured_characteristics(self, measured_characteristics):
        inversed_mc = self.scaler_mc.inverse_transform(measured_characteristics)
        #print("After inverse normalization (measured characteristics):", inversed_mc)
        return inversed_mc

    def inverse_transform_heat_treatment_parameters(self, heat_treatment_parameters):
        inversed_htp = self.scaler_ht.inverse_transform(heat_treatment_parameters)
        #print("After inverse normalization (heat treatment parameters):", inversed_htp)
        return inversed_htp