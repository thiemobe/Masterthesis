import os
from PIL import Image
import torch
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib  # For saving and loading the scaler (necceassary for the testing phase)
import pandas as pd

# Load and preprocess data; create tensors and handle missing data
class DataPreprocessor:
    def __init__(self, samples_folder):
        self.samples_folder = samples_folder
        self.image_after = []
        self.heat_treatment_parameters = []
        self.measured_characteristics = []
        self.scaler_ht = StandardScaler()  # Scaler for heat treatment parameters
        self.scaler_mc = StandardScaler()  # Scaler for measured characteristics
        self.heat_treatment_parameters_lengths = [] # New list to store original lengths of heat_treatment_parameters
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
                    # Check for the presence of image files
                    image_after_path = os.path.join(sample_path, 'after.png')
                    if os.path.exists(image_after_path):
                        self._process_images(sample_path)
                    
                    # Process heat treatment parameters regardless of image presence
                    # Process parameters
                    heat_treatment_parameters, measured_characteristics = self._process_parameters(sample_path)
                    if heat_treatment_parameters is not None:
                        heat_treatment_parameters_list.append(heat_treatment_parameters)
                    if measured_characteristics is not None:
                        measured_characteristics_list.append(measured_characteristics)
            
                    #print("Loading and preprocessing data...")
                    #print(f"Loaded {len(self.image_before)} 'image_before' samples")
                    #print(f"Loaded {len(self.image_after)} 'image_after' samples")
                    #print(f"Loaded {len(self.heat_treatment_parameters)} 'heat_treatment_parameters'")
                    #print(f"Loaded {len(self.measured_characteristics)} 'measured_characteristics'")
                    #print(f"Loaded {len(self.heat_treatment_parameters_lengths)} 'heat_treatment_parameters_lengths'")
        # Normalize heat treatment parameters and measured characteristics after loading all data
        # Normalize heat treatment parameters and measured characteristics after loading all data
        
        if heat_treatment_parameters_list:
            all_heat_treatment_parameters = np.vstack(heat_treatment_parameters_list)
            #print("All heat treatment parameters before normalization:", all_heat_treatment_parameters)
            # Fit the scaler
            self.scaler_ht.fit(all_heat_treatment_parameters)
            # Transform the data
            normalized_htp = self.scaler_ht.transform(all_heat_treatment_parameters)
            #print("All heat treatment parameters after normalization:", normalized_htp)
            # Normalize each parameter and convert to torch tensor
            self.heat_treatment_parameters = [torch.from_numpy(ht) for ht in normalized_htp]
            #print(f"Loaded {len(self.heat_treatment_parameters)} 'heat_treatment_parameters'")
            #print(f"Normalized heat treatment parameters")
            #for param in self.heat_treatment_parameters:
            #    print(param)
        if measured_characteristics_list:
            all_measured_characteristics = np.vstack(measured_characteristics_list)
            #print("All mcs before normalization:", all_measured_characteristics)
            self.scaler_mc.fit(all_measured_characteristics)
            normalized_mc = self.scaler_mc.transform(all_measured_characteristics)
            #print("All mcs after normalization:", normalized_mc)
            self.measured_characteristics = [torch.from_numpy(mc) for mc in normalized_mc]
            #for param in self.measured_characteristics:
            #print(param)

    def save_scalers(self, path):
        joblib.dump(self.scaler_ht, os.path.join(path, 'scaler_ht_forward.pkl'))
        joblib.dump(self.scaler_mc, os.path.join(path, 'scaler_mc_forward.pkl'))

    def load_scalers(self, path):
        self.scaler_ht = joblib.load(os.path.join(path, 'scaler_ht_forward.pkl'))
        self.scaler_mc = joblib.load(os.path.join(path, 'scaler_mc_forward.pkl'))

    #Load and preprocess images        
    def _process_images(self, sample_path):
        image_after_path = os.path.join(sample_path, 'after.png')   
        if os.path.exists(image_after_path):
            image_after = self.transform(Image.open(image_after_path))
            self.image_after.append(image_after) #transformed images asre stored in a list
        else: 
            self._handle_missing_images()

    
    def _handle_missing_images(self):
        # Initialize with zero tensors if images are missing
        self.image_after.append(torch.zeros((3, 224, 224)))

    def _process_parameters(self, sample_path):
        heat_treatment_parameters_path = os.path.join(sample_path, 'heat treatment sequence.txt')
        measured_characteristics_path = os.path.join(sample_path, 'features.txt')
        
        heat_treatment_parameters_sample = None
        measured_characteristics_sample = None

        if os.path.exists(heat_treatment_parameters_path): #Numpy array is loaded and converted to tensor
            heat_treatment_parameters_sample = pd.read_csv(heat_treatment_parameters_path, delimiter='\s+', header=None).values
            #print(f"Loaded HTP from {heat_treatment_parameters_path}: {heat_treatment_parameters_sample.shape}")
            self.heat_treatment_parameters_lengths.append(len(heat_treatment_parameters_sample))
            self.heat_treatment_parameters.append(torch.from_numpy(heat_treatment_parameters_sample).float())
        if os.path.exists(measured_characteristics_path):
            measured_characteristics_sample = np.loadtxt(measured_characteristics_path, dtype=np.float32)
            self.measured_characteristics.append(torch.from_numpy(measured_characteristics_sample))
        else:
            self._handle_missing_measured_characteristics()

        return heat_treatment_parameters_sample, measured_characteristics_sample

    def _handle_missing_measured_characteristics(self):
        # Initialize with zero tensors if measured characteristics are missing
        self.measured_characteristics.append(torch.zeros((4,))) #is 4 correct?


    def inverse_transform_measured_characteristics(self, measured_characteristics):
        return self.scaler_mc.inverse_transform(measured_characteristics)

    def inverse_transform_heat_treatment_parameters(self, heat_treatment_parameters):
        return [self.scaler_ht.inverse_transform(ht) for ht in heat_treatment_parameters]
   