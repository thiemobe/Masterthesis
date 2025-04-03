import torch
import os
from torch.utils.data import Dataset
from InverseModelpreprocessing import InverseModelPreprocessor


class HeatTreatmentInverseDataset(Dataset):
    def __init__(self, samples_folder, transform=None):
        self.samples_folder = samples_folder
        self.data_preprocessor = InverseModelPreprocessor(samples_folder)
        self.samples = []
        self.data_preprocessor.load_and_preprocess_data()
        self.samples = [os.path.join(samples_folder, f) for f in os.listdir(samples_folder) if os.path.isdir(os.path.join(samples_folder, f))]

        # Collect file paths and data
        for idx, sample in enumerate(self.data_preprocessor.measured_characteristics):
            sample_file = os.path.join(self.samples_folder, f"sample_{idx}.csv")  # Example file path
            parameters = sample
            self.samples.append((sample_file, parameters))
        # Debug print statements
        print(f"Number of measured characteristics: {len(self.data_preprocessor.measured_characteristics)}")
        print(f"Number of heat treatment parameters: {len(self.data_preprocessor.heat_treatment_parameters)}")
        print(f"Number of samples: {len(self.samples)}")
    #returns number of samples in the dataset
    def __len__(self):
        return len(self.data_preprocessor.measured_characteristics)

    def __getitem__(self, idx):
        measured_characteristics = self.data_preprocessor.measured_characteristics[idx]
        heat_treatment_parameters = self.data_preprocessor.heat_treatment_parameters[idx]
        sample_path = self.samples[idx]

        if len(self.data_preprocessor.image_after) > 0:
            #image_before = self.data_preprocessor.image_before[idx]
            image_after = self.data_preprocessor.image_after[idx]
            return sample_path, image_after, heat_treatment_parameters, measured_characteristics
        else:
            return sample_path, heat_treatment_parameters, measured_characteristics