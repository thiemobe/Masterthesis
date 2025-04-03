from torch.utils.data import Dataset
from ForwardPreprocessing1ImageInput import DataPreprocessor
import os

# Define datasets and dataloaders
class HeatTreatmentDataset(Dataset):
    def __init__(self, samples_folder, transform=None):
        self.samples_folder = samples_folder
        self.data_preprocessor = DataPreprocessor(samples_folder)
        self.transform = transform
        self.samples = []
        self.data_preprocessor.load_and_preprocess_data()
        self.samples = [os.path.join(samples_folder, f) for f in os.listdir(samples_folder) if os.path.isdir(os.path.join(samples_folder, f))]

        #print(f"Initialized dataset with {len(self.samples)} samples from {samples_folder}")

        # Collect file paths and data
        for idx, sample in enumerate(self.data_preprocessor.heat_treatment_parameters):
            sample_file = os.path.join(self.samples_folder, f"sample_{idx}.csv")  # Example file path
            parameters = sample
            self.samples.append((sample_file, parameters))

        for sample_folder in os.listdir(self.samples_folder):
            sample_path = os.path.join(self.samples_folder, sample_folder)
            if os.path.isdir(sample_path):
                self.samples.append(sample_path)
        # Print lengths for debugging
        #print(f"Length of heat_treatment_parameters: {len(self.data_preprocessor.heat_treatment_parameters)}")
        #print(f"Length of measured_characteristics: {len(self.data_preprocessor.measured_characteristics)}")
        #print(f"Length of image_after: {len(self.data_preprocessor.image_after)}")

        # Ensure all lists have the same length
        #assert len(self.data_preprocessor.heat_treatment_parameters) == len(self.data_preprocessor.measured_characteristics), \
        #    "Mismatch in length between heat_treatment_parameters and measured_characteristics"
        #if len(self.data_preprocessor.image_after) > 0:
        #    assert len(self.data_preprocessor.heat_treatment_parameters) == len(self.data_preprocessor.image_after), \
        #        "Mismatch in length between heat_treatment_parameters and image_after"


    def __len__(self):
        length = len(self.data_preprocessor.heat_treatment_parameters)
        #print(f"Dataset length: {length}")
        return length       

    def __getitem__(self, idx):
        heat_treatment_parameters = self.data_preprocessor.heat_treatment_parameters[idx]
        sample_path = self.samples[idx]
        measured_characteristics = self.data_preprocessor.measured_characteristics[idx]

        
        if len(self.data_preprocessor.image_after) > 0:
            image_after = self.data_preprocessor.image_after[idx]
            return sample_path, image_after, heat_treatment_parameters, measured_characteristics
        else:
            return sample_path, heat_treatment_parameters#, measured_characteristics
        
