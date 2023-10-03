from scripts.transformer_prediction_interface import TabPFNClassifier
import torch

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=1, only_inference=False)