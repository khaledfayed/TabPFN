from scripts.transformer_prediction_interface import TabPFNClassifier


classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=1, only_inference=False)