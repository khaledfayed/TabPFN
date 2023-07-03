from scripts.transformer_prediction_interface import TabPFNClassifier
from meta_dataset_loader import load_meta_data_loader, split_datasets, load_OHE_dataset, meta_dataset_loader
from sklearn.metrics import accuracy_score
import time
import torch

def evaluate_classifier(classifier, dids, train_data=0.4):
    
    datasets = load_OHE_dataset(dids)
    
    for dataset in datasets:
        
        train_index = int(len(dataset['data'])*train_data) if int(len(dataset['data'])*train_data) < 1000 else 1000
        # fit_batch = meta_dataset_loader([dataset], num_samples_per_class=5, one_batch=True, shuffle=False)
        fit_data = dataset['data'][:train_index]
        fit_target = dataset['target'][:train_index]
        start = time.time()
        classifier.fit(fit_data, fit_target)
        y_eval, p_eval = classifier.predict(dataset['data'], return_winning_probability=True)
        accuracy = accuracy_score(dataset['target'], y_eval)
        print('Dataset ID:',dataset['id'], 'Shape:', dataset['data'].shape, 'Prediction time: ', time.time() - start, 'Accuracy', accuracy, '\n')
    return accuracy
        
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=4, only_inference=False)
    evaluate_classifier(classifier, [11,15,22,31])
    pass

if __name__ == "__main__":
    main()
    

open_cc_dids = [11,
 14,
 15,
 16,
 18,
 22,
 23,
 31,
 37,
 54,
 458,
 
 1049,
 1050,
 1063,
 1068,
 1510,
 1494,
 1480,
 1462,
 1464,

 40966,
 40982,
 40994,]