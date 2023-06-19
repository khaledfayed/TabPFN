from scripts.transformer_prediction_interface import TabPFNClassifier
from meta_dataset_loader import load_meta_data_loader, split_datasets, load_OHE_dataset, meta_dataset_loader
from sklearn.metrics import accuracy_score
import time
import torch

def evaluate_classifier(classifier, dids):
    
    datasets = load_OHE_dataset(dids)
    
    for dataset in datasets:
        
        fit_batch = meta_dataset_loader([dataset], num_samples_per_class=5, one_batch=True, shuffle=False)
        start = time.time()
        classifier.fit(fit_batch[0]['x'], fit_batch[0]['y'])
        y_eval, p_eval = classifier.predict(dataset['data'], return_winning_probability=True)
        accuracy = accuracy_score(dataset['target'], y_eval)
        print('Dataset ID:',dataset['id'], 'Prediction time: ', time.time() - start, 'Accuracy', accuracy, '\n')
        return accuracy
        
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=4, only_inference=False)
    evaluate_classifier(classifier, [22])
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