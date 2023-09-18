from scripts.transformer_prediction_interface import TabPFNClassifier
from meta_dataset_loader import load_meta_data_loader, split_datasets, load_OHE_dataset, meta_dataset_loader
from sklearn.metrics import accuracy_score
import time
import torch
import wandb

def evaluate_classifier(classifier, dids, train_data=0.7):
    
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

def evaluate_classifier2(classifier, datasets, log=True,  train_data=0.7):
    
    logs = {}
    
    
    for dataset in datasets:
        
        fit_data = dataset['data'][:512]
        fit_target = dataset['target'][:512]
        start = time.time()
        classifier.fit(fit_data, fit_target)
        y_eval, p_eval = classifier.predict(dataset['data'][512:], return_winning_probability=True)
        accuracy = accuracy_score(dataset['target'][512:], y_eval)
        print('Dataset ID:',dataset['id'], 'Shape:', dataset['data'].shape, 'Prediction time: ', time.time() - start, 'Accuracy', accuracy, '\n')
        wandb_name = f'accuracy_{dataset["id"]}'
        logs[wandb_name] = accuracy
    
    average_accuracy = sum(logs.values())/len(logs.values())
    logs['average_accuracy'] = average_accuracy
    if log: wandb.log(logs)
    return average_accuracy
        
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=1, only_inference=False)
    data = load_OHE_dataset([799], one_hot_encode=False)
    evaluate_classifier2(classifier, data, False)
    pass

if __name__ == "__main__":
    main()
    

open_cc_dids = [
 14,
#  15,
 16,
 18,
 22,
 23,
#  31,
 37,
 54,
 458,
 
#  1049,
 1050,
 1063,
 1068,
 1510,
 1494,
 1480,
 1462,
 1464,

#  40966,
 40982]

auto_ml_dids = [23,
 28,
 30,
 44,
 46,
 60,
 181,
 182,
 715,
 728,
 737,
 740,
 772,
 799,
 803,
 837,
 847,
 871,
 903,
 934,
#  1049,
 1050,
 1068,
 1069,
 1462,
 1466,
 1475,
 1487,
 1494,
 1496,
 1497,
 1504,
 1507,
 1528,
 1529,
 1530,
 1542,
 1547,
 1552,
 40498,
 40646,
 40647,
 40648,
 40649,
 40650,
 40677,
 40680,
 40691,
 40701,
 40704,
 40706,
 40900,
 40982,
 40983,
 42193]

auto_ml_dids_train = [   23,    28,    30,    44,    46,    60,   181,   182,   375,
         725,   728,   735,   737,   752,   761,   772,   803,   807,
         816,   819,   833,   847,   871,   923,  1049,  1050,  1056,
        1069,  1462,  1466,  1475,  1487,  1496,  1497,  1504,  1507,
        1528,  1529,  1530,  1535,  1538,  1541,  4538, 40498, 40646,
       40647, 40648, 40649, 40650, 40677, 40680, 40691, 40701, 40704,
       40900, 40982, 40983, 42193]

auto_ml_dids_val = [1547, 841, 40705, 799, 311, 40706, 1494, 40693,1552]

auto_ml_dids_test = [934, 1068, 50, 903, 837, 1542, 458, 715, 740]