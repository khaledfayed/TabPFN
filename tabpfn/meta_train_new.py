import torch
import torch.nn as nn
import torch.optim as optim
from scripts.transformer_prediction_interface import TabPFNClassifier
from meta_dataset_loader import load_OHE_dataset, meta_dataset_loader3, split_datasets
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from evaluate_classifier import evaluate_classifier2
import wandb

def train(lr=0.00001, wandb_name='', num_augmented_datasets=100):
    
    epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    wandb.init(
    # set the wandb project where this run will be logged
    project="thesis",
    name=f"{wandb_name}_{num_augmented_datasets}_{lr}",
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "TabPFN",
    "dataset": "meta-dataset",
    "epochs": epochs,
    }
)
    
    train_dids = [23, 46, 50, 333, 334, 335, 1552, 923, 934, 469, 1480, 825, 826, 947, 949, 950, 951, 40646, 40647, 40648, 40649, 40650, 40680, 40693, 40701, 40705, 40706, 40677, 1549, 1553, 42193]
    # train_dids = [11, 23, 28, 30, 37, 44, 46, 50, 60, 181, 182, 311, 333, 334, 335, 1547, 1049, 1069, 4538, 1552, 1050, 799, 1056, 715, 725, 728, 735, 737, 740, 752, 772, 1068, 803, 807, 816, 819, 833, 837, 847, 1507, 871, 903, 923, 934, 1466, 1475, 1487, 1494, 761, 1496, 1497, 1504, 375, 377, 458, 469, 1063, 1462, 1480, 1510, 40496, 717, 750, 770, 825, 826, 841, 884, 886, 920, 936, 937, 947, 949, 950, 951, 40646, 40647, 40648, 40649, 40650, 40680, 40693, 40701, 40704, 40705, 40706, 40983, 40994, 40982, 40498, 40677, 40691, 40900, 1528, 1529, 1530, 1535, 1538, 1541, 1542, 1549, 1553, 42193]
    test_dids = [31]
    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=1, only_inference=False)
    classifier.model[2].train()
    optimizer = optim.Adam(classifier.model[2].parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    datasets = load_OHE_dataset(train_dids, one_hot_encode=False, num_augmented_datasets=num_augmented_datasets)
    
    support_dataset, query_dataset = meta_dataset_loader3(datasets)
    
    test_datasets = load_OHE_dataset(test_dids, shuffle=False, one_hot_encode=False)
        
    loss_history = []
    # label_encoder = LabelEncoder() 
    
    for e in range(epochs):
        
        # accumulator = 0
        
        for i in range(len(support_dataset)):
            
            accuracy = evaluate_classifier2(classifier, test_datasets)
            wandb.log({'accuracy': accuracy})
        
            x_support = support_dataset[i]['x']
            y_support = support_dataset[i]['y']
            x_query = query_dataset[i]['x']
            y_query = query_dataset[i]['y']    
            
            # y_query = label_encoder.fit_transform(y_query)
            classifier.fit(x_support, y_support)
                
            optimizer.zero_grad()
            prediction = classifier.predict_proba2(x_query)
            prediction = prediction.squeeze(0)
            loss = criterion(prediction,torch.from_numpy(y_query).to(device))
            print('epoch',e,'|','loss =',loss.item(), '|','accuracy =', accuracy)
            loss.backward()
            optimizer.step()
            wandb.log({'loss': loss.item()})
            # accumulator += loss.item()
            
        # accumulator /= len(support_dataset)
        # loss_history.append(accumulator)    
    
def main():
    train()

if __name__ == "__main__":
    main()        

