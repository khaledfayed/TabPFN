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
    
    train_dids = [1068]
    test_dids = [31]
    epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

