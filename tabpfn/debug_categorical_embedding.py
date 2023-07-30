import torch
import torch.nn as nn
import torch.optim as optim
from scripts.transformer_prediction_interface import TabPFNClassifier
from meta_dataset_loader import load_OHE_dataset, meta_dataset_loader3
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def train(lr=0.00001, one_hot_encode=True):
    epochs = 100
    did = [23]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=1, only_inference=False)
    classifier.model[2].train()
    optimizer = optim.Adam(classifier.model[2].parameters(), lr=lr, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    datasets = load_OHE_dataset(did, one_hot_encode=one_hot_encode)
    support_dataset, query_dataset = meta_dataset_loader3(datasets[0], 0.5)
    label_encoder = LabelEncoder()
    
    for e in range(epochs):
        # y_query = label_encoder.fit_transform(y_query)
        for i in range(len(support_dataset)):
            
            x_support = support_dataset[i]['x']
            x_query = query_dataset[i]['x']
            y_support = support_dataset[i]['y']
            y_query = query_dataset[i]['y']
            
            classifier.fit(x_support, y_support)
            # y_eval, p_eval = classifier.predict(x_query, return_winning_probability=True)
            # accuracy = accuracy_score(y_query, y_eval)
                
            optimizer.zero_grad()
            prediction = classifier.predict_proba2(x_query)
            prediction = prediction.squeeze(0)
            loss = criterion(prediction,torch.from_numpy(y_query).to(device))
            print('epoch',e,'|','loss =',loss.item())
            loss.backward()
            optimizer.step()
    
    # classifier.fit(x_support, y_support)
    # y_eval, p_eval = classifier.predict(x_query, return_winning_probability=True)
    # accuracy = accuracy_score(y_query, y_eval)
    # print('acc =',accuracy)    
    
def main():
    train(one_hot_encode=True)

if __name__ == "__main__":
    main()        

