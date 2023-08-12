import torch
import torch.nn as nn
import torch.optim as optim
from scripts.transformer_prediction_interface import TabPFNClassifier
from meta_dataset_loader import load_OHE_dataset, meta_dataset_loader3, split_datasets
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def train(lr=0.00001):
    epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=1, only_inference=False)
    classifier.model[2].train()
    optimizer = optim.Adam(classifier.model[2].parameters(), lr=lr, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    # datasets = load_OHE_dataset([1049])
    datasets = load_OHE_dataset([31], one_hot_encode=False, num_augmented_datasets=0, shuffle=False)
    train_dataset = {'data':datasets[0]['data'][:1024], 'target': datasets[0]['target'][:1024]}
    # val_dataset = {'data':datasets[0]['data'][1024:], 'target': datasets[0]['target'][1024:]}
    x_support = train_dataset['data'][:512]
    x_query = train_dataset['data'][512:1024]
    y_support = train_dataset['target'][:512]
    y_query = train_dataset['target'][512:1024]
    
    support_dataset, query_dataset = meta_dataset_loader3(datasets, shuffle=False)
    x_support = support_dataset[0]['x']
    x_query = query_dataset[0]['x']
    y_support = support_dataset[0]['y']
    y_query = query_dataset[0]['y']

    # x_test = val_dataset['data']
    # y_test = val_dataset['target']
    label_encoder = LabelEncoder()
    
    for e in range(epochs):
        # y_query = label_encoder.fit_transform(y_query)
        classifier.fit(x_support, y_support)
        with torch.no_grad():
        
            y_eval, p_eval = classifier.predict(x_query, return_winning_probability=True)
            query_accuracy = accuracy_score(y_query, y_eval)
            # y_eval, p_eval = classifier.predict(x_test, return_winning_probability=True)
            # test_accuracy = accuracy_score(y_test, y_eval)
            

        prediction = classifier.predict_proba2(x_query)
        
        prediction = prediction.squeeze(0)
        print(torch.argmax(prediction.detach().cpu(), axis=1) )
        loss = criterion(prediction,torch.from_numpy(y_query).to(device))
        print('epoch',e,'|','loss =',loss.item(),'|','query_acc =',query_accuracy)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()        
    
def main():
    train()

if __name__ == "__main__":
    main()        

