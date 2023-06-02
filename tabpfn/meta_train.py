from meta_dataset_loader import load_meta_data_loader, split_datasets
from scripts.transformer_prediction_interface import TabPFNClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from matplotlib import pyplot as plt

# settings:
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#hyper parameters:
epochs = 2
lr = 0.00001
batch_size = 32

classifier = TabPFNClassifier(device=device, N_ensemble_configurations=4, only_inference=False)
classifier.model[2].train()

optimizer = optim.Adam(classifier.model[2].parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train_dataset, test_dataset = split_datasets() #not needed

loss_history = []

#TODO: print accuracy for wine


#meta training loop
for parameters in classifier.model[2].parameters():
    print(parameters.grad)

for e in range(epochs):
    print('=' * 15, 'Epoch', e,'=' * 15)
    meta_data_loader = load_meta_data_loader(train_dataset)
    
    accumulator = 0
    
    for batch in meta_data_loader:
        x,y = batch['x'], batch['y']
        x_support, x_query = np.split(x,2)
        y_support, y_query = np.split(y,2)
        
        y_query = torch.from_numpy(y_query)
        
        if (len(np.unique(y_support))>0 and np.all(np.sort(np.unique(y_support)) == np.sort(np.unique(y_query)))):
            classifier.fit(x_support, y_support)
            optimizer.zero_grad()
            prediction = classifier.predict_proba2(x_query)
            prediction = prediction.squeeze(0)
            loss = criterion(prediction,y_query)
            print('loss =',loss.item())
            a = list(classifier.model[2].parameters())[9]
            loss.backward()
            optimizer.step()
            b = list(classifier.model[2].parameters())[9]
            is_equal =torch.equal(a.data, b.data)
            accumulator += loss.item()
            
    #TODO: print accuracy for wine        
      
    accumulator /= len(meta_data_loader)
    print('=' * 15, 'Accumulator', e,'=' * 15)     
    print('Accumulator =',loss.item())  
    
    loss_history.append(accumulator)

#plot loss history
plt.plot(loss_history)
plt.savefig('fig.png')
plt.show()

print('\n')

#meta testing loop
print('=' * 15, 'testing','=' * 15)

meta_data_loader = load_meta_data_loader(test_dataset)

with torch.no_grad():
    for batch in meta_data_loader:
        x,y = batch['x'], batch['y']
        x_support, x_query = np.split(x,2)
        y_support, y_query = np.split(y,2)
        
        y_query = torch.from_numpy(y_query)
        
        if (len(np.unique(y_support))>1 and np.all(np.sort(np.unique(y_support)) == np.sort(np.unique(y_query)))):
            classifier.fit(x_support, y_support)
            prediction = classifier.predict_proba2(x_query)
            prediction = prediction.squeeze(0)
            loss = criterion(prediction,y_query)
            print('loss =',loss.item()) 

print('\n')

#unseen data set accuracy 
print('=' * 15, 'unseen data set accuracy','=' * 15)

unseen_x, unseen_y = load_wine(return_X_y=True)
unseen_x_support, unseen_x_query, unseen_y_support, unseen_y_query = train_test_split(unseen_x, unseen_y, train_size=30, test_size=10, random_state=2)

start = time.time()
classifier.fit(unseen_x_support, unseen_y_support)
y_eval, p_eval = classifier.predict(unseen_x_query, return_winning_probability=True)
print('Prediction time: ', time.time() - start, 'Accuracy', accuracy_score(unseen_y_query, y_eval), '\n')

#untuned TabPFN accuracy
print('=' * 15, 'control TabPFN accuracy','=' * 15)

control_classifier = TabPFNClassifier(device=device, N_ensemble_configurations=4, only_inference=True)

start = time.time()
control_classifier.fit(unseen_x_support, unseen_y_support)
y_eval, p_eval = control_classifier.predict(unseen_x_query, return_winning_probability=True)
print('Prediction time: ', time.time() - start, 'Accuracy', accuracy_score(unseen_y_query, y_eval), '\n')