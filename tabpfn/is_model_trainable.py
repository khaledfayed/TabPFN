from meta_dataset_loader import load_meta_data_loader, split_datasets
from scripts.transformer_prediction_interface import TabPFNClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from matplotlib import pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(torch.cuda)

#hyper parameters:
epochs = 2
lr = 0.00001
batch_size = 32

classifier = TabPFNClassifier(device=device, N_ensemble_configurations=4, only_inference=False)
classifier.model[2].train()

optimizer = optim.Adam(classifier.model[2].parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train_dataset, test_dataset = split_datasets()

loss_history = []

#meta training loop
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=40, test_size=10, random_state=2)

start = time.time()
classifier.fit(X_train, y_train)
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)
print('Prediction time: ', time.time() - start, 'Accuracy', accuracy_score(y_test, y_eval), '\n')

classifier.model[2].train()

# for parameter in classifier.model[2].parameters():
#     print(parameter.requires_grad, parameter.grad_fn)
#     parameter.retain_grad()

y_test = torch.from_numpy(y_test)
optimizer = optim.Adam(classifier.model[2].parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

for e in range (20):
    optimizer.zero_grad()
    classifier.fit(X_train, y_train)
    prediction = classifier.predict_proba2(X_test)
    prediction = prediction.squeeze(0)
    loss = criterion(prediction,y_test.to(device))
    print(loss.item())
    loss.backward()
    optimizer.step()
    print()
    
start = time.time()
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)
print('Prediction time: ', time.time() - start, 'Accuracy', accuracy_score(y_test, y_eval), '\n')

