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
from torchinfo import summary
from torchview import draw_graph



# settings:
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#hyper parameters:
epochs = 2
lr = 0.00001
batch_size = 32

classifier = TabPFNClassifier(device=device, N_ensemble_configurations=4, only_inference=False)
classifier.model[2].train()

# optimizer = optim.Adam(classifier.model[2].parameters(), lr=lr)
# criterion = nn.CrossEntropyLoss()

unseen_x, unseen_y = load_wine(return_X_y=True)
unseen_x_support, unseen_x_query, unseen_y_support, unseen_y_query = train_test_split(unseen_x, unseen_y, train_size=30, test_size=10, random_state=2)

classifier.fit(unseen_x_support, unseen_y_support)
y_eval, p_eval = classifier.predict(unseen_x_query, return_winning_probability=True)
# print('Prediction time: ', time.time() - start, 'Accuracy', accuracy_score(unseen_y_query, y_eval), '\n')


x = torch.randn(40, 4, 100)
y = torch.randn(30, 100)


# summary(classifier.model[2])
graph = draw_graph(classifier.model[2], input_data=([x,y]))
graph.visual_graph()



# for parameters in classifier.model[2].parameters():
#     print(parameters.grad)


