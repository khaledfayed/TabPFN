from meta_dataset_loader import load_OHE_dataset, meta_dataset_loader
from scripts.transformer_prediction_interface import TabPFNClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
import time
from matplotlib import pyplot as plt
from evaluate_classifier import evaluate_classifier, open_cc_dids

def run_training(epochs=20, lr = 0.00001, num_samples_per_class=16):
    # settings:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #hyper parameters:
    batch_size = 32
    test_datasets = [22]

    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=4, only_inference=False)
    classifier.model[2].train()

    optimizer = optim.Adam(classifier.model[2].parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dids = [id for id in open_cc_dids if id not in test_datasets]
    print('=' * 15, 'Train Datasets','=' * 15, '\n', dids, '\n')   
    datasets = load_OHE_dataset(dids)
    # train_dataset= meta_dataset_loader(datasets)

    loss_history = []
    test_accuracy_history = []


    #print accuracy for test dataset
    test_accuracy_history.append(evaluate_classifier(classifier, test_datasets))

    #meta training loop
    for e in range(epochs):
        print('=' * 15, 'Epoch', e,'=' * 15)
        meta_data_loader = meta_dataset_loader(datasets, num_samples_per_class)
        
        accumulator = 0
        
        for i, batch in enumerate(meta_data_loader):
            x,y = batch['x'], batch['y']
            x_support, x_query = np.split(x,2)
            y_support, y_query = np.split(y,2)
            
            y_query = torch.from_numpy(y_query).to(device)
            classifier.fit(x_support, y_support)
            optimizer.zero_grad()
            prediction = classifier.predict_proba2(x_query)
            prediction = prediction.squeeze(0)
            loss = criterion(prediction,y_query)
            print('epoch',e,'|','loss =',loss.item()) if i%10 == 0 else None
            a = list(classifier.model[2].parameters())[9]
            loss.backward()
            optimizer.step()
            b = list(classifier.model[2].parameters())[9]
            is_equal =torch.equal(a.data, b.data)
            accumulator += loss.item()
                
        
        test_accuracy_history.append(evaluate_classifier(classifier, test_datasets))
        
        
        accumulator /= len(meta_data_loader)
        print('=' * 15, 'Accumulator', e,'=' * 15)     
        print('Accumulator =',accumulator)  
        
        loss_history.append(accumulator)
        

        #plot loss history
        plt.plot(loss_history)
        plt.show()

        #plot accuracy history
        plt.plot(test_accuracy_history)
        plt.show()

    # print('\n')

    # #meta testing loop
    # print('=' * 15, 'testing','=' * 15)

    # meta_data_loader = load_meta_data_loader(test_dataset)

    # with torch.no_grad():
    #     for batch in meta_data_loader:
    #         x,y = batch['x'], batch['y']
    #         x_support, x_query = np.split(x,2)
    #         y_support, y_query = np.split(y,2)
            
            
    #         if (len(np.unique(y_support))>1 and np.all(np.sort(np.unique(y_support)) == np.sort(np.unique(y_query)))):
    #             y_query = torch.from_numpy(y_query).to(device)
    #             classifier.fit(x_support, y_support)
    #             prediction = classifier.predict_proba2(x_query)
    #             prediction = prediction.squeeze(0)
    #             loss = criterion(prediction,y_query)
    #             print('loss =',loss.item()) 

    # print('\n')

    # #unseen data set accuracy 
    # print('=' * 15, 'unseen data set accuracy','=' * 15)

    # start = time.time()
    # classifier.fit(unseen_x_support, unseen_y_support)
    # y_eval, p_eval = classifier.predict(unseen_x_query, return_winning_probability=True)
    # print('Prediction time: ', time.time() - start, 'Accuracy', accuracy_score(unseen_y_query, y_eval), '\n')

    # #untuned TabPFN accuracy
    # print('=' * 15, 'control TabPFN accuracy','=' * 15)

    # control_classifier = TabPFNClassifier(device=device, N_ensemble_configurations=4, only_inference=True)

    # start = time.time()
    # control_classifier.fit(unseen_x_support, unseen_y_support)
    # y_eval, p_eval = control_classifier.predict(unseen_x_query, return_winning_probability=True)
    # print('Prediction time: ', time.time() - start, 'Accuracy', accuracy_score(unseen_y_query, y_eval), '\n')
    
def main():
    run_training()

if __name__ == "__main__":
    main()