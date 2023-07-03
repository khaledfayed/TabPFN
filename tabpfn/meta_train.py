from meta_dataset_loader import load_OHE_dataset, meta_dataset_loader2
from scripts.transformer_prediction_interface import TabPFNClassifier
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
import time
from matplotlib import pyplot as plt
from evaluate_classifier import evaluate_classifier, open_cc_dids
import wandb

def run_training(epochs=20, lr = 0.00001, num_samples_per_class=16, num_augmented_datasets=0, query_batch_size=16, support_batch_size=32, weight_decay=0, wandb_name='' ):
    
    wandb.init(
    # set the wandb project where this run will be logged
    project="thesis",
    name=f"{wandb_name}_{support_batch_size}_{query_batch_size}_{num_augmented_datasets}_{lr}_{weight_decay}",
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "TabPFN",
    "dataset": "meta-dataset",
    "epochs": epochs,
    'query_batch_size': query_batch_size,
    'support_batch_size': support_batch_size,
    }
)
    # settings:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #hyper parameters:
    batch_size = 32
    test_datasets = [22]

    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=4, only_inference=False)
    classifier.model[2].train()

    optimizer = optim.Adam(classifier.model[2].parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    dids = [id for id in open_cc_dids if id not in test_datasets]
    print('=' * 15, 'Train Datasets','=' * 15, '\n', dids, '\n')   
    datasets = load_OHE_dataset(dids, num_augmented_datasets)
    # train_dataset= meta_dataset_loader(datasets)

    loss_history = []
    test_accuracy_history = {id:[] for id in test_datasets}
    for id in test_datasets:
        test_accuracy_history[id] = test_accuracy_history[id] + [evaluate_classifier(classifier, [id])]


    #print accuracy for test dataset
    # test_accuracy_history.append(evaluate_classifier(classifier, test_datasets))
    
    start_time = time.time()
    label_encoder = LabelEncoder()
    #meta training loop
    for e in range(epochs):
        print('=' * 15, 'Epoch', e,'=' * 15)
        support_dataset, query_dataset = meta_dataset_loader2(datasets, query_batch_size, support_batch_size)
        
        accumulator = 0
        
        for i in range(len(support_dataset)):
    
            x_support, y_support = support_dataset[i].values()
            x_query, y_query = query_dataset[i].values()
            
            y_query = label_encoder.fit_transform(y_query)

            # if (len(np.unique(y_support))>0 and np.all(np.sort(np.unique(y_support)) == np.sort(np.unique(y_query)))):
            if(len(np.unique(y_support))>1):
                
                classifier.fit(x_support, y_support)
                optimizer.zero_grad()
                prediction = classifier.predict_proba2(x_query)
                prediction = prediction.squeeze(0)
                loss = criterion(prediction,torch.from_numpy(y_query).to(device))
                print('epoch',e,'|','loss =',loss.item()) if i%10 == 0 else None
                loss.backward()
                optimizer.step()
                accumulator += loss.item()
                    
                # classifier.fit(x_query, y_query)
                # optimizer.zero_grad()
                # prediction = classifier.predict_proba2(x_support)
                # prediction = prediction.squeeze(0)
                # loss = criterion(prediction,torch.from_numpy(y_support).to(device))
                # loss.backward()
                # optimizer.step()
                # accumulator += loss.item()
                
        
        for id in test_datasets:
            test_accuracy_history[id] = test_accuracy_history[id] + [evaluate_classifier(classifier, [id])]
        
        
        accumulator /= len(support_dataset)
        print('=' * 15, 'Accumulator', e,'=' * 15)     
        print('Accumulator =',accumulator)  
        
        loss_history.append(accumulator)
        
        for id in test_datasets:
            wandb.log({f"accuracy dataset {id}": test_accuracy_history[id][e]})
        
        wandb.log({"loss": loss_history[e]})

        #plot loss history
        plt.plot(loss_history)
        plt.savefig(f"plots/{start_time}_loss_history.png")
        plt.show()

        #plot accuracy history
        for id in test_datasets:
            plt.plot(test_accuracy_history[id], label = f"dataset {id}")
            plt.axhline(y=test_accuracy_history[id][0], linestyle='dashed' , color='grey')
        plt.savefig(f"plots/{start_time}_accuracy_history.png")
        plt.legend()
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
    run_training(epochs=50, weight_decay=0.01)

if __name__ == "__main__":
    main()