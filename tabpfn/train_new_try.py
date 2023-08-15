import torch
import torch.nn as nn
import torch.optim as optim
from scripts.transformer_prediction_interface import TabPFNClassifier
from meta_dataset_loader import load_OHE_dataset, meta_dataset_loader3, split_datasets, shuffle_dataset_features
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from evaluate_classifier import evaluate_classifier2
import wandb

import utils as utils
from utils import normalize_data, to_ranking_low_mem, remove_outliers
from utils import NOP, normalize_by_used_features_f
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler

normalize_with_test= False
normalize_with_sqrt= False
normalize_to_ranking = False
max_features = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def preprocess_input(eval_xs, eval_ys, eval_position):
        import warnings

        if eval_xs.shape[1] > 1:
            raise Exception("Transforms only allow one batch dim - TODO")

        # eval_xs, eval_ys = normalize_data(eval_xs), normalize_data(eval_ys)
        eval_xs = normalize_data(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position)

        # Removing empty features
        eval_xs = eval_xs[:, 0, :]
        sel = [len(torch.unique(eval_xs[0:eval_ys.shape[0], col])) > 1 for col in range(eval_xs.shape[1])]
        eval_xs = eval_xs[:, sel]

        warnings.simplefilter('default')

        eval_xs = eval_xs.unsqueeze(1)

        # TODO: Cautian there is information leakage when to_ranking is used, we should not use it
        eval_xs = remove_outliers(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position) if not normalize_to_ranking else normalize_data(to_ranking_low_mem(eval_xs))
        # Rescale X
        eval_xs = normalize_by_used_features_f(eval_xs, eval_xs.shape[-1], max_features,
                                               normalize_with_sqrt=normalize_with_sqrt)

        return eval_xs.to(device)

def train(lr=0.00001, wandb_name='', num_augmented_datasets=0, epochs = 100, weight_decay=0.0):

    wandb.init(
    # set the wandb project where this run will be logged
    project="thesis",
    name=f"{wandb_name}_{num_augmented_datasets}_{lr}_{weight_decay}",
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "TabPFN",
    "dataset": "meta-dataset",
    "epochs": epochs,
    })
    
    
    train_dids = [1049]
    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=1, only_inference=False)
    
    datasets1 = load_OHE_dataset(train_dids, one_hot_encode=False, num_augmented_datasets=num_augmented_datasets, shuffle=False)
    datasets = [{'data':datasets1[0]['data'][:1024], 'target': datasets1[0]['target'][:1024], 'id': 0}]
    x_test = datasets1[0]['data'][1024:]
    y_test = datasets1[0]['target'][1024:]
    #training setup
    

    model = classifier.model[2]
    config = classifier.c
    criterion = model.criterion
    n_out = criterion.weight.shape[0]
    aggregate_k_gradients = config['aggregate_k_gradients']
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    #add schedular here
    
    predictions = np.zeros((len(y_test)))
    
    print('Start training')
    
    model.train()
    
    accuracy = evaluate_classifier2(classifier, datasets)
    print('accuracy before training =',accuracy)
        
    for batch in range(epochs):
        
        support_dataset, query_dataset = meta_dataset_loader3(datasets)

    
        X_full = np.concatenate([support_dataset[0]['x'], query_dataset[0]['x']], axis=0)
        X_full = torch.tensor(X_full, device=device,dtype=torch.float32, requires_grad=True).float().unsqueeze(1)
        y_full = torch.tensor(support_dataset[0]['y'], device=device, dtype=torch.float32, requires_grad=True).float().unsqueeze(1)
        eval_pos = support_dataset[0]['x'].shape[0]
        num_classes = len(torch.unique(y_full))
        
        X_full_test = np.concatenate([support_dataset[0]['x'], x_test], axis=0)
        X_full_test = torch.tensor(X_full_test, device=device,dtype=torch.float32, requires_grad=False).float().unsqueeze(1)
        X_full_test = preprocess_input(X_full_test, y_full, eval_pos)
        X_full_test = torch.cat(
                [X_full_test,
                    torch.zeros((X_full_test.shape[0], X_full_test.shape[1], max_features - X_full_test.shape[2])).to(device)], -1)
        
        X_full = preprocess_input(X_full, y_full, eval_pos)
        # print(X_full[0])
        X_full = torch.cat(
                [X_full,
                 torch.zeros((X_full.shape[0], X_full.shape[1], max_features - X_full.shape[2])).to(device)], -1)
        
        criterion.weight=torch.ones(num_classes)
        
        model.to(device)

        
        output = model((None, X_full, y_full) ,single_eval_pos=eval_pos)[:, :, 0:num_classes] #TODO: check if we need to add some sort of style
        # output = torch.nn.functional.softmax(output, dim=-1)
        losses = criterion(output.reshape(-1, num_classes), torch.from_numpy(query_dataset[0]['y']).to(device).long().flatten())
        losses = losses.view(*output.shape[0:2])
        
        with torch.no_grad():
            test_output = model((None, X_full_test, y_full) ,single_eval_pos=eval_pos)[:, :, 0:num_classes] #TODO: check if we need to add some sort of style
            test_acc = accuracy_score( torch.from_numpy(y_test).long().flatten().cpu(), torch.argmax(test_output.reshape(-1, num_classes).detach().cpu(), axis=1) )
            prediction = torch.argmax(test_output.reshape(-1, num_classes).detach().cpu(), axis=1)
            predictions = [predictions[i] if prediction[i] == y_test[i] else predictions[i]+1 for i in range(len(prediction))]
            
            test_losses = criterion(test_output.reshape(-1, num_classes), torch.from_numpy(y_test).to(device).long().flatten())
            test_losses = losses.view(*test_losses.shape[0:2])
            test_loss, test_nan_share = utils.torch_nanmean(test_losses.mean(0), return_nanshare=True)
            
        loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)
        acc = accuracy_score( torch.from_numpy(query_dataset[0]['y']).long().flatten().cpu(), torch.argmax(output.reshape(-1, num_classes).detach().cpu(), axis=1) )
        loss.backward()
        print('Batch:', batch, "loss :", loss.item(), "accuracy :", acc,"test_loss:", test_loss.item(), "test_acc :", test_acc, )
        wandb.log({ "loss": loss.item(), "accuracy": acc, "test_loss": test_loss.item(), "test_acc": test_acc})
        optimizer.step()
        # accuracy = evaluate_classifier2(classifier, datasets)    
        optimizer.zero_grad()  
    top= []
    for i in range(len(predictions)):
        if predictions[i]>100:
            top.append((i, predictions[i]))
    print(predictions)  
    print(top)
        
            
        
        
    
def main():
    train()

if __name__ == "__main__":
    main()    