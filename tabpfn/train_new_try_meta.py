import torch
import torch.nn as nn
import torch.optim as optim
from scripts.transformer_prediction_interface import TabPFNClassifier
from meta_dataset_loader import load_OHE_dataset, meta_dataset_loader3, split_datasets
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from evaluate_classifier import evaluate_classifier2, open_cc_dids
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

def train(lr=0.00001, wandb_name='', num_augmented_datasets=0):
    
    epochs = 100
    
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
    })
    
    

    train_dids = [23, 46, 50, 333, 334, 335, 1552, 923, 934, 469, 1480, 825, 826, 947, 949, 950, 951, 40646, 40647, 40648, 40649, 40650, 40680, 40693, 40701, 40705, 40706, 40677, 1549, 1553, 42193]
    test_dids = [31]
    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=1, only_inference=False)
    
    datasets = load_OHE_dataset(train_dids, one_hot_encode=False, num_augmented_datasets=num_augmented_datasets, shuffle=False)
    
    support_dataset, query_dataset = meta_dataset_loader3(datasets)
    
    test_datasets = load_OHE_dataset(test_dids, shuffle=False, one_hot_encode=False)

    
    #training setup
    
    
    model = classifier.model[2]
    config = classifier.c
    criterion = model.criterion
    criterion.to(device)
    n_out = criterion.weight.shape[0]
    aggregate_k_gradients = config['aggregate_k_gradients']
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    #add schedular here
    
    
    print('Start training')
    
    model.train()
        
    for e in range(epochs):
        
        accumulator = 0
        
        for i in range(len(support_dataset)):
    
            X_full = np.concatenate([support_dataset[i]['x'], query_dataset[i]['x']], axis=0)
            X_full = torch.tensor(X_full, device=device,dtype=torch.float32, requires_grad=True).float().unsqueeze(1)
            # y_full = np.concatenate([support_dataset[0]['y'], np.zeros_like(query_dataset[0]['x'][:, 0])], axis=0)
            y_full = torch.tensor(support_dataset[i]['y'], device=device, dtype=torch.float32, requires_grad=True).float().unsqueeze(1)
            eval_pos = support_dataset[i]['x'].shape[0]
            num_classes = len(torch.unique(y_full))
            num_classes_query = len(np.unique(query_dataset[i]['y']))
            
            if num_classes > 1 and num_classes_query <= num_classes:
                X_full = preprocess_input(X_full, y_full, eval_pos)
                # print(X_full[0])
                X_full = torch.cat(
                        [X_full,
                        torch.zeros((X_full.shape[0], X_full.shape[1], max_features - X_full.shape[2])).to(device)], -1)
                
                criterion.weight=torch.ones(num_classes)
                
                output = model((None, X_full, y_full) ,single_eval_pos=eval_pos)[:, :, 0:num_classes] #TODO: check if we need to add some sort of style
                # output = torch.nn.functional.softmax(output, dim=-1)
            
                losses = criterion(output.reshape(-1, num_classes) , torch.from_numpy(query_dataset[i]['y']).to(device).long().flatten())
                losses = losses.view(*output.shape[0:2])
                
                loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)
                # loss = loss / aggregate_k_gradients
                
                accuracy = evaluate_classifier2(classifier, test_datasets)
                
                accumulator += loss.item()
                print('Epoch:', e, '|' "loss :", loss.item(), '|', "accuracy :", accuracy )
                did = support_dataset[i]['id']
                wandb.log({f"loss_{did}": loss.item(), "accuracy": accuracy})
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()    
            
            else:
                print('Skipping dataset', i, 'with only one class')
            # if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            #     try:
            #         optimizer.step()
            #         accuracy = evaluate_classifier2(classifier, datasets)
            #     except:
            #         print("Invalid optimization step encountered")
            #     optimizer.zero_grad()
            
        accumulator /= len(support_dataset)
        wandb.log({"average_loss": accumulator})
        
    
def main():
    train()

if __name__ == "__main__":
    main()    