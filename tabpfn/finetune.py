import torch
import torch.optim as optim
from scripts.transformer_prediction_interface import TabPFNClassifier
from scripts.model_builder import save_model
from meta_dataset_loader import load_OHE_dataset, meta_dataset_loader3, augment_datasets, generate_datasets_gaussian
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from evaluate_classifier import evaluate_classifier2, open_cc_dids, auto_ml_dids_train, auto_ml_dids_val, auto_ml_dids_train_full
import wandb
import copy

import utils as utils
from utils import normalize_data, to_ranking_low_mem, remove_outliers, get_cosine_schedule_with_warmup, get_restarting_cosine_schedule_with_warmup
from utils import normalize_by_used_features_f

import argparse

normalize_with_test= False
normalize_with_sqrt= False
normalize_to_ranking = False
max_features = 100
warmup_epochs=0
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

def train(lr=0.0001, wandb_name='', num_augmented_datasets=0, epochs = 100, weight_decay=0.0, augmentation_config = []):
    
    if device != 'cpu': wandb.init(
        project="thesis",
        name=f"{wandb_name}_{num_augmented_datasets}_{lr}",
        config={
        "learning_rate": lr,
        "architecture": "TabPFN",
        "dataset": "meta-dataset",
        "epochs": epochs,
        })
    
    print(augmentation_config)
    
    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=1, only_inference=False)

        
    # datasets = load_OHE_dataset(auto_ml_dids_train, one_hot_encode=False)
    datasets = load_OHE_dataset([715],one_hot_encode=False)
    
    dataset = datasets[0]
    rng = np.random.default_rng(seed=42)
    dataset_length = len(dataset['data'])
    dataset_indices = np.arange(dataset_length)
    rng.shuffle(dataset_indices)
    dataset['data'] = dataset['data'][dataset_indices]
    dataset['target'] = dataset['target'][dataset_indices]
    
    dataset['data'] = dataset['data'][:512]
    dataset['target'] = dataset['target'][:512]
    
    
    test_datasets = load_OHE_dataset([715], one_hot_encode=False)
    test_dataset = test_datasets[0]
    test_dataset_length = len(test_dataset['data'])
    test_dataset_indices = np.arange(test_dataset_length)
    rng.shuffle(test_dataset_indices)
    test_dataset['data'] = test_dataset['data'][test_dataset_indices]
    test_dataset['target'] = test_dataset['target'][test_dataset_indices]
    
    
    
    #training setup
    best_accuracy_so_far = 0
    
    
    model = classifier.model[2]
    config = classifier.c
    criterion = model.criterion
    aggregate_k_gradients = config['aggregate_k_gradients']
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, epochs if epochs is not None else 100) # when training for fixed time lr schedule takes 100 steps
    # scheduler = get_restarting_cosine_schedule_with_warmup(optimizer, warmup_epochs, 500, 500)
    
    # if torch.cuda.device_count() > 1:
    #     print('obaa')
    #     model = torch.nn.DataParallel(model)
        
    print('Start training')
    # with torch.no_grad():
    #     evaluate_classifier2(classifier, test_datasets, log= device != 'cpu')
    
    model.train()
    
#     for _, param in model.named_parameters():
#         param.requires_grad = False

# # Unfreeze the layers you want to train
#     for name, param in list(model.named_parameters())[-32:]:
#         param.requires_grad = True
        
    for e in range(epochs):
        
        accumulator = 0
        cloned_datasets = copy.deepcopy(datasets)
        augment_datasets(cloned_datasets, augmentation_config)
        # generate_datasets_gaussian(cloned_datasets)
        support_dataset, query_dataset = meta_dataset_loader3(cloned_datasets, batch_size=256)
        
        for i in range(len(support_dataset)):
                
            X_full = np.concatenate([support_dataset[i]['x'], query_dataset[i]['x']], axis=0)
            X_full = torch.tensor(X_full, device=device,dtype=torch.float32, requires_grad=False).float().unsqueeze(1)
            y_full = np.concatenate([support_dataset[0]['y'], np.zeros_like(query_dataset[0]['x'][:, 0])], axis=0)
            y_full = torch.tensor(support_dataset[i]['y'], device=device, dtype=torch.float32, requires_grad=True).float().unsqueeze(1)
            eval_pos = support_dataset[i]['x'].shape[0]
            num_classes = len(torch.unique(y_full))
            num_classes_query = len(np.unique(query_dataset[i]['y']))
                
            if num_classes > 1 and num_classes_query <= num_classes:
                
                X_full = preprocess_input(X_full, y_full, eval_pos)
                X_full.requires_grad=True
                X_full = torch.cat(
                        [X_full,
                        torch.zeros((X_full.shape[0], X_full.shape[1], max_features - X_full.shape[2])).to(device)], -1)
                
                criterion.weight=torch.ones(num_classes)
                
                model.to(device)
                
                label_encoder = LabelEncoder()
                
                output = model((None, X_full, y_full) ,single_eval_pos=eval_pos)[:, :, 0:num_classes] #TODO: check if we need to add some sort of style
                    
                losses = criterion(output.reshape(-1, num_classes) , torch.from_numpy(label_encoder.fit_transform(query_dataset[i]['y'])).to(device).long().flatten())
                losses = losses.view(*output.shape[0:2])
                
                loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)
                
                accumulator += loss.item()
                    
                did = support_dataset[i]['id']
                if device != 'cpu': wandb.log({f"loss_{did}": loss.item()})
                
                
                loss.backward()
                
                try:
                    optimizer.step()
                    # if e > warmup_epochs:
                    #     scheduler.step()
                    #     if device != 'cpu': wandb.log({"lr":  optimizer.param_groups[0]['lr']})
                except:
                    print("Invalid optimization step encountered")
                
                optimizer.zero_grad()
                # optimizer.step()
                # optimizer.zero_grad()    
            
            else:
                print('Skipping dataset', i, 'with only one class')
            
        accumulator /= len(support_dataset)

        if device != 'cpu': wandb.log({"average_loss": accumulator})
        
        # if e <= warmup_epochs:
        scheduler.step()
        if device != 'cpu': wandb.log({"lr":  optimizer.param_groups[0]['lr']})
        
        # if e % 1000 == 0:
        

        

        
        if e % 100 == 0:
            criterion.weight=torch.ones(10)
            model_save_name = f'{wandb_name}_e_{e}_lr_{lr}'
            checkpoint = f'prior_diff_real_checkpoint_{model_save_name}_n_0_epoch_100.cpkt'
            save_model(model, 'models_diff/', checkpoint, config)
            

                
        wandb.log({"epoch": e})
        print('Epoch:', e, 'loss:', accumulator)
        
        with torch.no_grad():
                
            fit_test_data = test_dataset['data'][:512]
            fit_test_target = test_dataset['target'][:512]
            classifier.fit(fit_test_data, fit_test_target)
            y_eval, p_eval = classifier.predict(test_dataset['data'][512:], return_winning_probability=True)
            accuracy = accuracy_score(test_dataset['target'][512:], y_eval)
            print(accuracy)
            
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script with command-line arguments")
    parser.add_argument("--epochs", type=int, help="The first argument (an integer)")
    parser.add_argument("--lr", type=float, help="The first argument (an integer)")
    parser.add_argument("--weight_decay", type=float, help="The first argument (an integer)")
    parser.add_argument("--name", type=str, help="The first argument (an integer)")
    args = parser.parse_args()
    
    # config = [('relabel', 2), ('drop_features', 1),('shuffle_features', 1)]
    config = [('shuffle_features', 1)]
    # config = [('drop_features', 1)]
    # config = []
    # config = [('relabel', 2)]
    

    
    train(wandb_name=args.name, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, augmentation_config=config)    
    # train()