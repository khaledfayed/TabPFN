import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils import normalize_data, to_ranking_low_mem, remove_outliers, get_cosine_schedule_with_warmup, get_restarting_cosine_schedule_with_warmup
from utils import normalize_by_used_features_f, torch_nanmean
from scripts.transformer_prediction_interface import TabPFNClassifier
import numpy as np
from sklearn.preprocessing import LabelEncoder
from evaluate_classifier import evaluate_classifier2, auto_ml_dids_train, auto_ml_dids_val
import argparse
# import wandb



normalize_with_test= False
normalize_with_sqrt= False
normalize_to_ranking = False
max_features = 100
warmup_epochs=20
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


class MetaNet(pl.LightningModule):
    def __init__(self, classifier, lr, weight_decay, augmentation_config = [], wandb_name = ''):
        super(MetaNet, self).__init__()
        self.best_accuracy_so_far = 0
        self.classifier = classifier
        self.model = classifier.model[2]
        self.config = classifier.c
        self.criterion =  self.model.criterion
        self.aggregate_k_gradients = self.config['aggregate_k_gradients']
        self.lr = lr
        self.weight_decay = weight_decay
        self.augmentation_config = augmentation_config
        self.accumulator = 0
        
        # if device != 'cpu': wandb.init(
        # project="thesis",
        # name=f"{wandb_name}_{lr}",
        # config={
        # "learning_rate": lr,
        # "architecture": "TabPFN",
        # "dataset": "meta-dataset",
        # })

    def forward(self, X_full, y_full, eval_pos, num_classes):
        x = self.model((None, X_full, y_full) ,single_eval_pos=eval_pos)[:, :, 0:num_classes]
        return x

    def training_step(self, batch, batch_idx):
        # x, y = batch
        # logits = self(x)
        # loss = self.loss(logits, y)
        # return loss
        support_batch, query_batch = batch
        
        print(1)
        
        X_full = np.concatenate([support_batch['x'], query_batch['x']], axis=0)
        X_full = torch.tensor(X_full, device=device,dtype=torch.float32, requires_grad=False).float().unsqueeze(1)
        y_full = np.concatenate([support_batch['y'], np.zeros_like(query_batch['x'][:, 0])], axis=0)
        y_full = torch.tensor(support_batch['y'], device=device, dtype=torch.float32, requires_grad=True).float().unsqueeze(1)
        eval_pos = support_batch['x'].shape[0]
        print(2)
        num_classes = len(torch.unique(y_full))
        num_classes_query = len(np.unique(query_batch['y']))
        print(3)
                
        X_full = preprocess_input(X_full, y_full, eval_pos)
        X_full.requires_grad=True
        X_full = torch.cat(
                [X_full,
                torch.zeros((X_full.shape[0], X_full.shape[1], max_features - X_full.shape[2])).to(device)], -1)
        print(4)
        self.criterion.weight=torch.ones(num_classes)
        print(5)
        self.model.to(device)
        print(6)
        label_encoder = LabelEncoder()
        print(7)
        output = self(X_full, y_full, eval_pos, num_classes)
        print(8)
        losses = self.criterion(output.reshape(-1, num_classes) , torch.from_numpy(label_encoder.fit_transform(query_batch['y'])).to(device).long().flatten())
        losses = losses.view(*output.shape[0:2])
        print(9)
        loss, _ = torch_nanmean(losses.mean(0), return_nanshare=True)
        print(10)
        # self.accumulator += loss.item()
        print(11)
        # did = support_batch['id']
        # if device != 'cpu': wandb.log({f"loss_{did}": loss.item()})
        print(12)
        return loss
        
        

    def validation_step(self, batch, batch_idx):
        accuracy = evaluate_classifier2(self.classifier, test_datasets, log= device != 'cpu')

    def configure_optimizers(self):
        optimizer =  torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, self.trainer.max_epochs)
        
        return {'optimizer': optimizer,'lr_scheduler': scheduler}

    
    def on_train_epoch_end(self):
        self.accumulator /= self.trainer.train_dataloader.num_batches
        # if device != 'cpu': wandb.log({"average_loss": self.accumulator, "lr":  self.optimizers().param_groups[0]['lr']})
        self.accumulator = 0


from torch.utils.data import DataLoader
from meta_dataset_loader import load_OHE_dataset,meta_dataset_loader3
from evaluate_classifier import auto_ml_dids_train

class TrainDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(TrainDataLoader, self).__init__(*args, **kwargs)
        self.num_batches = 0

    
    def __iter__(self):
        support_dataset, query_dataset = meta_dataset_loader3(datasets)
        self.num_batches = len(support_dataset)
        return iter(zip(support_dataset, query_dataset))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script with command-line arguments")
    parser.add_argument("--epochs", type=int, help="The first argument (an integer)")
    parser.add_argument("--lr", type=float, help="The first argument (an integer)")
    parser.add_argument("--weight_decay", type=float, help="The first argument (an integer)")
    parser.add_argument("--name", type=str, help="The first argument (an integer)")
    args = parser.parse_args()
    
    config = [('relabel', 2), ('drop_features', 1),('shuffle_features', 1)]
    # config = [('shuffle_features', 1)]
    # config = [('drop_features', 1)]
    # config = []
    

    
    # train(wandb_name=args.name, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, augmentation_config=config)  

    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=1, only_inference=False)

    metaNet = MetaNet(classifier, lr=0.0001, weight_decay=0.0001)
    
    datasets = load_OHE_dataset(auto_ml_dids_train, one_hot_encode=False)
    test_datasets = load_OHE_dataset(auto_ml_dids_val, one_hot_encode=False)

    train_loader = TrainDataLoader(datasets, num_workers=0)
    
    trainer = pl.Trainer(max_epochs=1000, log_every_n_steps=1, use_distributed_sampler=False, reload_dataloaders_every_n_epochs=1, limit_train_batches=10)
    trainer.fit(metaNet, train_loader, train_loader)


    # with torch.no_grad():
        # evaluate_classifier2(classifier, test_datasets, log= device != 'cpu')
        
    #  accumulator = 0
    #     cloned_datasets = copy.deepcopy(datasets)
    #     augment_datasets(cloned_datasets, augmentation_config)
    #     # generate_datasets_gaussian(cloned_datasets)
    #     support_dataset, query_dataset = meta_dataset_loader3(cloned_datasets)
    # accumulator += loss.item()
    # loss.backward()
    
    #  if i % aggregate_k_gradients == aggregate_k_gradients - 1:
    #                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    #                 try:
    #                     optimizer.step()
    
    #save
    #augment

