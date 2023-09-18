import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, fetch_openml
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import openml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import random


from numpy.random import default_rng

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os

openml.config.cache_directory = os.path.expanduser('./openml')

class TabularModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TabularModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        return x

# Assume you have your tabular data loaded into X and y

def generate_datasets(datasets, device='cpu'):

    # Instantiate the model
    print(device)

    model = TabularModel(1, 1).to(device)
    
    for i,dataset in enumerate(datasets):
        
        delattr(model, 'fc1')
        model.add_module('fc1', nn.Linear(dataset['data'].shape[1], np.random.randint(4, 101)))
        model.fc1.reset_parameters()
        model.fc1.to(device)
        
        X = torch.tensor(dataset['data'], dtype=torch.float32).to(device)
        
        output = model(X)
        
        datasets[i]['data'] = output.detach().cpu().numpy()
        
def shuffle_dataset_features(transformed_data):
    _, num_cols = transformed_data.shape
    augmented_data = transformed_data.copy()
    shuffled_columns = np.random.permutation(num_cols)
    augmented_data = augmented_data[:, shuffled_columns]
    
    return augmented_data

def zero_pad_dataset_features(transformed_data):
    padding_amount = 100 - transformed_data.shape[1]  # Adjust this value based on your needs
    padded_data = np.zeros((transformed_data.shape[0], transformed_data.shape[1] + padding_amount))
    padded_data[:, :transformed_data.shape[1]] = transformed_data
    
    return padded_data

def log_scale_features(data, epsilon=1e-10):
    data_with_epsilon = data + epsilon
    
    return np.log(data_with_epsilon)

def exponential_scale_features(data, power=2):
    
    return np.power(data, power)


def drop_dataset_features(input_array):
    
    num_features = input_array.shape[1]
    
    min_columns_to_drop = int(0.10 * num_features)
    max_columns_to_drop = int(0.20 * num_features)
    num_columns_to_drop = np.random.randint(min_columns_to_drop, max_columns_to_drop + 1)
    
    columns_to_drop_indices = np.random.choice(num_features, num_columns_to_drop, replace=False)
    
    result_array = np.delete(input_array, columns_to_drop_indices, axis=1)
    
    return result_array

def relabel_augmentation(features, labels, num_categorical_features):
    # Randomly select a feature to be used as the new label
    num_features = features.shape[1]
    new_label_index = np.random.randint(0, num_features)

    # Extract the selected feature as the new label
    new_labels = features[:, new_label_index]
    
    features = np.delete(features, new_label_index, axis=1)

    # Create a new feature using the old label
    old_label = labels.reshape(-1, 1)

    # Add the old label as a new feature
    augmented_features = np.hstack((features, old_label))
    
    if new_label_index > num_categorical_features:
        median = np.median(new_labels)
        min_value = np.min(new_labels)
        max_value = np.max(new_labels)
        
        value_range = max_value - min_value
        class_width = value_range / np.unique(old_label).shape[0]-1
        
        new_labels = ((new_labels - min_value) / class_width).astype(int)

    # Encode the new labels using LabelEncoder
    label_encoder = LabelEncoder()
    new_labels_encoded = label_encoder.fit_transform(new_labels)

    return augmented_features, new_labels_encoded

augmentation_dict = {
    'drop_features': drop_dataset_features,
    'shuffle_features': shuffle_dataset_features,
    'log_scaling': log_scale_features,
    'exp_scaling': exponential_scale_features,
    'relabel': relabel_augmentation
}

def load_OHE_dataset(dids, num_augmented_datasets=0, one_hot_encode=True, shuffle=True, drop_features=False, augmentation_config=[]):
    encoder = OneHotEncoder() if one_hot_encode else OrdinalEncoder()
    label_encoder = LabelEncoder()

    datasets = openml.datasets.get_datasets(dids)
    
    encoded_datasets = []
    
    for dataset in datasets:
        X, y, categorical_features, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute,
        )
        
        num_categorical_features = np.sum(categorical_features)
                
        df = pd.DataFrame(X, columns=attribute_names)
        
        if drop_features:
            random = np.random.randint(len(attribute_names),size=3)
            dropped_columns = np.array(attribute_names)[random]
            
            for column in dropped_columns:
                index = attribute_names.index(column)
                attribute_names.remove(column)
                del categorical_features[index]
            
            df.drop(dropped_columns, axis=1, inplace=True)
        
                
        categorical_columns = [col for col, is_categorical in zip(attribute_names, categorical_features) if is_categorical]
        
        preprocessor = ColumnTransformer(
            transformers=[('cat', encoder, categorical_columns)],
            remainder='passthrough'
        )
        
        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        
        transformed_data = pipeline.fit_transform(df)
        transformed_targets = label_encoder.fit_transform(y)
        
        for i, (augmentation, amount) in enumerate(augmentation_config):
            augmentation_function = augmentation_dict[augmentation]
            for _ in range(amount):
                if augmentation == 'relabel':
                    relabelled_data, relabelled_targets = augmentation_function(transformed_data, transformed_targets, num_categorical_features)
                    encoded_datasets.append({'data': relabelled_data, 'target': relabelled_targets, 'id': dataset.id, 'augmentation': augmentation })
                else:
                    encoded_datasets.append({'data': augmentation_function(transformed_data), 'target': transformed_targets, 'id': dataset.id, 'augmentation': augmentation })
                    
        encoded_datasets.append({'data': shuffle_dataset_features(transformed_data) if shuffle else transformed_data, 'target': transformed_targets, 'id': dataset.id, 'augmentation': 'none'})
        
        for i in range(num_augmented_datasets):
            encoded_datasets.append({'data': shuffle_dataset_features(transformed_data), 'target': transformed_targets, 'id': f"dataset.id_augmented_{i}"})
    
    return encoded_datasets

def split_datasets(datasets, test_percentage=0.2):
    
    train_datasets = [{ 'id': dataset['id'], 'data': dataset['data'][:int((1-test_percentage)*len(dataset['data']))], 'target': dataset['target'][:int((1-test_percentage)*len(dataset['target']))] }for dataset in datasets]
    test_datasets = [{'id': dataset['id'], 'data': dataset['data'][int((1-test_percentage)*len(dataset['data'])):], 'target': dataset['target'][int((1-test_percentage)*len(dataset['target'])):] }for dataset in datasets]
    
    return train_datasets, test_datasets
    
def load_meta_data_loader( datasets, batch_size=16, shuffle=True, num_workers=0):
    
    rng = default_rng()

    meta_dataset = []
    
    #length of each dataset
    dataset_lengths = [ len(dataset['data']) // batch_size for dataset in datasets]
    shuffled_datasets_indices = [np.arange(len(dataset['data'])) for dataset in datasets]
    
    for i, dataset in enumerate(datasets):
        rng.shuffle(shuffled_datasets_indices[i])
        shuffled_data = dataset['data'][shuffled_datasets_indices[i]]
        shuffled_targets = dataset['target'][shuffled_datasets_indices[i]]
        
        num_batches = len(shuffled_data) // batch_size
        
        shuffled_data = shuffled_data[:num_batches * batch_size].reshape(num_batches, batch_size, len(shuffled_data[0]))
        shuffled_targets = shuffled_targets[:num_batches * batch_size].reshape(num_batches, batch_size)
        
        batched_dataset = [{'x': data, 'y':target } for data, target in zip(shuffled_data, shuffled_targets)]
                
        meta_dataset = batched_dataset if i==0 else np.append(meta_dataset,batched_dataset, axis=0)

    
    rng.shuffle(meta_dataset, axis=0)
    
    return meta_dataset

def meta_dataset_loader3(datasets, batch_size=512, shuffle=True):
    
    support_meta_dataset = []
    query_meta_dataset = []
    rng = default_rng()
    
    for dataset in datasets:
        
        dataset_length = len(dataset['data'])
        
        dataset_indices = np.arange(dataset_length)
        rng.shuffle(dataset_indices) if shuffle else None
        
        data = dataset['data'][dataset_indices]
        targets = dataset['target'][dataset_indices]
        
        for i in range(0, dataset_length, batch_size):
            if (dataset_length-i) > batch_size:
                s_y = np.unique(targets[i:i+batch_size])
                q_y = np.unique(targets[i+batch_size: i +2*batch_size])
                if np.all(np.isin(s_y, q_y)):
                    support_meta_dataset.append({'x': data[i:i+batch_size], 'y': targets[i:i+batch_size],  'id' : dataset['id']})
                    query_meta_dataset.append({'x': data[i+batch_size: i +2*batch_size], 'y': targets[i+batch_size: i +2*batch_size], 'id' : dataset['id']})
                else:
                    print('skipped this batch')
    meta_dataset_indices = np.arange(len(support_meta_dataset))
    rng.shuffle(meta_dataset_indices)
    return np.array(support_meta_dataset)[meta_dataset_indices], np.array(query_meta_dataset)[meta_dataset_indices]
        

def meta_dataset_loader2(datasets, support_batch_size=32, query_batch_size=16 ):
    # datasets = [{'data': np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20],[29,220],[29,220]]), 'target': np.array([0,1,1,1,1,0,0,1,1,0,0,1])}]

    rng = default_rng()
    
    meta_dataset_support = []
    meta_dataset_query = []

    support_percentage = support_batch_size / (support_batch_size + query_batch_size)

    for dataset in datasets:
        dataset_indices = np.arange(len(dataset['data']))
        rng.shuffle(dataset_indices)
        
        support_length = (support_percentage * len(dataset['data']))
        support_length = int(support_length - support_length % support_batch_size)
        
        support_indices = dataset_indices[:support_length]
        query_indices = dataset_indices[support_length:]
        
        support_data = dataset['data'][support_indices]
        support_targets = dataset['target'][support_indices]
        
        
        batched_support_data = [support_data[i:i+support_batch_size] for i in range(0, len(support_data), support_batch_size)]
        batched_support_target = [support_targets[i:i+support_batch_size] for i in range(0, len(support_targets), support_batch_size)]
        
        batched_dataset = [{'x': data, 'y':target } for data, target in zip(batched_support_data, batched_support_target)]
        
        meta_dataset_support += batched_dataset

        
        for batch in batched_support_target:
            targets = np.unique(batch)
            batch_indices = []
            for i, index in enumerate(query_indices):
                if dataset['target'][index] in targets:
                    batch_indices.append(index)
                    np.delete(query_indices, i)
                
                if len(batch_indices) == query_batch_size:
                    break
                
            meta_dataset_query.append({'x': dataset['data'][batch_indices], 'y': dataset['target'][batch_indices]})
            
    meta_dataset_support = np.array(meta_dataset_support)
    meta_dataset_query = np.array(meta_dataset_query)
    
    indices = np.random.permutation(len(meta_dataset_support))
    
    meta_dataset_support = meta_dataset_support[indices] 
    meta_dataset_query =  meta_dataset_query[indices]
 
    
    
    return meta_dataset_support, meta_dataset_query
    
                                            
    
    
def meta_dataset_loader(datasets,  num_samples_per_class = 2, one_batch=False, shuffle=True):
    # datasets = load_OHE_dataset([31])
    # datasets = [{'data': np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20],[29,220]]), 'target': np.array([0,1,1,1,0,0,1,1,0,0,1])}]
    
    rng = default_rng()
    
    meta_dataset = []
    
    for dataset in datasets:
        
        targets = np.unique(dataset['target'])
        num_classes = len(targets)
        
        target_indices = [np.where(dataset['target'] == target)[0] for target in targets]
        
        if shuffle:
            for i in range(num_classes):
                rng.shuffle(target_indices[i]) 
        
        target_lengths = [len(target) for target in target_indices]       
        max_target_length = max(target_lengths)
        target_indices = [np.append(np.tile(target,(max_target_length//target_lengths[i],1)), target[:max_target_length%target_lengths[i]]) for i, target in enumerate(target_indices)]
            
        shuffled_target_indices = target_indices
        
        batch_size = num_classes * num_samples_per_class
        
        dataset_length = 0 
        
        for i in range(len(shuffled_target_indices)): dataset_length += len(shuffled_target_indices[i])
        
        number_of_batches =  1 if one_batch else dataset_length // batch_size
                
        # batched_dataset = []
        
        for i in range(number_of_batches):
            
            data_query = []
            data_support = []
            target_query = []
            target_support = []
        
            for j in range(len(shuffled_target_indices)):
                
                batch_indices = shuffled_target_indices[j][i*num_samples_per_class:(i+1)*num_samples_per_class]
                
                x = dataset['data'][batch_indices[:num_samples_per_class//2][0]]
                                
                data_query.append(dataset['data'][batch_indices[:num_samples_per_class//2]][0])
                data_support.append(dataset['data'][batch_indices[num_samples_per_class//2:]][0])
                target_query.append(dataset['target'][batch_indices[:num_samples_per_class//2]][0])
                target_support.append(dataset['target'][batch_indices[num_samples_per_class//2:]][0])
                            
            permutation_query = np.random.permutation(len(data_query))
            permutation_support = np.random.permutation(len(data_support))
            
            data_query = np.array(data_query)
            data_support = np.array(data_support)
            target_query = np.array(target_query)
            target_support = np.array(target_support)
            
            if shuffle:
                data_query = data_query[permutation_query]
                data_support = data_support[permutation_support]
                target_query = target_query[permutation_query]
                target_support = target_support[permutation_support]
            
            meta_dataset.append({'x': np.append(data_query, data_support, axis=0 ), 'y': np.append(target_query, target_support, axis=0)})

    rng.shuffle(meta_dataset) if shuffle else None
    
    return meta_dataset
    

def main():
    auto_ml_dids_train = [   23,    28,    30,    44,    46,    60,   181,   182,   375,
         725,   728,   735,   737,   752,   761,   772,   803,   807,
         816,   819,   833,   847,   871,   923,  1049,  1050,  1056,
        1069,  1462,  1466,  1475,  1487,  1496,  1497,  1504,  1507,
        1528,  1529,  1530,  1535,  1538,  1541,  4538, 40498, 40646,
       40647, 40648, 40649, 40650, 40677, 40680, 40691, 40701, 40704,
       40900, 40982, 40983, 42193]
    
    config = [('relabel',1),('drop_features', 1),('shuffle_features', 2), ('exp_scaling', 1), ('log_scaling', 1) ]
    datasets = load_OHE_dataset(auto_ml_dids_train, one_hot_encode=False)
    generate_datasets(datasets)
    pass
    # support, query = meta_dataset_loader3(datasets)
    
    # print(len(support), len(query))
    # for i in range(len(support)):
    #     print(len(support[i]['x']), len(query[i]['x']))
   

if __name__ == "__main__":
    main()