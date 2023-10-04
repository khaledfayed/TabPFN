import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
import openml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from numpy.random import default_rng
import torch
import torch.nn as nn
import os

openml.config.set_cache_directory(os.path.abspath('openml'))
    
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Example linear layer with bias
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)  # Normal initialization with mean=0.0 and std=0.01
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        # x = self.sig(x)
        # x = self.fc2(x)
        return x

def generate_datasets_gaussian(datasets, device='cpu'):
    
    model = MyModel(1, 1).to(device)
    
    for i,dataset in enumerate(datasets):
        
        delattr(model, 'fc1')
        model.add_module('fc1', nn.Linear(dataset['data'].shape[1], np.random.randint(4, 101)))
        nn.init.normal_(model.fc1.weight, mean=0.0, std=0.01)
        model.fc1.to(device)
        
        X = torch.tensor(dataset['data'], dtype=torch.float32).to(device)
        
        output = model(X)
        
        datasets[i]['data'] = output.detach().cpu().numpy()

        
def augment_datasets(datasets, augmentation_config):
    
    for i, (augmentation, amount) in enumerate(augmentation_config):
        
            augmentation_function = augmentation_dict[augmentation]
            
            for i,dataset in enumerate(datasets):
                
                datasets[i]['data'] = augmentation_function(datasets[i]['data'])
                
                
def shuffle_dataset_features(transformed_data):
    _, num_cols = transformed_data.shape
    augmented_data = transformed_data.copy()
    shuffled_columns = np.random.permutation(num_cols)
    augmented_data = augmented_data[:, shuffled_columns]
    
    return augmented_data


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

def load_OHE_dataset(dids, one_hot_encode=True):
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
        
                
        categorical_columns = [col for col, is_categorical in zip(attribute_names, categorical_features) if is_categorical]
        
        preprocessor = ColumnTransformer(
            transformers=[('cat', encoder, categorical_columns)],
            remainder='passthrough'
        )
        
        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        
        transformed_data = pipeline.fit_transform(df)
        transformed_targets = label_encoder.fit_transform(y)
                    
        encoded_datasets.append({'data': transformed_data, 'target': transformed_targets, 'id': dataset.id})
    
    return encoded_datasets


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

        
        for i in range(0, dataset_length, 2*batch_size):
            print('i', i)
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
    

def main():
    auto_ml_dids_train = [   23,    28,    30,    44,    60,   181,   182,   375,
         725,   728,   735,   737,   752,   761,   772,   803,   807,
         816,   819,   833,   847,   871,   923,  1049,  1050,  1056,
        1069,  1462,  1466,  1475,  1487,  1496,  1497,  1504,  1507,
        1528,  1529,  1530,  1535,  1538,  1541,  4538, 40498, 40646,
       40647, 40648, 40649, 40650, 40677, 40680, 40691, 40701, 40704,
       40900, 40982, 40983, 42193]
    
    # config = [('relabel',1),('drop_features', 1),('shuffle_features', 2), ('exp_scaling', 1), ('log_scaling', 1) ]
    datasets = load_OHE_dataset(auto_ml_dids_train, one_hot_encode=False)
    x, y = meta_dataset_loader3(datasets)
    
    # augment_datasets(datasets, [('shuffle_features',1)])

    # support, query = meta_dataset_loader3(datasets)
    
    # print(len(support), len(query))
    # for i in range(len(support)):
    #     print(len(support[i]['x']), len(query[i]['x']))
   

if __name__ == "__main__":
    main()