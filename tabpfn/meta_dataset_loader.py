import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, fetch_openml
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import openml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

from numpy.random import default_rng

def shuffle_dataset_features(transformed_data):
    _, num_cols = transformed_data.shape
    augmented_data = transformed_data.copy()
    shuffled_columns = np.random.permutation(num_cols)
    augmented_data = augmented_data[:, shuffled_columns]
    
    return augmented_data

def load_OHE_dataset(dids, num_augmented_datasets=0):
    encoder = OneHotEncoder()
    label_encoder = LabelEncoder()

    datasets = openml.datasets.get_datasets(dids)
    
    encoded_datasets = []
    
    for dataset in datasets:
        X, y, categorical_features, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute,
        )
        
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
    datasets = load_OHE_dataset([31])
    data = meta_dataset_loader2(datasets)
   
    pass

if __name__ == "__main__":
    main()