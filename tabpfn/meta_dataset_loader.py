import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, fetch_openml
from sklearn.model_selection import train_test_split

from numpy.random import default_rng

def split_datasets(test_percentage=0.2):
    breast_cancer = load_breast_cancer()
    iris = load_iris()
    
    datasets = [breast_cancer, iris]
    
    train_datasets = [{'data': dataset['data'][:int((1-test_percentage)*len(dataset['data']))], 'target': dataset['target'][:int((1-test_percentage)*len(dataset['target']))] }for dataset in datasets]
    test_datasets = [{'data': dataset['data'][int((1-test_percentage)*len(dataset['data'])):], 'target': dataset['target'][int((1-test_percentage)*len(dataset['target'])):] }for dataset in datasets]
    
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

    
    ################################################################################one dataset##############################################
            
    # for dataset_name in dataset_names:
    #     data = fetch_openml(name=dataset_name, as_frame=False)
    #     features = data['data']
    #     labels = data['target']
    #     print(labels)
    #     dataset = TensorDataset(features, labels)
    #     datasets.append(dataset)
        
    # meta_dataset = MetaDataset(datasets)
    
    # num_datasets = len(datasets)

    # meta_data_loader = DataLoader(
    # dataset=meta_dataset,
    # batch_size=batch_size,
    # shuffle=shuffle,
    # num_workers=num_workers)
    
    # return meta_data_loader
    
    

def main():
    split_datasets()

if __name__ == "__main__":
    main()