import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import read_file

config = read_file('mer_config.json')

def generate_losocv_dataset(dataset, subject, visualize=True):
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    emotions = ['anger', 'sadness', 'surprise', 'fear', 'happiness', 'disgust']

    for x, y in zip(dataset['X'], dataset['y']):
        if subject in x[0]:
            test_X.append(x)
            test_y.append(y)
        else:
            train_X.append(x)   
            train_y.append(y)

    if visualize:
        train_set = []
        test_set = []
        for i, emo in enumerate(emotions):
            train_set.append(train_y.count(i))
            test_set.append(test_y.count(i))

        df = pd.DataFrame({
            'train': train_set,
            'test': test_set
        }, index=emotions)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax = df.plot.bar(rot=0, ax=ax)
        for container in ax.containers:
            ax.bar_label(container)

        ax.set_title('Data Distribution')
        plt.show()

    return train_X, train_y, test_X, test_y

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def generate_dataloader(X, y, batch_size):
    X_data = []
    y_data = []

    g = torch.Generator()
    g.manual_seed(config['gen_seed'])

    for (of, ops, h, v), y in zip(X, y):
        of = cv2.imread(of, cv2.IMREAD_GRAYSCALE)
        ops = cv2.imread(ops, cv2.IMREAD_GRAYSCALE)
        h = cv2.imread(h, cv2.IMREAD_GRAYSCALE)
        v = cv2.imread(v, cv2.IMREAD_GRAYSCALE)
        img = np.array([ops, h, v])

        X_data.append(img)
        y_data.append(y)

    X_tensor = torch.Tensor(np.array(X_data))
    y_tensor = torch.Tensor(y_data).to(dtype=torch.long)

    tensor_dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(tensor_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=4,
                            worker_init_fn=seed_worker,
                            generator=g)

    return data_loader

def get_subjects(dataset):
    subjects = []
    for data in dataset['X']:
        _, __, subject, __ = data[0].split(' ')
        if not subject in subjects:
            subjects.append(subject)
    
    return subjects