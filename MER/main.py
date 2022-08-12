import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from model import MER
from data_util import generate_losocv_dataset, generate_dataloader, get_subjects
from utils import read_file

torch.manual_seed(0)

config = read_file('mer_config.json')
dataset = read_file(f'../.data/dataset_{config["img_dim"]}.json')
print(config)


def compute_f1_recall(real, pred):
    TN, FP, FN, TP = confusion_matrix(real, pred).ravel()
    if 2 * TP + FP + FN == 0:
        f1_score = 0
    else:
        f1_score = 2 * TP / (2 * TP + FP + FN)

    if real.count(1) == 0:
        recall = 0
    else:
        recall = TP / real.count(1)
    
    return f1_score, recall


def evaluation_metrics(real, pred, emotions):
    f1_list = []
    recall_list = []
    for i, emotion in enumerate(emotions):
        y_real = [1 if i == j else 0 for j in real]
        y_pred = [1 if i == j else 0 for j in pred]

        try:
            f1, recall = compute_f1_recall(y_real, y_pred)
            f1_list.append(f1)
            recall_list.append(recall)
        except Exception:
            pass
    
    uf1 = np.mean(f1_list)
    uar = np.mean(recall_list)

    return uf1, uar


def train_locosv():
    print('== Training LOSOCV ==')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emotions = ['anger', 'sadness', 'surprise', 'fear', 'happiness', 'disgust']

    actual = []
    predicted = []
    uf1_history = []
    uar_history = []
    accuracy_history = []
    num_correct = 0
    total_sample = 0

    f = open('.logs/_logs.txt', 'a')
    cm_df = None

    subjects = get_subjects(dataset)
    # subjects = [subject for subject in subjects if 'c_' in subject]

    for i, subject in enumerate(subjects):
        f.write(f'Subject ({i+1}/{len(subjects)}): {subject}\n')
        print(f'Subject ({i+1}/{len(subjects)}): {subject}')

        train_X, train_y, test_X, test_y = generate_losocv_dataset(dataset, subject, visualize=False)
        train_loader = generate_dataloader(train_X, train_y, batch_size=config['batch_size'])
        test_loader = generate_dataloader(test_X, test_y, batch_size=config['batch_size'])

        model = MER().to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.CrossEntropyLoss()

        epochs = config['epoch']

        # TRAINING
        model.train()
        for epoch in range(epochs):
            for batch, y in train_loader:
                batch = batch.to(device)
                y = y.to(device)

                ops = batch[:, :1, :, :]
                h = batch[:, 1:2, :, :]
                v = batch[:, 2:, :, :]

                optimizer.zero_grad()
                y_hat = model.forward((ops, h, v))
                loss = criterion(y_hat, y)

                loss.backward()
                optimizer.step()

        # TESTING
        model.eval()
        with torch.no_grad():
            y_preds = torch.Tensor().to(device)
            y_test = torch.Tensor().to(device)

            for batch, y in test_loader:
                batch = batch.to(device)
                y = y.to(device)

                ops = batch[:, :1, :, :]
                h = batch[:, 1:2, :, :]
                v = batch[:, 2:, :, :]

                y_hat = model.forward((ops, h, v))
                y_hat = torch.argmax(y_hat, dim=1)
                y_preds = torch.cat((y_preds, y_hat), dim=0)
                y_test = torch.cat((y_test, y), dim=0)
            
            y_test = y_test.to('cpu')
            y_preds = y_preds.to('cpu')
            f.write(f'Ground Truth:\t{[int(i) for i in y_test.tolist()]}\n')
            f.write(f'Predicted:\t{[int(i) for i in y_preds.tolist()]}\n')
            print(f'Ground Truth:\t{[int(i) for i in y_test.tolist()]}')
            print(f'Predicted:\t{[int(i) for i in y_preds.tolist()]}')

            cm = confusion_matrix(y_test, y_preds, labels=range(6))
            if type(cm_df) != pd.DataFrame:
                cm_df = pd.DataFrame(cm, index=emotions, columns=emotions)
            else:
                cm_df = cm_df.add(pd.DataFrame(cm, index=emotions, columns=emotions))
            
            num_correct += (y_preds == y_test).sum().item()
            total_sample += len(y_test)
            accuracy = num_correct / total_sample
            accuracy_history.append(accuracy)

            actual.extend(y_test.tolist())
            predicted.extend(y_preds.tolist())

            uf1, uar = evaluation_metrics(actual, predicted, emotions)
            uf1_history.append(uf1)
            uar_history.append(uar)

            f.write(f'[OVERALL] Accuracy: {accuracy:.4f} | UF1 score: {uf1:.4f} | UAR score: {uar:.4f}\n')
            f.write('=========================================================\n')
            print(f'[OVERALL] Accuracy: {accuracy:.4f} | UF1 score: {uf1:.4f} | UAR score: {uar:.4f}')
            print('=========================================================')


    uf1, uar = evaluation_metrics(actual, predicted, emotions) 

    f.write(f'Final Accuracy: {accuracy_history[-1]:.4f}\n')
    f.write(f'Final UF1 score: {uf1:.4f}\n')
    f.write(f'Final UAR score: {uar:.4f}\n')
    print(f'Final Accuracy: {accuracy_history[-1]:.4f}')
    print(f'Final UF1 score: {uf1:.4f}')
    print(f'Final UAR score: {uar:.4f}')
    f.close()

    cm_df.to_csv('_cm_df.csv')
    plt.plot(uf1_history, label='UF1')
    plt.plot(uar_history, label='UAR')
    plt.title('FULL')
    plt.legend()
    plt.savefig('_uf1_uar.png')
    plt.close('all')

def initialize_model():
    model = MER()
    num_of_parameters = sum(map(torch.numel, model.parameters()))
    print(f'Number of parameters: {num_of_parameters/1000000} million')

def main():
    initialize_model()
    train_locosv()

if __name__ == '__main__':
    main()