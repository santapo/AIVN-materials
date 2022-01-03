import torch
from torch.utils.data import Dataset

class KNearestNeighbor:
    def __init__(self):
        pass

    def train(self, train_data):
        if isinstance(train_data, Dataset):
            X_train = []
            y_train = []
            for i in range(len(train_data)):
                X, y = train_data[i]
                X_train.append(X)
                y_train.append(y)
            self.X_train = torch.stack(X_train, dim=0)
            self.y_train = torch.tensor(y_train, dtype=torch.int8)
        else:
            self.X_train, self.y_train = train_data

    def predict(self, X, k):
        num_test = X.shape[0]
        y_pred = torch.zeros(num_test, dtype=self.y_train.dtype)
        for i in range(num_test):
            similarity = torch.sum(torch.abs(self.X_train - X[i, :]), axis=1)
            distances, indices = similarity.topk(k, largest=False, sorted=True)
            retrieved_neighbors = torch.gather(self.y_train, 0, indices)
            y_pred[i] = torch.mode(retrieved_neighbors, 0)[0]
        return y_pred