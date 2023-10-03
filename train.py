import torch
from utils.dataset import Dataset
from models.VGG16 import VGG16
from tqdm import tqdm

class VGGTrainer():
    def Cfg(self, epochs: int = 100, lr: float = 0.001):
        self.batch_size = 16
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VGG16().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.datasetname = "cifar10"


    def Train(self):
        train_dataloader = Dataset(image_size = 224).__getitem__(train = True, dataset_name = self.datasetname)
        for epoch in tqdm(range(self.epochs)):
            for batch_idx, (X, y) in enumerate(train_dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (batch_idx % 20) == 0:
                    print(f"Epoch: {epoch} | Loss: {loss.item()}")

        test_dataloader = Dataset(image_size = 224).__getitem__(train = False, dataset_name = self.datasetname)
        correct = 0
        total = 0
        for x_test, y_test in test_dataloader:
            x_test = x_test.to(self.device)
            y_test = y_test.to(self.device)
            outputs = self.model(x_test)
            _, predictions = torch.max(outputs, 1)
            total += y_test.shape[0]
            correct += (predictions == y_test).sum()

        accuracy = 100 * correct / float(total)
        print(f"Accuracy on the test images: {accuracy} ")


if __name__ == "__main__":
    vgg = VGGTrainer()
    vgg.Cfg()
    vgg.Train()