import torch
from utils.dataset import Dataset
from models.VGG import VGG
from tqdm import tqdm

class VGGTrainer():
    def Cfg(self, epochs: int = 100, lr: float = 0.001):
        self.batch_size = 16
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VGG().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.datasetname = "cifar10"


    def Train(self):
        train_dataloader = Dataset(image_size = 224).__getitem__(train = True, dataset_name = self.datasetname)
        test_dataloader = Dataset(image_size = 224).__getitem__(train=False, dataset_name=self.datasetname)

        loop = tqdm(range(self.epochs), leave=True)
        for epoch in loop:
            for batch_idx, (X, y) in enumerate(train_dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # if (batch_idx % 100) == 0:
                    # print(f"Epoch: {epoch + 1}/{self.epochs} | Step {batch_idx + 1}/{Dataset().__len__(train=True)} | Loss: {loss.item()}")
                loop.set_postfix(epoch = epoch + 1,  batch = f"{batch_idx + 1 } / {Dataset().__len__(train=True)}" , loss = loss.item(),)

            correct = 0
            total = 0
            with torch.inference_mode():
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