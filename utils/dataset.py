from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class Dataset(Dataset):
    def __init__(self, image_size: int = 32, batch_size: int = 32):
        self.image_size = image_size
        self.batch_size = batch_size

    def __getitem__(self, dataset_name = "cifar10", train = True):

        if dataset_name == "cifar10":
            if train:
                train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size = (self.image_size, self.image_size)),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])

                train_dataset = datasets.CIFAR10(root='dataset', train = True, transform = train_transform, download = True)
                data_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)

            else:
                test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size = (self.image_size, self.image_size)),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])

                test_dataset = datasets.CIFAR10(root = 'dataset', train = False, transform = test_transform, download = True)
                data_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle = False)
            
        elif dataset_name == "mnist":
            if train:
                train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size = (self.image_size, self.image_size)),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

                train_dataset = datasets.MNIST(root = 'dataset', train = True, transform = train_transform, download = True)
                data_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)

            else: 
                test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size = (self.image_size, self.image_size)),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                
                test_dataset = datasets.MNIST(root = 'dataset', train = False, transform = test_transform, download = True)
                data_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle = False)

        else: 
            print("Wrong Dataset name! One is only allowed to choose 'MNIST' or 'CIFAR10' dataset! ")

        return data_loader