from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from src.data.process import data_transform


class CustomImageFolder(ImageFolder):
    # Return image path, data and target

    def __init__(self, root: str, transform):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        data = self.loader(path)
        if not (self.transform is None):
            data = self.transform(data)

        return path, data, target


class LoadData:

    def __init__(self, batch_size, size, mean, std):
        
        self.batch_size = batch_size
        self.transform = data_transform(size, mean, std)

    # def train_loader(self, train_path):
    #     # Load train set for evaluating model
    #     image_folder = ImageFolder(train_path, self.transform)
    #     train_data = DataLoader(image_folder, batch_size=self.batch_size, shuffle=True)
    #     return train_data

    # def val_loader(self, val_path):
    #     # Load validation set for evaluating model
    #     image_folder = ImageFolder(val_path, self.transform)
    #     val_data = DataLoader(image_folder, batch_size=self.batch_size, shuffle=False)
    #     return val_data

    # def test_loader(self, test_path):
    #     # Load test set for evaluating model
    #     image_folder = CustomImageFolder(test_path, self.transform)
    #     test_data = DataLoader(image_folder, batch_size=self.batch_size, shuffle=False)
    #     return test_data

    def load_data(self, path, mode):
        # Show logging
        print("Data path:", path)
        print("Data mode:", mode)
        if mode == "train":
            image_folder = ImageFolder(path, self.transform)
            data = DataLoader(image_folder, batch_size=self.batch_size, shuffle=True)
        elif mode == "val":
            image_folder = CustomImageFolder(path, self.transform)
            data = DataLoader(image_folder, batch_size=self.batch_size, shuffle=False)
        elif mode == "test":
            image_folder = CustomImageFolder(path, self.transform)
            data = DataLoader(image_folder, batch_size=self.batch_size, shuffle=False)
        return data