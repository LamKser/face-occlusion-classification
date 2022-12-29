from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader


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

    def __init__(self, batch_size, input_size, mean, std):
        
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def train_loader(self, train_path):
        # Load train set for evaluating model
        image_folder = ImageFolder(train_path, self.transform)
        train_data = DataLoader(image_folder, batch_size=self.batch_size, shuffle=True)
        return train_data

    def val_loader(self, val_path):
        # Load validation set for evaluating model
        image_folder = ImageFolder(val_path, self.transform)
        val_data = DataLoader(image_folder, batch_size=self.batch_size, shuffle=True)
        return val_data

    def test_loader(self, test_path):
        # Load test set for evaluating model
        image_folder = CustomImageFolder(test_path, self.transform)
        test_data = DataLoader(image_folder, batch_size=self.batch_size, shuffle=False)
        return test_data
