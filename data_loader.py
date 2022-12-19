from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import glob
from PIL import Image

class CustomImageFolder(ImageFolder):
    def __init__(self, root: str, transform):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return path, sample, target

class InstanceReader(Dataset):
    def __init__(self, image_path, transform=None):
        self.transform = transform
        self.isdir= Path(image_path).is_dir()
        
        if self.isdir:
            self.files = glob.glob(image_path + '/*')
        else:
            self.files=[image_path]

    def __getitem__(self, index):
        im_file = self.files[index]
        img = self.im_reader(im_file)
        return img, im_file

    def im_reader(self,im_file):
        im = Image.open(im_file)
        im = im.convert('RGB')
        return self.transform(im)

    def __len__(self):
        return len(self.files)

class LoadData:
    def __init__(self, train_path=None, val_path=None, test_path=None, batch_size=10, input_size=[224, 224]):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.input_size = input_size

    def train_loader(self):
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image_folder = ImageFolder(self.train_path, transform)
        train_data = DataLoader(
            image_folder, batch_size=self.batch_size, shuffle=True)

        return train_data

    def val_loader(self):
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image_folder = ImageFolder(self.val_path, transform)
        val_data = DataLoader(
            image_folder, batch_size=self.batch_size, shuffle=True)

        return val_data

    def test_loader(self):
        '''
        Load test folder for evaluating model
        '''
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image_folder = CustomImageFolder(self.test_path, transform)
        test_data = DataLoader(
            image_folder, batch_size=self.batch_size, shuffle=False)

        return test_data

def get_instance_data_loader(instance_directory,batch_size=1):
    """
    INSTANCE_DIRECTORY: directoy of images, or path of a single image
    RETURNS: DataLoader for iteration over the given directory or path
    """
    # print(instance_directory)
    data_loader = DataLoader(
        InstanceReader(instance_directory, transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])),
        batch_size=batch_size, shuffle=False)

    return data_loader