from torchvision import transforms

def data_transform(size, mean, std):
    return transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])