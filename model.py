from torchvision import models
from torch import nn, optim
import torch.nn.functional as F
import torch
from tqdm import tqdm
import pandas as pd
from data_loader import LoadData, get_instance_data_loader
import os
from PIL import ImageFile
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Model(nn.Module):

    def __init__(self, name, num_class, pretrained=False):
        super(Model, self).__init__()
        
        # =================Resnet=======================
        # ResNet 18
        if name == 'resnet18':
            if pretrained:
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet18()

        # ResNet 34
        elif name == 'resnet34':
            if pretrained:
                self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet101()
                
        # ResNet 50
        if name == 'resnet50':
            if pretrained:
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet50()

        # ResNet 101
        elif name == 'resnet101':
            if pretrained:
                self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet101()
        # ResNet 152
        if name == 'resnet152':
            if pretrained:
                self.model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet152()

        
        # =================Vgg=======================
        elif name == 'vgg11':
            if pretrained:
                self.model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
            else:
                self.model = models.vgg11()
        elif name == 'vgg16':
            if pretrained:
                self.model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
            else:
                self.model = models.vgg11()

        # Change the number of class
        if 'resnet' in name:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_class)
        elif 'vgg' in name:
            in_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(in_features, num_class)

    def forward(self, x):
        return self.model(x)

class RunModel():

    def __init__(self, device, name,
                 train_path, val_path, test_path, batch_size,
                 lr, weight_decay, momentum,
                 step_size, gamma,
                 num_class=1, pretrained=False):
        self.device = device
        self.model = Model(name, num_class, pretrained).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=lr,
                                   weight_decay=weight_decay,
                                   momentum=momentum)

        self.critetion = nn.CrossEntropyLoss().to(self.device)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=step_size,
                                                    gamma=gamma)
        self.data = LoadData(train_path, val_path, test_path, batch_size)

        print('Model used:', name)
        print("Device use:", self.device)
        print("Done load dataset")

    def __save_model(self, save_path, weight_file, epoch):
        # Create path if not exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save({'state_dict': self.model.state_dict(),
                    'Epoch': epoch},
                    os.path.join(save_path, weight_file))

    def __train_one_epoch(self, epoch, epochs, train_data):
        with torch.set_grad_enabled(True):
            self.model.train()
            total_loss = 0
            total_acc = 0
            total = 0

            pbar = tqdm(enumerate(train_data),
                        total=len(train_data),
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            pbar.set_description(f'Epoch [{epoch}/{epochs}][{self.scheduler.get_last_lr()[0]}][Train]')

            for step, (images, targets) in pbar:
                self.optimizer.zero_grad()
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)

                loss = self.critetion(outputs, targets)
                loss.backward()

                _, predict = torch.max(outputs.data, 1)
                total_acc = total_acc + (predict == targets).sum().item()
                total_loss = total_loss + loss.item()
                total = total + images.size(0)
                self.optimizer.step()

                if step % 250:
                    pbar.set_postfix(acc=f'{total_acc/total:.4f}', loss=f'{total_loss/(step + 1):.4f}')

            ave_acc = total_acc / total
            ave_loss = total_loss / (step + 1)
            pbar.set_postfix(acc=f'{ave_acc:.4f}', loss=f'{ave_loss:.4f}')

        return ave_acc, ave_loss

    def __val(self, epoch, epochs, val_data):
        with torch.set_grad_enabled(False):
            self.model.eval()
            total_loss = 0
            total_acc = 0
            total = 0

            pbar = tqdm(enumerate(val_data),
                        total=len(val_data),
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            pbar.set_description(' ' * len(f'Epoch [{epoch}/{epochs}][{self.scheduler.get_last_lr()[0]}]') + '[Valid]')

            for step, (images, targets) in pbar:
                images, targets = images.to(
                    self.device), targets.to(self.device)
                outputs = self.model(images)

                loss = self.critetion(outputs, targets)

                _, predict = torch.max(outputs.data, 1)
                total_acc = total_acc + (predict == targets).sum().item()
                total_loss = total_loss + loss.item()
                total = total + images.size(0)

                if step % 200:
                    pbar.set_postfix(acc=f'{total_acc/total:.4f}', loss=f'{total_loss/(step + 1):.4f}')

            ave_acc = total_acc / total
            ave_loss = total_loss / (step + 1)
            pbar.set_postfix(acc=f'{ave_acc:.4f}', loss=f'{ave_loss:.4f}')

        return ave_acc, ave_loss

    def train(self, epochs, save_path, weight_file, logger_path, val=True, continue_train=False, is_scheduler=False):

        best_acc = 0.0
        train_data = self.data.train_loader()
        if val:
            val_data = self.data.val_loader()

        # Write loss and accuracy to log file
        writer = SummaryWriter(logger_path)

        # Load pretrained weight
        if continue_train:
            checkpoint = torch.load(os.path.join(save_path, weight_file))
            self.model.load_state_dict(checkpoint['state_dict'])

        for epoch in range(1, epochs+1):
            # Train
            __train_acc, __train_loss = self.__train_one_epoch(epoch, epochs, train_data)

            # Validation
            if val:
                __val_acc, __val_loss = self.__val(epoch, epochs, val_data)
                
                # Save best accuracy
                # if best_acc < __val_acc:
                #     best_acc = __val_acc
                #     self.__save_model(os.path.join(save_path, f'best_{epoch}_'), weight_file)

            # Save model
            if __train_acc > __val_acc and best_acc < __val_acc:
                best_acc = __val_acc
                self.__save_model(save_path, weight_file, epoch)

            # if is_scheduler:
            #     self.scheduler.step()

            # Write to log file
            if val:
                writer.add_scalars('Loss', {'train': __train_loss,
                                            'val': __val_loss},
                                            epoch)
                writer.add_scalars('Accuracy', {'train': __train_acc,
                                                'val': __val_acc},
                                                epoch)
            else:
                writer.add_scalars('Loss', {'train': __train_loss}, epoch)
                writer.add_scalars('Accuracy', {'train': __train_acc}, epoch)

            # Save last model weight
            self.__save_model(save_path, 'last_' + weight_file, epoch)

        writer.close()

    def test(self, file_csv, weight_file, device=None):
        if device:
            device = self.device

        df = pd.DataFrame(columns=['fname', 'ground_truth', 'predict'])
        df = pd.DataFrame(columns=['fname', 'ground_truth', 'predict', 'prob', f'time_{str(device)}'])
        test_data = self.data.test_loader()

        # Load state_dict file
        checkpoint = torch.load(weight_file)
        print('Best model at epoch', checkpoint['Epoch'])

        self.model.load_state_dict(checkpoint['state_dict'])

        paths = []
        ground_truths = []
        predicts = []
        times = []
        probs = []
        
        with torch.set_grad_enabled(False):
            self.model.eval()
            total_acc = 0
            total = 0

            pbar = tqdm(enumerate(test_data),
                        total=len(test_data),
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            pbar.set_description('Testing model')

            for step, (path, images, targets) in pbar:
                images, targets = images.to(self.device), targets.to(self.device)
                
                start = time.time()
                outputs = self.model(images)
                end = time.time()
                
                # _, predict = torch.max(outputs.data, 1)
                prob = torch.softmax(outputs.data, 1)
                percent, predict = torch.max(prob, 1)

                

                total_acc = total_acc + (predict == targets).sum().item()
                total = total + images.size(0)

                # Save to csv
                paths.append(path)
                predicts.append(predict.data.cpu().numpy())
                ground_truths.append(targets.data.cpu().numpy())
                probs.append(percent.data.cpu().numpy())
                times.append(time)

                if step % 200:
                    pbar.set_postfix(acc=f'{total_acc/total:.4f}')

            ave_acc = total_acc / total
            pbar.set_postfix(acc=f'{ave_acc:.4f}')

        # Save to csv
        paths = np.array([subpath.split('\\')[-1] for p in paths for subpath in p])
        predicts = np.array([subpredict for s in predicts for subpredict in s])
        ground_truths = np.array([subtruth for truth in ground_truths for subtruth in truth])
        probs = np.array([subprob for prob in probs for subprob in prob])

        df['fname'] = paths
        df['ground_truth'] = ground_truths
        df['predict'] = predicts
        df[f'time_{str(device)}'] = times
        df['prob'] = probs
        df.to_csv(file_csv, index=False)

        print(f'Saved results in {file_csv}')

'''
    def test_image(self, instance_directory, file_csv, weight_file, device):
        df = pd.DataFrame(columns=['fname', 'predict', 'Prob', f'time_{str(device)}'])
        test_data = get_instance_data_loader(instance_directory)
        # img, im_file
        # Load state_dict file
        checkpoint = torch.load(weight_file)
        self.model.load_state_dict(checkpoint['state_dict'])

        paths = []
        predicts = []
        times = []
        with torch.set_grad_enabled(False):
            self.model.eval()

            pbar = tqdm(enumerate(test_data),
                        total=len(test_data),
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            pbar.set_description('Testing model')

            for step, (images, path) in pbar:
                images = images.to(self.device)

                # Calculate time inference
                start = time.time()
                outputs = self.model(images)
                end = time.time()

                _, predict = torch.max(outputs.data, 1)

                # Save to csv
                paths.append(path)
                predicts.append(predict.data.cpu().numpy())
                times.append(end-start)


        paths = np.array([subpath.split('\\')[-1] for p in paths for subpath in p])
        predicts = np.array([subpredict for s in predicts for subpredict in s])

        df['fname'] = paths
        df['predict'] = predicts
        df[f'time_{str(device)}'] = times
        df.to_csv(file_csv, index=False)
        print(f'Saved results in {file_csv}')
'''

# if __name__ == '__main__':
#     from torchsummary import summary
#     tmp = Model('vgg11', 2)
#     checkpoint = torch.load('D:\\Unicloud\\liveness_detection\\face-occlusion-classification\\weights\\vgg11_224_224\\vgg11_3_224_224.pt')
#     tmp.load_state_dict(checkpoint['state_dict'])
#     # print(tmp)
#     print(summary(tmp.cuda(), (3, 224, 224)))