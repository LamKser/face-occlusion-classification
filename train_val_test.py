from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch import nn, optim
import torch

from PIL import Image
from tqdm import tqdm 
import pandas as pd
import numpy as np
import yaml
import os

from data_loader import LoadData
from model import Model


# Read configure file
config_yaml_file = open("config.yaml", 'r')
config = yaml.load(config_yaml_file, Loader=yaml.SafeLoader)


class Run():

    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model(config['model'], config["train"]['num_class'], config["train"]["pretrain"]).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=config["train"]['learning_rate'],
                                   weight_decay=config["train"]['weight_decay'],
                                   momentum=config["train"]['momentum']
        )
        self.critetion = nn.CrossEntropyLoss().to(self.device)
        self.data = LoadData(config["train"]['batch_size'], config["train"]['input_size'], 
                             config['mean'], config['std'])
        
        if not os.path.exists(config["save"]["path"]):
            os.makedirs(
                os.path.join(config["save"]["path"], config["model"])
            )

        self.wandb = config["wandb"]
        
        print(f'Using {self.device.type}')

    def save_weight(self, epoch: int, save_dir: str, weight_name: str):
        # Create folder weight if not exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save model weight
        torch.save({'state_dict': self.model.state_dict(),
                    'epoch': epoch},
                    os.path.join(save_dir, weight_name)
                    )
        
    def load_weight(self, save_dir: str, weight_name: str):
        checkpoints = torch.load(os.path.join(save_dir, weight_name))
        print('Model at epoch:', checkpoints['epoch'])
        self.model.load_state_dict(checkpoints['state_dict'])

    def train_one_epoch(self, epoch: str, train_data):

        # Train model in an epoch
        with torch.set_grad_enabled(True):
            self.model.train()
            total_loss = 0
            total_accuracy = 0
            total_data = len(train_data.dataset)

            progress_bar = tqdm(enumerate(train_data),
                                total=len(train_data),
                                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
                            )
            progress_bar.set_description(f'Epoch [{epoch}][Train]')

            # Training step
            for step, (images, targets) in progress_bar:
                self.optimizer.zero_grad()
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)

                step_loss = self.critetion(outputs, targets)
                step_loss.backward()
                self.optimizer.step()

                probability = torch.softmax(outputs.data, 1)
                _, predicts = torch.max(probability.data, 1)
                
                # Calculate loss and accuracy
                total_loss = total_loss + step_loss.item()
                total_accuracy = total_accuracy + (predicts == targets).sum().item()

                if step % 200:
                    progress_bar.set_postfix(acc=f'{total_accuracy/total_data:.4f}', loss=f'{total_loss/(step + 1):.4f}')
            progress_bar.set_postfix(acc=f'{total_accuracy/total_data:.4f}', loss=f'{total_loss/(step + 1):.4f}')

            train_accuracy = total_accuracy/total_data
            train_loss = total_loss/(step + 1)

        return train_accuracy, train_loss

    def validate(self, epoch: str, val_data):

        # Validate model in an epoch
        with torch.set_grad_enabled(False):
            self.model.eval()
            total_loss = 0
            total_accuracy = 0
            total_data = len(val_data.dataset)

            progress_bar = tqdm(enumerate(val_data),
                                total=len(val_data),
                                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
                            )
            progress_bar.set_description(' ' * len(f'Epoch [{epoch}]') + '[Val]')

            # Validation step
            for step, (images, targets) in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)

                step_loss = self.critetion(outputs, targets)

                probability = torch.softmax(outputs.data, 1)
                _, predicts = torch.max(probability.data, 1)
                
                # Calculate loss and accuracy
                total_loss = total_loss + step_loss.item()
                total_accuracy = total_accuracy + (predicts == targets).sum().item()

                if step % 200:
                    progress_bar.set_postfix(acc=f'{total_accuracy/total_data:.4f}', loss=f'{total_loss/(step + 1):.4f}')
            progress_bar.set_postfix(acc=f'{total_accuracy/total_data:.4f}', loss=f'{total_loss/(step + 1):.4f}')

            val_accuracy = total_accuracy/total_data
            val_loss = total_loss/(step + 1)
            

        return val_accuracy, val_loss

    def train(self):
        
        best_accuracy = 0
        train_set = self.data.train_loader(config["data"]["train"])
        val_set = self.data.val_loader(config["data"]["val"])

        # Create logger file (Tensorboard)
        writer = SummaryWriter(
            os.path.join(config["save"]["path"], config["save"]["logger"])
        )
        epochs = config["train"]['epochs']
        
        # Init wandb project
        if self.wandb["project"]:
            import wandb

            if self.wandb["resume_id"]:
                id = self.wandb["resume_id"]
                resume = 'allow'
            else:
                id = wandb.util.generate_id()
                resume = 'never'

            wb = wandb.init(
                project=self.wandb["project"],
                name=self.wandb["name"],
                resume=resume,
                id=id
            )
        
        # Training
        for epoch in range(1, epochs + 1):
            train_accuracy, train_loss = self.train_one_epoch(str(f'{epoch}/{epochs}'), train_set)
            val_accuracy, val_loss = self.validate(str(f'{epoch}/{epochs}'), val_set)

            # Wrtie to log file
            writer.add_scalars('Loss', {'train': train_loss,
                                        'val': val_loss},
                                        epoch)

            writer.add_scalars('Accuracy', {'train': train_accuracy,
                                            'val': val_accuracy},
                                             epoch)
            if self.wandb["project"]:
                wb.log({
                    "loss/train": train_loss,
                    "loss/val": val_loss,
                    "epoch": epoch
                })
                wb.log({
                    "accuracy/train": train_accuracy,
                    "accuracy/val": val_accuracy,
                    "epoch": epoch
                })

            # Save best model
            if val_accuracy > best_accuracy:
                self.save_weight(epoch, config["save"]["path"], 
                                 'best_' + config["save"]["weight"])
                best_accuracy = val_accuracy
        
            # Save last model
            self.save_weight(epoch, config["save"]["path"], 
                             'last_' + config["save"]["weight"])

        writer.close()

    def test(self):
        
        # Saving predict results to csv
        df_test = pd.DataFrame(columns=['fname', 'ground_truth', 'predict'])
        fnames, labels, models  = list(), list(), list() 

        test_set = self.data.test_loader(config["data"]["test"])

        # Load weight
        self.load_weight(config["save"]["path"], config["save"]["weight"])

        with torch.set_grad_enabled(False):
            self.model.eval()
            total_accuracy = 0
            total_data = len(test_set.dataset)

            progress_bar = tqdm(enumerate(test_set),
                                total=len(test_set),
                                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                                desc='Testing'
                            )
            # Test step
            for step, (path, images, targets) in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)

                probability = torch.softmax(outputs.data, 1)
                _, predicts = torch.max(probability.data, 1)
                
                # Calculate accuracy
                total_accuracy = total_accuracy + (predicts == targets).sum().item()

                # Save to list
                fnames.append(path)
                labels.append(targets.data.cpu().numpy())
                models.append(predicts.data.cpu().numpy())

                if step % 200:
                    progress_bar.set_postfix(acc=f'{total_accuracy/total_data:.4f}')
            progress_bar.set_postfix(acc=f'{total_accuracy/total_data:.4f}')
        
        df_test['fname'] = np.array([subname.split('\\')[-1] for fname in fnames for subname in fname])
        df_test['ground_truth'] = np.array([sublabel for label in labels for sublabel in label])
        df_test['predict'] = np.array([submodel for model in models for submodel in model])

        df_test.sort_values(by=['fname'], inplace=True)

        save_path = os.path.join(config["save"]["path"], config["save"]["csv"])
        df_test.to_csv(save_path, index=False)
        print(f'Saved csv at {save_path}')
        
    def test_image(self, image_path, save_dir, weight_name):

        labels = {0: 'Non-occluded', 1: 'Occluded'}
        self.load_weight(save_dir, weight_name)
        self.model.eval()
        
        transform = transforms.Compose([
            transforms.Resize(config['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std'])
        ])
        
        image = Image.open(image_path)
        image = transform(image).to(self.device)

        outputs = self.model(image.unsqueeze(0))
        outputs = torch.softmax(outputs, 1)
        probability, predict = torch.max(outputs, 1)
        
        print(labels[predict.item()], '-', '{:.2f} %'.format(probability.item() * 100))

