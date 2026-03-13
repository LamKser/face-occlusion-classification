import yaml
from os.path import join
import os

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
import torch

from src.data.loader import LoadData
from src.model import Model
from src.utils import save_weight, resume_train


class TrainModel:
    def __init__(self, config):

        # Load config
        # with open(config_path, 'r') as f:
        #     self.config = yaml.load(f, Loader=yaml.SafeLoader)
        self.config = config
        self.save = self.config["save"]
        self.data = self.config["data"]

        # Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device:", self.device)

        if (self.config['model']["classifier"] == "sigmoid") \
            and (self.config["train"]['num_class'] != 1):
            self.config["train"]['num_class'] = 1

        self.model = Model(self.config['model']["name"], 
                           self.config["train"]['num_class'], 
                           self.config["train"]["pretrain"]).to(self.device)
        
        # Loss function
        if self.config['model']["classifier"] == "softmax":
            self.critetion = nn.CrossEntropyLoss().to(self.device)
        elif self.config['model']["classifier"] == "sigmoid":
            self.critetion = nn.BCEWithLogitsLoss().to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr = self.config["train"]['learning_rate'],
                                   weight_decay = self.config["train"]['weight_decay'],
                                   momentum = self.config["train"]['momentum']
        )
        # Create save dir
        self.save_dir = join(self.save["path"], self.config["model"]["name"])
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Data
        data = LoadData(self.config["train"]['batch_size'], 
                        self.data['size'], 
                        self.data['mean'], 
                        self.data['std'])

        self.train_data = data.load_data(self.data["train"], "train")
        self.val_data = data.load_data(self.data["val"], "val")

        # Init wandb & tensorboard
        # wandb
        wandb_ = self.config["wandb"]
        self.wandb_status = False
        if wandb_["project"]:
            import wandb
            wandb.login(
                relogin=True
            )
            self.wandb_status = True

            if wandb_["resume_id"]:
                id = wandb_["resume_id"]
                resume = 'allow'
            else:
                id = wandb.util.generate_id()
                resume = 'never'

            self.wb = wandb.init(
                project=wandb_["project"],
                name=wandb_["name"],
                resume=resume,
                id=id
            )

        # tensorboard
        self.writer = SummaryWriter(join(self.save["path"], self.save["logger"]))
        # print(f'Using {self.device.type}')

    
    def __train_one_epoch(self, epoch):
        with torch.set_grad_enabled(True):
            self.model.train()
            loss, accuracy, total_data = 0, 0, len(self.train_data.dataset)

            # Progress bar
            progress_bar = tqdm(enumerate(self.train_data, start = 1),
                                total=len(self.train_data),
                                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
                            )
            progress_bar.set_description(f'Epoch [{epoch}][Train]')

            # Trainig
            for step, (imgs, targets) in progress_bar:
                self.optimizer.zero_grad()
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                outputs = self.model(imgs)
                loss_ = self.critetion(outputs, targets)
                loss_.backward()
                self.optimizer.step()

                if self.config['model']["classifier"] == "softmax":
                    probs = torch.softmax(outputs.data, 1)
                    _, preds = torch.max(probs.data, 1)
                elif self.config['model']["classifier"] == "sigmoid":
                    probs = torch.sigmoid(outputs)
                    preds = probs.round()

                loss = loss + loss_.item()
                accuracy = accuracy + (preds == targets).sum().item()
                if step % 200:
                    progress_bar.set_postfix(acc=f'{accuracy / total_data:.4f}', loss=f'{loss / step:.4f}')
            progress_bar.set_postfix(acc=f'{accuracy / total_data :.4f}', loss=f'{loss / step:.4f}')

        return accuracy / total_data, loss / step

    def __validation(self, epoch):
        with torch.set_grad_enabled(False):
            self.model.eval()
            loss, accuracy, total_data = 0, 0, len(self.val_data.dataset)
            progress_bar = tqdm(enumerate(self.val_data, start = 1),
                                total=len(self.val_data),
                                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
                            )
            progress_bar.set_description(f'Epoch [{epoch}][ Val ]')

            # Validation
            for step, (imgs, targets) in progress_bar:
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                outputs = self.model(imgs)

                loss_ = self.critetion(outputs, targets)

                if self.config['model']["classifier"] == "softmax":
                    probs = torch.softmax(outputs.data, 1)
                    _, preds = torch.max(probs.data, 1)
                elif self.config['model']["classifier"] == "sigmoid":
                    probs = torch.sigmoid(outputs)
                    preds = probs.round()
                
                # Calculate loss and accuracy
                loss = loss + loss_.item()
                accuracy = accuracy + (preds == targets).sum().item()
                if step % 200:
                    progress_bar.set_postfix(acc=f'{accuracy / total_data:.4f}', loss=f'{loss / step:.4f}')
            progress_bar.set_postfix(acc=f'{accuracy / total_data :.4f}', loss=f'{loss / step:.4f}')

        return accuracy / total_data, loss / step
    
    def train(self):
        best_acc = 0
        epochs = self.config["train"]["epochs"]

        # Resume train
        if self.config["train"]["resume"]:
            self.model, start_epoch = resume_train(self.model, self.config["train"]["resume"])
            start_epoch += 1
            print(f"Resume training from epoch {start_epoch}")
        else:
            start_epoch = 1

        # Train
        for epoch in range(start_epoch, epochs + 1):
            train_acc, train_loss = self.__train_one_epoch(f"{epoch}/{epochs}")
            val_acc, val_loss = self.__validation(f"{epoch}/{epochs}")

            # Tensorboard
            self.writer.add_scalars(
                "loss", {
                    "train": train_loss,
                    "val": val_loss
                },
                epoch
            )
            self.writer.add_scalars(
                "accuracy", {
                    "train": train_acc,
                    "val": val_acc
                },
                epoch
            )

            # wandb
            if self.wandb_status:
                self.wb.log({
                    "loss/train": train_loss,
                    "loss/val": val_loss,
                    "epoch": epoch
                })
                self.wb.log({
                    "acc/train": train_acc,
                    "acc/val": val_acc,
                    "epoch": epoch
                })

            # Save model
            if val_acc > best_acc:
              print("Save best model at epoch", epoch)
              save_weight(self.model, epoch, self.save_dir, "best_" + self.save["weight"])
              best_acc = val_acc
        save_weight(self.model, epoch, self.save_dir, "last_" + self.save["weight"])
        self.writer.close()


# if __name__ == "__main__":
#     from argparse import ArgumentParser

#     parser = ArgumentParser()
#     parser.add_argument("--opt", type=str, help='Enter train config yaml')
#     args = parser.parse_args()
#     train = Train(args.opt).train()
    
