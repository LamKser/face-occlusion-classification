import yaml
from os.path import join
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from data_loader import LoadData
from model import Model
from utils import load_weight


class Test:
    def __init__(self, config):
        self.config = yaml.load(config, Loader=yaml.SafeLoader)
        self.data = self.config["data"]

        # model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Model(self.config['model'], 
                           self.config["train"]['num_class'], 
                           self.config["train"]["pretrain"]).to(self.device)
        self.model = load_weight(model, self.config["weight"])

        # save path
        self.save_results = join(self.config["path"], self.config["model"])
        if not os.path.exists(self.save_results):
            os.makedirs(self.save_results)

        # test data
        data = LoadData(self.data['batch_size'], 
                        self.data['input_size'], 
                        self.data['mean'],
                        self.data['std'])

        self.test_data = data.test_loader(self.data["test"])

    def test(self):
        df = pd.DataFrame(columns=['fname', 'labels', 'preds'])
        fnames, labels, preds = list(), list(), list()

        with torch.set_grad_enabled(False):
            self.model.eval()
            accuracy = 0
            progress_bar = tqdm(enumerate(self.test_set, start = 1),
                                total=len(self.test_set),
                                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                                desc='Testing'
                            )
            
            # test
            for step, (path, imgs, targets) in progress_bar:
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                outputs = self.model(imgs)

                probs = torch.softmax(outputs.data, 1)
                _, preds_ = torch.max(probs.data, 1)

                accuracy = accuracy + (preds_ == targets).sum().item()

                # Write to data frame
                fnames.append(path)
                labels.append(targets.data.cpu().numpy())
                preds.append(preds_.data.cpu().numpy())
            
            progress_bar.set_postfix(acc=f'{accuracy / step:.4f}')
        
        # Save data frame
        df['fname'] = np.array([subname.split('\\')[-1] for fname in fnames for subname in fname])
        df['labels'] = np.array([sublabel for label in labels for sublabel in label])
        df['preds'] = np.array([submpred for pred in preds for submpred in pred])

        df.sort_values(by=['fname'], inplace=True)

        save_path = join(self.save_results, self.config["model"])
        df.to_csv(save_path, index=False)

        print("Results saved at:", save_path)
        

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--opt", type=str, help='Enter test config yaml')
    args = parser.parse_args()

    train = Test(args.opt).test()