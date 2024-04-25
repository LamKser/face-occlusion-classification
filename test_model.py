import yaml
from os.path import join
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import (accuracy_score, 
                             f1_score, 
                             precision_score, 
                             recall_score)

from data_loader import LoadData
from model import Model
from utils import load_weight

def eval(results):
    print('*' * 20 + "Evaluate model" + '*' * 20)
    print("Accuracy:", accuracy_score(results["labels"], results["preds"]))
    print("Precision:", precision_score(results["labels"], results["preds"]))
    print("Recall:", recall_score(results["labels"], results["preds"]))
    print("F1:", f1_score(results["labels"], results["preds"]))


class Test:
    def __init__(self, config):
        fyml = open(config, 'r')
        self.config = yaml.load(fyml, Loader=yaml.SafeLoader)
        self.data = self.config["data"]

        # model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Model(self.config['model'], 
                           self.data['num_class'], 
                           False).to(self.device)
        self.model = load_weight(model, self.config["weight"])
        # print(self.model)
        # save path
        self.save_results = join(self.config["save"]["path"], self.config["model"])
        if not os.path.exists(self.save_results):
            os.makedirs(self.save_results)

        # test data
        data = LoadData(self.data['batch_size'], 
                        self.data['size'], 
                        self.data['mean'],
                        self.data['std'])

        self.test_data = data.test_loader(self.data["test"])

    def test(self):
        df = pd.DataFrame(columns=['fname', 'labels', 'preds'])
        fnames, labels, preds = list(), list(), list()
        time_infer = list()

        # Warm-up model
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(self.device)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(10):
          _ = self.model(dummy_input)

        with torch.set_grad_enabled(False):
            self.model.eval()
            accuracy = 0
            progress_bar = tqdm(enumerate(self.test_data, start = 1),
                                total=len(self.test_data),
                                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                                desc='Testing'
                            )
            
            # test
            for step, (path, imgs, targets) in progress_bar:
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                
                # Measure time inference
                starter.record()
                outputs = self.model(imgs)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize(device=self.device)
                curr_time = starter.elapsed_time(ender)
                time_infer.append(curr_time)

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

        save_path = join(self.save_results, self.config["model"] + ".csv")
        df.to_csv(save_path, index=False)
        print("Results saved at:", save_path)
        print("Time inference:", np.mean(np.array(time_infer)), "(ms)")
        print(f"Number of paremeters: {sum(p.numel() for p in self.model.parameters()):,}")
        eval(df)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--opt", type=str, help='Enter test config yaml')
    args = parser.parse_args()

    train = Test(args.opt).test()