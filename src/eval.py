import os
from os.path import join

import numpy as np
import yaml
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

from src.data.loader import LoadData
from src.model import Model
from src.utils import load_weight


class EvalModel:
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

        model = Model(self.config['model']["name"], 
                           self.config["data"]['num_class'], 
                           False).to(self.device)
        self.model = load_weight(model, self.config["weight"])
        
        # Data
        if not os.path.exists(self.config["save"]["path"]):
            os.makedirs(self.config["save"]["path"])

        data = LoadData(self.data['batch'], 
                        self.data['size'], 
                        self.data['mean'],
                        self.data['std'])
        self.test_data = data.load_data(self.data["test"], "val")

    def val(self):
        df = pd.DataFrame(columns=['fname', 'labels', 'preds'])
        fnames, labels, preds = list(), list(), list()
        time_infer = list()

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
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

                if self.config['model']["classifier"] == "softmax":
                    probs = torch.softmax(outputs.data, 1)
                    _, predicts = torch.max(probs.data, 1)
                elif self.config['model']["classifier"] == "sigmoid":
                    probs = torch.sigmoid(outputs)
                    predicts = probs.round()
                
                accuracy = accuracy + (predicts == targets).sum().item()

                # Write to data frame
                fnames.append(path)
                labels.append(targets.data.cpu().numpy())
                preds.append(predicts.data.cpu().numpy())
            
            progress_bar.set_postfix(acc=f'{accuracy / step:.4f}')
        
        # Save data frame
        df['fname'] = np.array([subname.split('\\')[-1] for fname in fnames for subname in fname])
        df['labels'] = np.array([sublabel for label in labels for sublabel in label])
        df['preds'] = np.array([submpred for pred in preds for submpred in pred])

        df.sort_values(by=['fname'], inplace=True)

        save_path = join(self.config["save"]["path"], self.config["model"]["name"] + ".csv")
        df.to_csv(save_path, index=False)

        print("Results saved at:", save_path)
        print("Time inference:", np.mean(np.array(time_infer)), "(ms)")
        print(f"Number of paremeters: {sum(p.numel() for p in self.model.parameters()):,}")
        accuracy, precicsion, reccall, f1, cm = self.eval(df)
        
        # Metrics
        print('*' * 20 + "Evaluate model" + '*' * 20)
        print("Accuracy:", accuracy)
        print("Precision:", precicsion)
        print("Recall:", reccall)
        print("F1:", f1)

        # Confusion matrix
        print()
        cm_df = pd.DataFrame(cm, index=["(Label) Non-occluded", " " * 8 + "Occluded"], columns=["(Predict) Non-occluded", "Occluded"])
        print(cm_df)

        
    def eval(self, results):
        return (
            accuracy_score(results["labels"], results["preds"]),
            precision_score(results["labels"], results["preds"]),
            recall_score(results["labels"], results["preds"]),
            f1_score(results["labels"], results["preds"]),
            confusion_matrix(results["labels"], results["preds"], labels=[0, 1])
        )
