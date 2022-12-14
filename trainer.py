import copy
from dataclasses import dataclass
import os

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler, SGD, RMSprop
from tqdm import tqdm


@dataclass
class RunConfig:  # default parameters from the paper and official implementation
    learning_rate: float = 0.01
    num_epochs: int = 200
    weight_decay: float = 5e-4
    num_warmup_steps: int = 0
    save_each_epoch: bool = False
    output_dir: str = "."


class Trainer:
    def __init__(self, model, opt, momentum):
        self.model = model
        self.opt = opt
        self.momentum = momentum

    def train(self, features, train_labels, val_labels, test_labels, additional_matrix, device, run_config, log=True):
        self.model = self.model.to(device)
        features = features.to(device)
        train_labels = train_labels.to(device)
        test_labels = test_labels.to(device)
        additional_matrix = additional_matrix.to(device)  # adjacency or laplacian matrix depending on the model
        if self.opt == "Adam":
          print(self.opt)
          optimizer = Adam(self.model.parameters(), lr=run_config.learning_rate, weight_decay=run_config.weight_decay)
          # optimizer = Adam(self.model.parameters(), lr=run_config.learning_rate, weight_decay=run_config.weight_decay)
        if self.opt == "RMSProp":
          optimizer = RMSprop(self.model.parameters(), lr=run_config.learning_rate, weight_decay=run_config.weight_decay)
        else:
          optimizer = SGD(self.model.parameters(), lr=run_config.learning_rate, weight_decay=run_config.weight_decay, momentum=self.momentum)
        # https://huggingface.co/transformers/_modules/transformers/optimization.html#get_linear_schedule_with_warmup
        def lr_lambda(current_step: int):
            if current_step < run_config.num_warmup_steps:
                return float(current_step) / float(max(1, run_config.num_warmup_steps))
            return max(0.0, float(run_config.num_epochs - current_step) /
                       float(max(1, run_config.num_epochs - run_config.num_warmup_steps)))

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if log:
            print("Training started:")
            print(f"\tNum Epochs = {run_config.num_epochs}")

        best_loss, best_model_accuracy = float("inf"), 0
        best_model_state_dict = None
        train_accuracy_dict, test_accuracy_dict, embed_dict = {}, {}, {}

        train_iterator = tqdm(range(0, int(run_config.num_epochs)), desc="Epoch")
        for epoch in train_iterator:
            self.model.train()
            outputs = self.model(features, additional_matrix, train_labels)
            loss = outputs[1]
            embed_dict[epoch] = outputs[2]

            self.model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            val_loss, val_accuracy = self.evaluate(features, val_labels, additional_matrix, device)

            train_iterator.set_description(f"Training loss = {loss.item():.4f}, "
                                           f"val loss = {val_loss:.4f}, val accuracy = {val_accuracy:.2f}")
            
            train_loss, train_accuracy = self.evaluate(features, train_labels, additional_matrix, device)
            test_loss, test_accuracy = self.evaluate(features, test_labels, additional_matrix, device)

            train_accuracy_dict[epoch] = train_accuracy
            test_accuracy_dict[epoch] = test_accuracy

            save_best_model = val_loss < best_loss
            if save_best_model:
                best_loss = val_loss
                best_model_accuracy = val_accuracy
                best_model_state_dict = copy.deepcopy(self.model.state_dict())
            if save_best_model or run_config.save_each_epoch or epoch + 1 == run_config.num_epochs:
                output_dir = os.path.join(run_config.output_dir, f"Epoch_{epoch + 1}")
                self.save(output_dir)
        if log:
            print(f"Best model val CE loss = {best_loss:.4f}, best model val accuracy = {best_model_accuracy:.2f}")
        # reloads the best model state dict, bit hacky :P
        self.model.load_state_dict(best_model_state_dict)
        return embed_dict, train_accuracy_dict, test_accuracy_dict

    def evaluate(self, features, test_labels, additional_matrix, device):
        features = features.to(device)
        test_labels = test_labels.to(device)
        additional_matrix = additional_matrix.to(device)

        self.model.eval()

        outputs = self.model(features, additional_matrix, test_labels)
        ce_loss = outputs[1].item()

        ignore_label = nn.CrossEntropyLoss().ignore_index
        predicted_label = torch.max(outputs[0], dim=1).indices[test_labels != ignore_label]
        true_label = test_labels[test_labels != -100]
        accuracy = torch.mean((true_label == predicted_label).type(torch.FloatTensor)).item()

        return ce_loss, accuracy

    def save(self, output_dir):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        model_path = os.path.join(output_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)
