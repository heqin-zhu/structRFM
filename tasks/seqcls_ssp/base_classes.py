import os
import os.path as osp
import shutil
import abc

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score)

import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import Stack


class BaseCollator(object):
    def __init__(self):
        self.stack_fn = Stack()

    def __call__(self, raw_data_b):
        raise NotImplementedError("Must implement __call__ method.")


class BaseTrainer(object):
    def __init__(self,
                 args,
                 tokenizer,
                 model,
                 pretrained_model=None,
                 indicator=None,
                 ensemble=None,
                 train_dataset=None,
                 eval_dataset=None,
                 data_collator=None,
                 loss_fn=None,
                 optimizer=None,
                 compute_metrics=None,
                 visual_writer=None):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.pretrained_model = pretrained_model
        self.indicator = indicator
        self.ensemble = ensemble
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.compute_metrics = compute_metrics
        # default name_pbar is the first metric
        self.name_pbar = self.compute_metrics.metrics[0]
        self.visual_writer = visual_writer
        self.max_metric = 0.
        # init dataloaders
        self._prepare_dataloaders()

    def _get_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def _prepare_dataloaders(self):
        if self.train_dataset:
            self.train_dataloader = self._get_dataloader(self.train_dataset)

        if self.eval_dataset:
            self.eval_dataloader = self._get_dataloader(self.eval_dataset)

    def save_model(self, metrics_dataset, epoch):
        """
        Save model after epoch training in save_dir.
        Args:
            metrics_dataset: metrics of dataset
            epoch: training epoch number

        Returns:
            None
        """
        cur_metric = metrics_dataset[self.name_pbar]
        if cur_metric > self.max_metric:
            self.max_metric = metrics_dataset[self.name_pbar]
            save_model_path = osp.join(self.args.output_dir, f"model_ep{epoch}_{cur_metric:.4f}_{self.max_metric:.4f}.pt")
            data = {'model': self.model.state_dict()}
            if getattr(self, 'pretrained_model'):
                data['pretrained_model'] = self.pretrained_model.state_dict()
            torch.save(data, save_model_path)
            print("Model saved at:", save_model_path)

    def train(self, epoch):
        raise NotImplementedError("Must implement train method.")

    def eval(self, epoch):
        raise NotImplementedError("Must implement eval method.")


class BaseMetrics(abc.ABC):
    """Base class for functional tasks metrics
    """

    def __init__(self, metrics):
        """
        Args:
            metrics: names in list
        """
        self.metrics = [x.lower() for x in metrics]

    @abc.abstractmethod
    def __call__(self, outputs, labels):
        """
        Args:
            kwargs: required args of model (dict)

        Returns:
            metrics in dict
        """
        preds = torch.argmax(outputs, axis=-1)
        preds = preds.cpu().numpy().astype('int32')
        labels = labels.cpu().numpy().astype('int32')

        res = {}
        for name in self.metrics:
            func = getattr(self, name)
            if func:
                if func == self.auc:
                    # given two neural outputs, calculate their logits
                    # and then calculate auc
                    logits = torch.sigmoid(outputs).cpu().numpy()
                    m = func(logits, labels)
                else:
                    m = func(preds, labels)
                res[name] = m
            else:
                raise NotImplementedError
        return res

    @staticmethod
    def accuracy(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            accuracy
        """
        return accuracy_score(labels, preds)

    @staticmethod
    def precision(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        return precision_score(labels, preds, average='macro')

    @staticmethod
    def recall(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        return recall_score(labels, preds, average='macro')

    @staticmethod
    def f1s(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        return f1_score(labels, preds, average='macro')

    @staticmethod
    def mcc(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        return matthews_corrcoef(labels, preds)

    @staticmethod
    def auc(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        labels += 1
        preds = preds[:, 1]
        return roc_auc_score(labels, preds)


class MlpProjector(nn.Module):
    """MLP projection head.
    """

    def __init__(self, n_in, output_size=128):
        """
        Args:
            n_in:
            output_size:
        """
        super(MlpProjector, self).__init__()
        self.dense = nn.Linear(n_in, output_size)
        self.activation = nn.ReLU()
        self.projection = nn.Linear(output_size, output_size)
        self.n_out = output_size

    def get_n_out(self):
        return self.n_out

    def forward(self, embeddings):
        """
        Args:
            embeddings:

        Returns:

        """
        x = self.dense(embeddings)
        x = self.activation(x)
        x = self.projection(x)

        return x
