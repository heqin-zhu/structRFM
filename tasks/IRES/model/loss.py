import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import sys
import dgl
def BCEWithLogitslosss(pred, label):
      criterion = nn.BCEWithLogitsLoss()
      pred = pred.squeeze()
      loss = criterion(pred, label.float())
      return loss
def CrossEntropyloss(pred, label):
       criterion = nn.CrossEntropyLoss()
       pred = pred.squeeze()
       loss = criterion(pred, label.float())
       return loss
def BCEloss(pred, label):
    criterion=nn.BCELoss()
    pred = pred.squeeze()
    loss = criterion(pred, label.float())
    return loss
