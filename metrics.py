from __future__ import print_function
from __future__ import division

from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import os
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchsummary import summary
from data_reader_isic import get_data_loaders
from utils import *
from torch.backends import cudnn
from augment_data_isic import augment_images
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(self,model, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    target_all = []
    predicted_all = []
    print("\ntesting with previous accuracy {}".format(self.best_acc))
    for batch_idx, (inputs, targets) in enumerate(self.testloader):
        if self.use_cuda:
            inputs = inputs.cuda()
            targets = np.asarray(targets).astype(np.int64)
            targets = torch.from_numpy(targets).cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        loss = self.criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        predicted_batch = predicted.eq(targets.data).cpu()
        predicted_reshaped = predicted_batch.numpy().reshape(-1)
        predicted_all = np.concatenate((predicted_all, predicted_reshaped), axis=0)

        targets_reshaped = targets.data.cpu().numpy().reshape(-1)
        target_all = np.concatenate((target_all, targets_reshaped), axis=0)
        total += targets.size(0)
        correct += predicted_batch.sum()

        progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    self.writer.add_scalar('test loss', test_loss, epoch)
    # Save checkpoint.
    acc = 100. * correct / total
    writer.add_scalar('test accuracy', acc, epoch)
    if acc > self.best_acc:
        pass
    self.save_model(acc, epoch)
    self.best_acc = acc
    print("Accuracy:{}".format(acc))
    cm = metrics.confusion_matrix(target_all, predicted_all)
    print("Confsusion metrics: \n{}".format(cm))
