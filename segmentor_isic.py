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
from data_reader_isic import get_data_sets
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Classifier(object):
    def __init__(self,model_details):
        self.device_ids=[0,1]
        self.model_details=model_details
        self.log_dir=self.model_details.logs_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.writer = SummaryWriter(self.log_dir)

        self.use_cuda = torch.cuda.is_available()
        self.best_acc = 0  # best test accuracy
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.net = None
        self.criterion = None
        self.optimizer = None
        self.model_name_str = None

    def load_data(self):
        # Data
        print('==> Preparing data..')

        trainset, validationset = get_data_sets(self.model_details)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.model_details.batch_size, shuffle=True,
                                                  num_workers=2)
        self.testloader = torch.utils.data.DataLoader(validationset, batch_size=self.model_details.batch_size, shuffle=False, num_workers=2)

        train_count = len(self.trainloader) * self.model_details.batch_size
        test_count = len(self.testloader) * self.model_details.batch_size
        print('==> Total examples, train: {}, test:{}'.format(train_count, test_count))

    def load_model(self):
        model_details=self.model_details
        self.learning_rate=model_details.learning_rate
        model_name = model_details.model_name
        model_name_str = model_details.model_name_str
        print('\n==> using model {}'.format(model_name_str))
        self.model_name_str="{}".format(model_name_str)

        # Model
        try:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert (os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!')
            checkpoint = torch.load('./checkpoint/{}_ckpt.t7'.format(self.model_name_str ))
            model = checkpoint['net']
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
        except Exception as e:
            model = model_details.model
            print('==> Building model..')


        if self.use_cuda:
            model=model.cuda()
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True
        self.model=model

        summary(model, (3, 224, 224))
        self.criterion = nn.CrossEntropyLoss()

        if model_details.optimizer=="adam":
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate, eps=model_details.eps)

        self.writer.add_scalar("leanring rate", self.learning_rate)
        self.writer.add_scalar("eps", model_details.eps)

    def train(self, epoch):
        print('\n Training Epoch:{} '.format(epoch))
        model = self.model
        optimizer=self.optimizer
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            step = epoch * len(self.trainloader) + batch_idx
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
                inputs1, targets1 = inputs.cuda(1), targets.cuda(1)

            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            batch_loss=train_loss / (batch_idx + 1)
            if batch_idx%2==0:
                self.writer.add_scalar('step loss',batch_loss,step)

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (batch_loss, 100.*correct/total, correct, total))
        self.writer.add_scalar('train loss',train_loss, epoch)


    def save_model(self, acc, epoch):
        print('\n Saving new model with accuracy {}'.format(acc))
        state = {
            'model': self.model.module if self.use_cuda else self.net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}_ckpt.t7'.format(self.model_name_str ))

    def test(self, epoch):
        writer=self.writer
        model=self.model
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        target_all=[]
        predicted_all=[]
        print("\ntesting with previous accuracy {}".format(self.best_acc))
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            predicted_batch=predicted.eq(targets.data).cpu()
            predicted_reshaped=predicted_batch.numpy().reshape(-1)
            predicted_all=np.concatenate((predicted_all,predicted_reshaped),axis=0)

            targets_reshaped = targets.data.cpu().numpy().reshape(-1)
            target_all = np.concatenate((target_all, targets_reshaped), axis=0)
            total += targets.size(0)
            correct += predicted_batch.sum()

            progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        self.writer.add_scalar('test loss',test_loss, epoch)
        # Save checkpoint.
        acc = 100.*correct/total
        writer.add_scalar('test accuracy', acc, epoch)
        if acc > self.best_acc:
            pass
        self.save_model(acc, epoch)
        self.best_acc = acc
        print("Accuracy:{}".format(acc))
        cm = metrics.confusion_matrix(target_all, predicted_all)
        print("Confsusion metrics: \n{}".format(cm))
