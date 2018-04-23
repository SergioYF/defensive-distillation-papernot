import sys
import os
import argparse
import warnings
from time import time

import numpy as np
import torch
from torch.autograd import Variable
from torch import FloatTensor, LongTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import inception_v3
from IPython import embed
ALPHA = 0.9
GAMMA = 5e-4
KAPPA = 40
BATCH_SIZE = 64
LOG_INTERVAL = 640 / BATCH_SIZE

PROJ_NAME = 'distillation'

CUR_PATH = os.getcwd()
PROJ_PATH = CUR_PATH.split(PROJ_NAME)[0] + PROJ_NAME + '/'

pretrain_model_path = PROJ_PATH + 'mnist_papernot/mnist_train.pth'

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, metavar='N1',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N2',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N3',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--alpha', type=float, default=ALPHA, metavar='A',
                    help='number of epochs to train (default: 0.9)')
parser.add_argument('--gamma', type=float, default=GAMMA, metavar='G',
                    help='number of epochs to train (default: 0.9)')
parser.add_argument('--kappa', type=float, default=KAPPA, metavar='K',
                    help='number of epochs to train (default: 0.9)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=LOG_INTERVAL, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def get_train_loader():
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    return train_loader

def get_test_loader():
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    return test_loader

class Net(nn.Module):
    def __init__(self, output_units=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, output_units)

    def logits(self, x, drop_out=False):
        if drop_out:
            x = F.dropout(x, p=0.2, training=True)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        if drop_out:
            x = F.dropout(x, p=0.5, training=True)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

    def forward(self, x, softmax=True, drop_out=False):
        x = self.logits(x, drop_out=drop_out)
        if softmax:
            # return F.log_softmax(x)
            return F.softmax(x)
        else:
            return x

def load_dict(pth_path=pretrain_model_path):
    state_dict = torch.load(pth_path)
    return state_dict

def relabel(model, data_loader=None, test_size=None, test_times=20, normalize=True, save_mark=None):

    if data_loader is None:
        data_loader = get_train_loader()

    model.eval()
    # model.train()

    def judge_uncertainty_np(array):
        N, n = array.shape
        ave = array.mean(axis=0)
        unc = np.zeros(N)
        for i in range(N):
            unc_vector = array[i] - ave
            unc[i] = unc_vector.dot(unc_vector)
        sigma = unc.mean()
        return sigma

    def calc_uncertainty(x, model=model, test_times=20):
        if isinstance(x, Variable):
            x_shape = list(x.data.size())
        elif isinstance(x, FloatTensor):
            x_shape = list(x.size())
            warnings.warn("x should be type of torch.Variable")
        else:
            raise TypeError
        assert len(x_shape) == 4
        sigma_list = []
        for i in range(int(x_shape[0])):
            var = x[i:i+1]
            logits_result = np.zeros((test_times, 10))
            softmax_result = np.zeros((test_times, 10))
            log_softmax_result = np.zeros((test_times, 10))
            for j in range(test_times):
                logits_var = model(var, softmax=False, drop_out=True)
                logits = logits_var.data.numpy()
                sm = F.softmax(logits_var).data.numpy()
                logits_result[j] = logits[0]
                softmax_result[j] = sm[0]
                log_softmax_result[j] = np.log(sm[0])
            sigma = judge_uncertainty_np(logits_result)
            sigma_list.append(sigma)
        return sigma_list

    sigma_list = []

    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        sigmas = calc_uncertainty(data, model=model, test_times=test_times)
        sigma_list.extend(sigmas)
        if batch_idx % args.log_interval == 0:
            print('Labeling: [{}/{} ({:.0f}%)]'.format(
                batch_idx * len(data), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader)))
        if test_size is not None:
            if len(sigma_list) > test_size:
                break

    np_list = np.array(sigma_list)
    if normalize:
        max = np_list.max() # base could be altered into m + 4 * s
        sigmas = np_list / max
        if save_mark is not None:
            np.savez('sigmas_{}'.format(save_mark), sigmas=sigmas, base=max)
        else:
            np.savez('sigmas', sigmas=sigmas, base=max)
    else:
        if save_mark is not None:
            np.save('./uncertainty_{}'.format(save_mark), np_list)
        else:
            np.save('./uncertainty', np_list)
    return np_list

def process_sigma(uncertainty, save_mark=None):
    from copy import deepcopy
    ordered = deepcopy(uncertainty)
    ordered = np.sort(ordered)[::-1]
    length = ordered.shape[0]
    base_1 = ordered[int(length/100)]
    sigmas = np.zeros(length)
    for i in range(length):
        if uncertainty[i] >= base_1:
            sigmas[i] = 1
        else:
            sigmas[i] = uncertainty[i] / base_1
    if save_mark is not None:
        np.save('./sigma_{}'.format(save_mark), sigmas)
    else:
        np.save('./sigma', sigmas)
    return sigmas

def train_papernot(sigmas, epochs=10, output_units=11):
    # 20 is too little for calculation for uncertainty
    assert len(sigmas.shape) == 1
    kappa_var = Variable(FloatTensor([args.kappa]), requires_grad=False)
    sigmas = sigmas * args.alpha
    train_loader = get_train_loader()
    model_papernot = Net(output_units=output_units)
    optimizer = optim.SGD(model_papernot.parameters(), lr=args.lr, momentum=args.momentum)
    sigma_list = [sigmas[i] for i in range(sigmas.shape[0])]
    for epoch in range(1, epochs+1):
        model_papernot.train()
        gen = sigma_list.__iter__()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            N = target.data.numpy().shape[0]
            sigmas = [gen.next() for _ in range(N)]
            tens = Variable(LongTensor([10 for _ in range(N)]))
            optimizer.zero_grad()
            logits = model_papernot(data, softmax=False)
            output = F.log_softmax(logits)

            loss1 = Variable(FloatTensor([0.]))
            for i in range(N):
                loss_plus = F.nll_loss(output[i:i+1], target[i:i+1])
                loss1 = loss1 + torch.mul(loss_plus, (1-sigmas[i]))
            loss2 = Variable(FloatTensor([0.]))
            for i in range(N):
                loss_plus = F.nll_loss(output[i:i+1], tens[i:i+1])
                loss2 = loss2 + torch.mul(loss_plus, sigmas[i])
            loss3 = Variable(FloatTensor([0.]))
            for i in range(N):
                inner_max = torch.max(logits[i][0:output_units-1])
                label_index = target[i].data.numpy()[0]
                label_compo = logits[i][label_index]
                middle = inner_max - label_compo
                loss_plus = torch.max(middle, kappa_var)
                loss3 = loss3 + torch.mul(loss_plus, args.gamma)

            loss = loss1 + loss2 + loss3
            loss = torch.div(loss, float(N))
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))
    torch.save(model_papernot.state_dict(), PROJ_PATH+'mnist_papernot/mnist_papernot.pth')
    return model_papernot

def test_papernot_model(model, sigmas, test_loader=None):
    def adjust_target(target, generator, alpha=args.alpha, output_units=11):
        if isinstance(target, FloatTensor):
            target = target.numpy()
            # warnings.warn("target should be of type Variable")
        elif isinstance(target, Variable):
            target = target.data.numpy()
        N = target.shape[0]
        sigma_list = []
        for _ in range(N):
            sigma = generator.next()
            sigma_list.append([alpha * sigma])
        sigma_np = np.array(sigma_list)
        new_target = np.zeros((N, output_units - 1))
        for i in range(N):
            new_target[i, target[i]] = 1 - sigma_np[i]
        new_target = np.concatenate((new_target, sigma_np), axis=1)
        new_target = Variable(FloatTensor(new_target))
        return new_target
    if test_loader is None:
        test_loader = get_test_loader()
    # else:
        # warnings.warn("You are not using test loader to test accuracy!")
    model.eval()
    sigma_list = [sigmas[i] for i in range(sigmas.shape[0])]
    gen = sigma_list.__iter__()
    criterion = torch.nn.MSELoss(size_average=False)
    test_loss = 0
    correct = 0
    doubt_correct = 0
    result = np.zeros((1, 5))
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        target = adjust_target(target, generator=gen)
        output = model(data)
        test_loss += criterion(output, target).data[0] # sum up batch loss
        pred_doubt = output.data.max(1)[1] # get the index of the max log-probability
        label_doubt = target.data.max(1)[1]
        pred = output.data[:, 0:10].max(1)[1]
        label = target.data[:, 0:10].max(1)[1]
        correct += pred.eq(label).cpu().sum()
        doubt_correct += pred_doubt.eq(label_doubt).cpu().sum()

        sigmas_calced = output[:, 10].data.numpy()
        size = sigmas_calced.size
        sigmas_trans = sigmas_calced.reshape([size, 1])
        mat = np.concatenate([pred.numpy(), pred_doubt.numpy(), label.numpy(), label_doubt.numpy(), sigmas_trans], axis=1)
        result = np.concatenate((result, mat), axis=0)
    result = result[1:, :]
    np.save(PROJ_PATH+'mnist_papernot/result_{}'.format(args.alpha), result)

    test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Doubt Accuracy: {}/{} ({:.0f}%)\n'.
    #     format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset),
    #            doubt_correct, len(test_loader.dataset), 100. * doubt_correct / len(test_loader.dataset)))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{}, Doubt Accuracy: {}/{}'.
          format(test_loss, correct, len(test_loader.dataset), doubt_correct, len(test_loader.dataset)))

def inspect_model_result(result_path=PROJ_PATH+'mnist_papernot/result_{}.npy'.format(args.alpha)):
    # pred, pred_doubt, label, label_doubt, sigma
    result = np.load(result_path)
    size = result.shape[0]
    sigma_mean = result[:,4].mean()
    pred_doubt_list = np.array([result[i, 1] == 10 for i in range(size)])
    pred_doubt_sum = pred_doubt_list.sum()
    label_doubt_list = np.array([result[i, 3] == 10 for i in range(size)])
    label_doubt_sum = label_doubt_list.sum()
    all_doubt_list = np.array([result[i, 1] == 10 and result[i, 3] == 10 for i in range(size)])
    all_doubt_sum = all_doubt_list.sum()
    print("Mean uncertainty:{0}, Doubts in prediction:{1}, Doubts in label:{2}, "
          "Doubts predicted right: {3}".format(sigma_mean, pred_doubt_sum, label_doubt_sum, all_doubt_sum))

