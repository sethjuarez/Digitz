import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from misc.digits import Digits
import torch.nn.functional as F
from misc.helpers import print_info, print_args, check_dir, info, save_model

def main(args):
    # digit data
    digits = Digits(args.data, args.batch)
    test_x, test_y = digits.test

    _, size_x = test_x.shape
    _, size_y = test_y.shape

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    
    model = nn.Sequential(nn.Linear(size_x, size_y))
    loss = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # optimization loop
    for epoch in range(args.epochs):
        info("Epoch {}".format(epoch+1))
        for i, (train_x, train_y) in enumerate(digits):
            optimizer.zero_grad()
            y = model(train_x)
            cost = loss(y, train_y)
            c = cost.item()
            cost.backward()
            optimizer.step()
            print("\r Batch {}/{} - Cost {}".format(i+1, digits.total,c), end="")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN Training for Image Recognition.')
    parser.add_argument('-d', '--data', help='directory to training and test data', default='data')
    parser.add_argument('-e', '--epochs', help='number of epochs', default=10, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=100, type=int)
    parser.add_argument('-l', '--lr', help='learning rate', default=0.001, type=float)
    parser.add_argument('-o', '--output', help='output directory', default='output')
    args = parser.parse_args()

    args.data = check_dir(os.path.abspath(args.data))
    args.output = os.path.abspath(args.output)
    unique = datetime.now().strftime('%m.%d_%H.%M')
    args.log = check_dir(os.path.join(args.output, 'logs', 'log_{}'.format(unique)))
    args.model = check_dir(os.path.join(args.output, 'models', 'model_{}'.format(unique)))
    
    main(args)