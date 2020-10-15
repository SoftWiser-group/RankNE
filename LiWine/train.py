import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
import math

import matplotlib.pyplot as plt
from sklearn import manifold
from scoring import evaluate

from utils import load_data
from models import LIWINE

parser = argparse.ArgumentParser()

parser.add_argument('--prefix', type=str, default='..', help='dir prefix')
parser.add_argument('--dataset', type=str, default='ppi', help='dataset name')
parser.add_argument('--embedding_dim', type=int, default=128, help='the output dim of final layer')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
parser.add_argument('--epochs', type=int, default=300, help='epochs')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--fin_lr', type=float, default= 5 * 1e-6, help='final learning rate')
parser.add_argument('--weight_decay', type=float, default=0.000, help='weight_decay')
parser.add_argument('--neg_sample', type=int, default=1000, help='number of negative samples')
parser.add_argument('--init', type=bool, default=True, help='whether to initialization')
parser.add_argument('--init_method', type=str, default='SVD', help='init method')
parser.add_argument('--context', type=str, default='N1', help='context')
parser.add_argument('--random_seed', type=bool, default=True, help='whether to use random seed')

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print('training on', device)
print(torch.cuda.device_count(), "GPUs")
print("batch_size", args.batch_size, "neg", args.neg_sample, "context", args.context, "init_method", args.init_method)


def train(train_data):
    G = train_data[0].todense()
    labels = train_data[1]
    normalized_term = train_data[2]
    degree = np.power(normalized_term.squeeze(), 0.75)
    degree /= np.sum(degree)
    U = train_data[3]
    V = train_data[4]
    #U /= 4

    

    num_nodes = G.shape[0]

    print(num_nodes, np.mean(normalized_term))
    F_features = nn.Embedding(num_nodes, args.embedding_dim).to(device)
    H_features = nn.Embedding(num_nodes, args.embedding_dim).to(device)
    if U is None:
        print("F random initialization")
        F_features.weight = nn.Parameter(F.normalize(torch.randn(num_nodes, args.embedding_dim), dim=1), requires_grad=True)
    else:
        print("F using initialization")
        F_features.weight = nn.Parameter(torch.FloatTensor(U), requires_grad=True)
    if V is None:
        print("H random initialization")
        H_features.weight = nn.Parameter(F.normalize(torch.randn(num_nodes, args.embedding_dim), dim=1), requires_grad=True)
    else:
        print("H using initialization")
        H_features.weight = nn.Parameter(torch.FloatTensor(V), requires_grad=True)


    model = LIWINE(args.embedding_dim,
                   F_features=F_features,
                   H_features=H_features,
                   n_nodes=num_nodes,
                   device=device).to(device)
    cnt = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            cnt +=1
    print(cnt)

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, model.parameters()),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    lr = args.lr
    for epoch in range(args.epochs):
        lr = args.fin_lr + (args.lr - args.fin_lr) * math.exp(-epoch / 9)
        for p in optimizer.param_groups:
            p['lr'] = lr
        print("Changing lr to {}".format(lr))
        start = 0
        train_arr = np.random.permutation(num_nodes)
        while start + args.batch_size < num_nodes:
            end = min(num_nodes, start + args.batch_size)
            train_nodes = train_arr[start: end]
            target = G[train_nodes]
            neg_target = (target < 1e-12).astype(float)
            neg_sample = np.random.choice(num_nodes, (args.batch_size, args.neg_sample), replace=True, p=degree)
            neg_mask = np.zeros(target.shape)
            for i in range(len(neg_mask)):
                neg_mask[i, neg_sample[i]] = 1
            neg_target = np.multiply(neg_target, neg_mask)
            all_target = (target + neg_mask != 0).astype(float)
            batch_normalized_term = normalized_term[train_nodes]
            target /= batch_normalized_term
            loss = model.loss(torch.LongTensor(train_nodes).to(device),
                              torch.FloatTensor(target).to(device),
                              torch.FloatTensor(neg_target).to(device),
                              torch.FloatTensor(all_target).to(device))
            loss.backward()
            optimizer.step()

            if (start / args.batch_size) % 6 == 0:
                print("epoch {:3} from {:6} to {:6}: loss {:.5f}".format(epoch, start, end, loss.data.item()/args.batch_size))

            start += args.batch_size

    all_nodes = [item for item in range(num_nodes)]
    output_features = model.forward(torch.LongTensor(all_nodes).to(device))
    output_features = output_features.cpu().data.numpy()

    evaluate(output_features, labels, args.dataset)
    output_file = '{}/data/embedding/{}_LIWINE_{}.txt'.format(args.prefix, args.dataset, args.init_method)
    with open(output_file, 'w') as f:
        print(len(output_features), len(output_features[0]), file=f)
        for i in range(len(output_features)):
            print(i, end='', file=f)
            for item in output_features[i]:
                print(' {}'.format(item), file=f, end='')
            print('', file=f)
        f.close()




def visualize(features, labels, title):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    features_2D = tsne.fit_transform(features)
    plt.scatter(features_2D[:, 0], features_2D[:, 1], c=labels, marker='.', cmap=plt.cm.rainbow)
    # plt.title(title)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    


def main():
    if args.random_seed:
        np.random.seed(1)
        random.seed(1)
        torch.manual_seed(1) 
    print('Loading {} data'.format(args.dataset))
    train_data = load_data(args.prefix, args.dataset, args.init, args.init_method, args.context)
    print("Loading completed. Training starts")
    train(train_data)



if __name__ == "__main__":
    main()