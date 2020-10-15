import torch
import torch.nn as nn
import torch.nn.functional as F


class LIWINE(nn.Module):
    def __init__(self, embedding_dim, F_features, H_features, n_nodes, device='cpu'):
        super(LIWINE, self).__init__()

        self.F_features = F_features
        self.H_features = H_features
        self.embedding_dim = embedding_dim
        self.n_node = n_nodes
        self.device = device

        self.act = MySigmoid.apply



    def forward(self, nodes):
        return self.F_features(nodes)

    def loss(self, nodes, target, neg_target, all_target):
        output = self.forward(nodes)

        output = torch.matmul(output, self.H_features.weight.t())
        output = self.act(torch.clamp(output, min=-9.9, max=9.9))

        output = output / torch.sum(torch.mul(output, all_target), dim=1, keepdim=True)
        neg_output = torch.log(1 - output)
        output = torch.log(output)

        loss = torch.sum(torch.sum(-torch.mul(output, target), dim=1)) + torch.sum(torch.sum(-torch.mul(neg_output, neg_target), dim=1))
        return loss



class MySigmoid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        input_ = input.clamp(min=-9.9, max=9.9)
        output = torch.sigmoid(input_)
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = grad_input * output * (1.0 - output)
        grad_input[input < -9.9] = 0
        grad_input[input > 9.9] = 0
        return grad_input
