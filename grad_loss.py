import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

class Grad_loss(nn.Module):
    def __init__(self):
        super(Grad_loss, self).__init__()
        self.lamda = 1

    def forward(self, inputs, images):
        delta_x = inputs.grad.data
        torch.clamp(delta_x, min=-0.5, max=0.5, out=delta_x)
        images.add_(delta_x * self.lamda)
        torch.clamp(images, min=-1.0, max=1.0, out=images)
        return images
        # loss, _ = torch.max(inputs.grad.data.abs().view(1, -1), dim=1)
        # print(loss)
        #
        # return loss[0] * 100


def pertubation(inputs, images):
    a = range(10,100)
    lamda = random.sample(a,1)[0]
    delta_x = inputs.grad.data
    torch.clamp(delta_x * lamda, min=-0.5, max=0.5, out=delta_x)
    images.add_(delta_x)
    torch.clamp(images, min=-1.0, max=1.0, out=images)
    images = (images*256).type(torch.cuda.IntTensor).type(torch.cuda.FloatTensor)/256
    return images

def cal_max_grad(inputs):
    delta_x = inputs.grad.data.abs()
    return torch.max(delta_x.view(-1, 1), dim=0)
