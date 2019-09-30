import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# if self.smooth:
#     one_hot_key = torch.clamp(
#         one_hot_key, self.smooth/(self.num_class-1), 1.0 - self.smooth)

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha =torch.ones(num_class, 1)
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float)

        self.gamma = gamma
        self.num_class = num_class
        self.size_average = size_average

    def forward(self, inputs, targets):
        P = F.softmax(inputs, dim=1)
        N = inputs.size(0)
        C = inputs.size(1)

        class_mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1).cpu().long()
        class_mask.scatter_(1, ids.data, 1.)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


if __name__ == '__main__':
    pred = torch.tensor([[1,20,3,4],
                       [1,9,7,1],
                       [8,0,0,10],
                       [3,30,1,2]], dtype=torch.float)/10
    target = torch.tensor([1,2,3,0], dtype=torch.int8)
    alpha = [1,1,10,1]
    criterion1 = FocalLoss(num_class=4,  alpha=alpha, gamma=2)
    loss = criterion1(pred, target)
    print(loss)
