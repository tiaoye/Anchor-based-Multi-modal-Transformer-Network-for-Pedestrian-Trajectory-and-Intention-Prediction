import torch
import torch.nn.functional as F
import pdb

from torch import nn
import torch
from torch.nn import functional as F

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 2, size_average=True):
        """
        focal_loss损失函数, -alpha(1-yi)**gamma *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[alpha, 1-alpha, 1-alpha, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels:torch.Tensor):
        """
        focal_loss calculation
        :param preds: size:[B,C]
        :param labels: size:[B]
        :return:
        """
        labels = labels.to(dtype=torch.int64)
        assert preds.dim()==2 and labels.dim()==1
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

def mutual_inf_mc(x_dist):
    dist = x_dist.__class__
    H_y = dist(probs=x_dist.probs.mean(dim=0)).entropy()
    return (H_y - x_dist.entropy().mean(dim=0)).sum()

def cvae_loss(pred_goal, pred_traj, target, select_metric, best_of_many=True):
        '''
        please select traj_rmse when no pred_goal.
        Params:
            pred_goal: (Batch, K, pred_dim)
            pred_traj: (Batch, T, K, pred_dim)
            target: (Batch, T, pred_dim)
            select_metric: 'goal_rmse' or 'traj_rmse'
            best_of_many: whether use best of many loss or not
        Returns:
            loss_goal: (1)
            loss_traj: (1)
            best_idx: (Batch)
        '''
        K = pred_goal.shape[1] if pred_goal is not None else pred_traj.shape[2]
        batch_size = pred_traj.shape[0]
        target = target.unsqueeze(2).repeat(1, 1, K, 1) # (Batch, T, K, pred_dim)
    
        # select bom based on  goal_rmse
        goal_rmse = torch.sqrt(torch.sum((pred_goal - target[:, -1, :, :])**2, dim=-1)) if pred_goal is not None else torch.zeros(batch_size,K,device=pred_traj.device)# (Batch, K)
        traj_rmse = torch.sqrt(torch.sum((pred_traj - target)**2, dim=-1)).sum(dim=1) # (Batch, K) totoal rmse of each traj_step
        if best_of_many:
            best_idx = torch.argmin(goal_rmse, dim=1) if select_metric == 'goal_rmse' else torch.argmin(traj_rmse, dim=1)
            loss_goal = goal_rmse[torch.arange(best_idx.size(0),device=best_idx.device), best_idx].mean() # (1)
            loss_traj = traj_rmse[torch.arange(best_idx.size(0),device=best_idx.device), best_idx].mean() # (1)
        else:
            loss_goal = goal_rmse.mean()
            loss_traj = traj_rmse.mean()
        
        return loss_goal, loss_traj, best_idx
        
def bom_traj_loss(pred, target):
    '''
    pred: (B, T, K, dim)
    target: (B, T, dim)
    '''
    K = pred.shape[2]
    target = target.unsqueeze(2).repeat(1, 1, K, 1)
    traj_rmse = torch.sqrt(torch.sum((pred - target)**2, dim=-1)).sum(dim=1)
    best_idx = torch.argmin(traj_rmse, dim=1)
    loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
    return loss_traj

def fol_rmse(x_true, x_pred):
    '''
    Params:
        x_pred: (batch, T, pred_dim) or (batch, T, K, pred_dim)
        x_true: (batch, T, pred_dim) or (batch, T, K, pred_dim)
    Returns:
        rmse: scalar, rmse = \sum_{i=1:batch_size}()
    '''

    L2_diff = torch.sqrt(torch.sum((x_pred - x_true)**2, dim=-1))#
    L2_diff = torch.sum(L2_diff, dim=-1).mean()

    return L2_diff