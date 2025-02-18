import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, image_inputs, text_inputs, targets, features, momentum):
        # 保存必要的张量以便在后向传播时使用
        ctx.save_for_backward(image_inputs, text_inputs, targets)
        ctx.features = features
        ctx.momentum = momentum

        # 使用图像特征与Memory Bank中的文本特征进行对比学习
        outputs = image_inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        # image_inputs计算损失，text_inputs更新memory bank
        image_inputs, text_inputs, targets = ctx.saved_tensors

        # 梯度初始化为None
        grad_image_inputs = None

        # 如果需要输入的梯度，计算图像特征的梯度
        if ctx.needs_input_grad[0]:
            grad_image_inputs = grad_outputs.mm(ctx.features)

        # 使用文本特征更新Memory Bank 
        for x, y in zip(text_inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_image_inputs, None, None, None, None


def cm(image_inputs, text_inputs, targets, features, momentum=0.5):
    return CM.apply(image_inputs, text_inputs, targets, features, torch.Tensor([momentum]).to(image_inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, image_inputs, text_inputs, targets, features, momentum):
        ctx.save_for_backward(image_inputs, text_inputs, targets)
        ctx.features = features
        ctx.momentum = momentum

        # 使用图像特征与Memory Bank中的文本特征进行对比学习
        outputs = image_inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        image_inputs, text_inputs, targets = ctx.saved_tensors

        grad_image_inputs = None
        if ctx.needs_input_grad[0]:
            grad_image_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(text_inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        # 选择最难的样本进行更新
        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            # 选择距离最小的文本特征
            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_image_inputs, None, None, None


def cm_hard(image_inputs, text_inputs, targets, features, momentum=0.5):
    return CM_Hard.apply(image_inputs, text_inputs, targets, features, torch.Tensor([momentum]).to(image_inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.02, momentum=0.2, use_hard=False, margin=0.2):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.margin = margin
        # Memory Bank 存储文本特征
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        

    def forward(self, image_inputs, text_inputs, targets):
        # 对图像特征和文本特征进行归一化
        image_inputs = F.normalize(image_inputs, dim=1).cuda()
        text_inputs = F.normalize(text_inputs, dim=1).cuda()

        # 根据是否使用Hard策略选择cm或cm_hard
        if self.use_hard:
            outputs = cm_hard(image_inputs, text_inputs, targets, self.features, self.momentum)
        else:
            outputs = cm(image_inputs, text_inputs, targets, self.features, self.momentum)

        # 缩放输出
        outputs /= self.temp
        
        

        # 计算交叉熵损失
        loss = F.cross_entropy(outputs, targets)
        
        # 修改为TAL loss形式
        # loss = compute_TAL(outputs, targets, tau=self.temp, margin=self.margin)
        return loss

def compute_TAL(scores, pid, tau=0.02, margin=0.2):
    batch_size =scores.shape[0]
    pid = pid.view(batch_size, 1)
    memory_ids = torch.arange(scores.shape[1]).to(pid.device).view(1, -1)  # [1, num_samples]
    labels = (pid == memory_ids).float()
    mask = 1- labels
    pos_sim = (scores * labels).sum(1) 
    neg_sim = tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log()
    loss = (-pos_sim+neg_sim+margin).clamp(min=0)
    return loss.sum()



class InstanceM(autograd.Function):
    @staticmethod
    def forward(ctx, image_inputs, text_inputs, image_ids, features, pids, momentum):
        ctx.features = features
        ctx.pids = pids
        ctx.momentum = momentum
        ctx.save_for_backward(image_inputs, text_inputs, image_ids)
        # 使用图像特征与Memory Bank中的文本特征进行对比学习
        outputs = image_inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        image_inputs, text_inputs, image_ids = ctx.saved_tensors
        grad_image_inputs = None
        if ctx.needs_input_grad[0]:
            grad_image_inputs = grad_outputs.mm(ctx.features)

        # 使用图像特征更新Memory Bank
        for x, img_id in zip(text_inputs, image_ids):
            ctx.features[img_id] = ctx.momentum * ctx.features[img_id] + (1. - ctx.momentum) * x
            ctx.features[img_id] /= ctx.features[img_id].norm()

        return grad_image_inputs, None, None, None, None, None


def instance_memory(image_inputs, text_inputs, image_ids, features, pids, momentum=0.5):
    return InstanceM.apply(image_inputs, text_inputs, image_ids, features, pids,
                          torch.Tensor([momentum]).to(image_inputs.device))


class InstanceMemory(nn.Module, ABC):
    def __init__(self, num_features, num_instances, temp=0.05, momentum=0.2, margin=0.2):
        """
        Args:
            num_features: 特征维度
            num_instances: 图像实例的总数量
            temp: 温度参数
            momentum: 动量参数
            margin: TAL损失的margin参数
        """
        super(InstanceMemory, self).__init__()
        self.num_features = num_features
        self.num_instances = num_instances
        self.momentum = momentum
        self.temp = temp
        self.margin = margin
        
        # 为每个图像实例初始化特征向量和对应的pid
        self.register_buffer('features', torch.zeros(num_instances, num_features))
        self.register_buffer('pids', torch.zeros(num_instances))

    def forward(self, image_inputs, text_inputs, image_ids, pids):
        """
        Args:
            image_inputs: 图像特征 [batch_size, num_features]，用于更新memory bank
            text_inputs: 文本特征 [batch_size, num_features]，用于计算损失
            image_ids: 图像ID [batch_size]
            pids: 当前batch中样本对应的行人ID [batch_size]
        """
        # 更新memory bank中的pid信息
        # for img_id, pid in zip(image_ids, pids):
        #     self.pids[img_id] = pid

        # 特征归一化
        image_inputs = F.normalize(image_inputs, dim=1).cuda()
        text_inputs = F.normalize(text_inputs, dim=1).cuda()
        
        # 计算相似度并更新memory bank
        outputs = instance_memory(image_inputs, text_inputs, image_ids, 
                                self.features, self.pids, self.momentum)
        
        # 应用温度缩放
        outputs /= self.temp
        
        # 计算InfoNCE loss，将同一pid的样本视为正样本
        # loss = self.compute_TAL_instance(outputs, pids, tau=self.temp, margin=self.margin)
        loss = self.compute_nce_loss(outputs, pids)
        return loss
    
    # def compute_TAL(self,scores, pid, tau=0.02, margin=0.2):
    #     batch_size =scores.shape[0]
    #     pid = pid.view(batch_size, 1)
    #     # memory_ids = torch.arange(scores.shape[1]).to(pid.device).view(1, -1)  # [1, num_samples]
    #     memory_ids = self.pids.view(1, -1)  # [1, num_instances]
    #     labels = (pid == memory_ids).float()
    #     mask = 1- labels
    #     # alpha =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    #     pos_sim = (scores * labels).sum(1) 
    #     neg_sim = tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log()
    #     loss = (-pos_sim+neg_sim+margin).clamp(min=0)
    #     # loss = (-  (alpha*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)  \
       
    #     return loss.sum()
    
    def compute_TAL_instance(self,scores, pid, tau=0.02, margin=0.2):
        batch_size =scores.shape[0]
        pid = pid.view(batch_size, 1)
        # memory_ids = torch.arange(scores.shape[1]).to(pid.device).view(1, -1)  # [1, num_samples]
        memory_ids = self.pids.view(1, -1)  # [1, num_instances]
        labels = (pid == memory_ids).float()
        mask = 1- labels
        alpha =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
        # pos_sim = (scores * labels).sum(1) 
        # neg_sim = tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log()
        # loss = (-pos_sim+neg_sim+margin).clamp(min=0)
        loss = (-  (alpha*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)  
       
        return loss.sum()
    
    # def compute_nce_loss(self, scores, pids):
    #     """
    #     计算InfoNCE loss,将memory bank中与当前样本具有相同pid的样本都视为正样本
    #     使用正样本相似度的平均值
    #     """
    #     batch_size = scores.shape[0]
    #     pids = pids.view(batch_size, 1)
    #     memory_pids = self.pids.view(1, -1)  # [1, num_instances]
        
    #     # 创建标签矩阵，相同pid的位置为1，不同的为0
    #     labels = (pids == memory_pids).float()
        
    #     # 计算exp(sim/temp)
    #     exp_scores = torch.exp(scores)
    #     exp_scores_masked = exp_scores * labels  # 只保留正样本的分数
    #     # 对于每个查询，计算所有正样本的exp(sim/temp)之和
    #     pos_exp_sum = (exp_scores * labels).sum(1)
    #     exp_scores_masked[labels == 0] = -1e9
        
    #     top3_exp_scores, _ = torch.topk(exp_scores_masked, k=min(3, int(labels.sum().item())), dim=1)
    #     # 计算InfoNCE loss: -log(pos_exp_mean / all_exp_sum)
    #     pos_exp_sum = top3_exp_scores.sum(1)
    #     all_exp_sum = exp_scores.sum(1)
    #     loss = -torch.log(pos_exp_sum / all_exp_sum + 1e-8)  # 添加小值避免除零
        
    #     return loss.mean() 
    
    def compute_nce_loss(self, scores, pids):
        """
    #     计算InfoNCE loss,将memory bank中与当前样本具有相同pid的样本都视为正样本
    #     使用正样本相似度的平均值
    #     """
        batch_size = scores.shape[0]
        pids = pids.view(batch_size, 1)
        memory_pids = self.pids.view(1, -1)  # [1, num_instances]
        
        # 创建标签矩阵，相同pid的位置为1，不同的为0
        labels = (pids == memory_pids).float()
        
        # 计算exp(sim/temp)
        exp_scores = torch.exp(scores)
        
        # 对于每个查询，计算所有正样本的exp(sim/temp)之和
        pos_exp_sum = (exp_scores * labels).sum(1)
        # 对于每个查询，计算所有正样本的exp(sim/temp)的平均值
        # 首先计算每个查询的正样本数量
        # num_pos = labels.sum(1)  # [batch_size]
        # 计算正样本的exp(sim/temp)平均值
        # pos_exp_mean = (exp_scores * labels).sum(1) / (num_pos)  # 添加小值避免除零
        
        # 计算所有样本的exp(sim/temp)之和
        all_exp_sum = exp_scores.sum(1)
        
        # 计算InfoNCE loss: -log(pos_exp_mean / all_exp_sum)
        loss = -torch.log(pos_exp_sum / all_exp_sum + 1e-8)  # 添加小值避免除零
        
        return loss.mean()
    
    
    