import torch
import torch.nn.functional as F
from torch import nn, autograd


class InstanceM(autograd.Function):
    @staticmethod
    def forward(ctx, image_inputs, text_inputs, image_ids, features, pids, momentum):
        ctx.features = features
        ctx.pids = pids
        ctx.momentum = momentum
        ctx.save_for_backward(image_inputs, text_inputs, image_ids)
        # 使用文本特征与Memory Bank中的图像特征进行对比学习
        outputs = text_inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        image_inputs, text_inputs, image_ids = ctx.saved_tensors
        grad_text_inputs = None
        if ctx.needs_input_grad[1]:
            grad_text_inputs = grad_outputs.mm(ctx.features)

        # 使用图像特征更新Memory Bank
        for x, img_id in zip(image_inputs, image_ids):
            ctx.features[img_id] = ctx.momentum * ctx.features[img_id] + (1. - ctx.momentum) * x
            ctx.features[img_id] /= ctx.features[img_id].norm()

        return None, grad_text_inputs, None, None, None, None


def instance_memory(image_inputs, text_inputs, image_ids, features, pids, momentum=0.5):
    return InstanceM.apply(image_inputs, text_inputs, image_ids, features, pids,
                          torch.Tensor([momentum]).to(image_inputs.device))


class InstanceMemory(nn.Module):
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
        self.register_buffer('pids', torch.zeros(num_instances, dtype=torch.long))

    def forward(self, image_inputs, text_inputs, image_ids, pids):
        """
        Args:
            image_inputs: 图像特征 [batch_size, num_features]，用于更新memory bank
            text_inputs: 文本特征 [batch_size, num_features]，用于计算损失
            image_ids: 图像ID [batch_size]
            pids: 当前batch中样本对应的行人ID [batch_size]
        """
        # 更新memory bank中的pid信息
        for img_id, pid in zip(image_ids, pids):
            self.pids[img_id] = pid

        # 特征归一化
        image_inputs = F.normalize(image_inputs, dim=1).cuda()
        text_inputs = F.normalize(text_inputs, dim=1).cuda()
        
        # 计算相似度并更新memory bank
        outputs = instance_memory(image_inputs, text_inputs, image_ids, 
                                self.features, self.pids, self.momentum)
        
        # 应用温度缩放
        outputs /= self.temp
        
        # 计算InfoNCE loss，将同一pid的样本视为正样本
        loss = self.compute_nce_loss(outputs, pids)
        return loss

    def compute_nce_loss(self, scores, pids):
        """
        计算InfoNCE loss，将memory bank中与当前样本具有相同pid的样本都视为正样本
        """
        batch_size = scores.shape[0]
        pids = pids.view(batch_size, 1)
        memory_pids = self.pids.view(1, -1)  # [1, num_instances]
        
        # 创建标签矩阵，相同pid的位置为1，不同的为0
        labels = (pids == memory_pids).float()
        
        # 计算exp(sim/temp)
        exp_scores = torch.exp(scores)
        
        # 对于每个查询，计算所有正样本的exp(sim/temp)之和
        pos_exp_sum = (exp_scores * labels).sum(1)
        
        # 计算所有样本的exp(sim/temp)之和
        all_exp_sum = exp_scores.sum(1)
        
        # 计算InfoNCE loss: -log(pos_exp_sum / all_exp_sum)
        loss = -torch.log(pos_exp_sum / all_exp_sum + 1e-8)  # 添加小值避免除零
        
        return loss.mean() 