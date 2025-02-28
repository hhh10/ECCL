#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List, Tuple
# from ..torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_

Tensor = torch.Tensor

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True
        
    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)
            
class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        
class Linear_gate(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        self.lora_gate_layer = nn.Linear(in_features, 1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        # nn.init.xavier_uniform_(self.gate_layer.weight)
        # if self.gate_layer.bias is not None:
        #     nn.init.zeros_(self.gate_layer.bias)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    pass
                self.merged = False
                #     self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                # self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            lora_output = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
           
            gate_value = F.sigmoid(self.lora_gate_layer(x))
            lora_output = lora_output * gate_value
            result += lora_output
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        
class Linear_share(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        shared_lora=None,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.shared_lora = shared_lora
        
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.shared_lora @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.shared_lora @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.shared_lora @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        

        
def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight  
     
class Linear_MoE(nn.Linear, LoRALayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 4,
        blc_alpha: float = 0.0,
        blc_weight: float = 0.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        top_k: int = 1,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.lora_num = lora_nums
        self.blc_alpha = blc_alpha
        self.blc_weight = blc_weight
        self.top_k = top_k
        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        if r > 0:
            self.lora_route = nn.Linear(in_features, self.lora_num, bias=False)
            for i in range(self.lora_num):
                setattr(self, f"lora_A{i}", nn.Linear(in_features, r, bias=False))
                setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        
        if hasattr(self, "lora_A0"):
            for i in range(self.lora_num):
                nn.init.kaiming_uniform_(getattr(self, f"lora_A{i}").weight, a=math.sqrt(5))
                nn.init.zeros_(getattr(self, f"lora_B{i}").weight)

            nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_route.train(mode)
        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").train(mode)
            getattr(self, f"lora_B{i}").train(mode)

    def eval(self):
        nn.Linear.eval(self)
        self.lora_route.eval()
        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").eval()
            getattr(self, f"lora_B{i}").eval()

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)[0]
        return x.float().var() / (x.float().mean()**2 + eps)

    def forward(self, x, task_types=None):
        if self.lora_A0.training:
            if isinstance(x,list):
                x,pre_loss=x[0],x[1] 
            else:
                pre_loss=0
        
        if self.disable_adapters:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            raise ImportError(":(") 
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            
            sequence_length, batch_size, hidden_dim = x.shape
            x = x.view(-1,hidden_dim) 
            if self.r > 0:
                # logits = self.lora_route(x)
                # noise = torch.randn_like(logits)*F.softplus(self.noise(x))
                # noisy_logits = logits + noise
                # route_weight = nn.functional.softmax(noisy_logits, dim=-1, dtype=torch.float32).to(result.dtype)
                
                route_weight = nn.functional.softmax(self.lora_route(x), dim=-1, dtype=torch.float32).to(result.dtype)
                
                topk_weights, topk_experts = torch.topk(route_weight, self.top_k, dim=-1)
                
                a=0
                b=0
                c=0
                d=0
                for i in topk_experts:
                    e1,e2 = i
                    if e1 == 0:
                        a+=1
                    elif e1 == 1:
                        b+=1
                    elif e1 ==2:
                        c+=1
                    elif e1 ==3:
                        d+=1
                    if e2 == 0:
                        a+=1
                    elif e2 == 1:
                        b+=1
                    elif e2 ==2:
                        c+=1
                    elif e2 ==3:
                        d+=1
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
                
                expert_mask = F.one_hot(topk_experts, num_classes=self.lora_num).permute(2,1,0)
                
                final_hidden_states = torch.zeros(
                    (batch_size*sequence_length, self.out_features), dtype=x.dtype, device=x.device
                )
                
                for expert_idx in range(self.lora_num):
                    idx, top_x = torch.where(expert_mask[expert_idx])
                    if top_x.shape[0] == 0:
                        continue
                    top_x_list = top_x.tolist()
                    idx_list = idx.tolist()
                    
                    current_state = x[None, top_x_list].reshape(-1, hidden_dim)
                    expert_output = getattr(self, f"lora_B{expert_idx}")(getattr(self, f"lora_A{expert_idx}")(self.lora_dropout(current_state))) * self.scaling
                    current_hidden_states = expert_output * topk_weights[top_x_list, idx_list, None]
                    final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))
                
                final_hidden_states = final_hidden_states.reshape(sequence_length, batch_size, self.out_features)
                result += final_hidden_states
                if self.lora_A0.training:
                    topk_experts = topk_experts.view(-1)
                    expert_mask = F.one_hot(topk_experts, num_classes=self.lora_num)
                    expert_mask1 = torch.sum(expert_mask, dim=-2)
                    
                    tokens_per_expert = expert_mask1 / torch.sum(expert_mask1)
                    
                    router_prob_per_expert = torch.mean(route_weight, dim=0)
                    layer_loss = torch.sum(tokens_per_expert * router_prob_per_expert) * self.lora_num
                    # layer_loss = torch.sum(tokens_per_expert * router_prob_per_expert) * self.lora_num
                    
                    layer_loss += pre_loss
                    
                                
                    # for i in range(self.lora_num):
                    #     result = result + torch.unsqueeze(route_weight[:,:,i], -1) * getattr(self, f"lora_B{i}")(getattr(self, f"lora_A{i}")(self.lora_dropout(x))) * self.scaling
                    return [result,layer_loss]
                else:
                    return result
        # blcls = torch.zeros(1)[0].to(result)
        # if task_types != None:
        #     if self.blc_weight != 0:
        #         task_types = task_types.view(-1, 1)
        #         blcls = self.cv_squared((
        #             route_weight.sum(dim=(1)) * torch.where(
        #                 torch.concat(
        #                     ((task_types==1).repeat(1, self.lora_num//2), (task_types==0).repeat(1, self.lora_num//2)), dim=-1
        #                     ), 1.0+self.blc_alpha, 1.0-self.blc_alpha
        #                 )
        #             ).flatten()
        #         ) * self.blc_weight

        # return result, blcls
# class Linear(nn.Linear, LoRALayer):
#     # LoRA implemented in a dense layer
#     def __init__(
#         self, 
#         in_features: int, 
#         out_features: int, 
#         r: int = 0, 
#         lora_alpha: int = 1, 
#         lora_dropout: float = 0.,
#         fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
#         merge_weights: bool = True,
#         **kwargs
#     ):
#         nn.Linear.__init__(self, in_features, out_features, **kwargs)
#         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
#                            merge_weights=merge_weights)

#         self.fan_in_fan_out = fan_in_fan_out
#         # Actual trainable parameters
#         if r > 0:
#             self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
#             self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
#             self.scaling = self.lora_alpha / self.r
#             # Freezing the pre-trained weight matrix
#             self.weight.requires_grad = False
#         self.reset_parameters()
#         if fan_in_fan_out:
#             self.weight.data = self.weight.data.transpose(0, 1)

#     def reset_parameters(self):
#         nn.Linear.reset_parameters(self)
#         if hasattr(self, 'lora_A'):
#             # initialize A the same way as the default for nn.Linear and B to zero
#             nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#             nn.init.zeros_(self.lora_B)

#     def train(self, mode: bool = True):
#         def T(w):
#             return w.transpose(0, 1) if self.fan_in_fan_out else w
#         nn.Linear.train(self, mode)
#         if mode:
#             if self.merge_weights and self.merged:
#                 # Make sure that the weights are not merged
#                 if self.r > 0:
#                     self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
#                 self.merged = False
#         else:
#             if self.merge_weights and not self.merged:
#                 # Merge the weights and mark it
#                 if self.r > 0:
#                     self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
#                 self.merged = True       

#     def forward(self, x: torch.Tensor):
#         def T(w):
#             return w.transpose(0, 1) if self.fan_in_fan_out else w
#         if self.r > 0 and not self.merged:
#             result = F.linear(x, T(self.weight), bias=self.bias)            
#             result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
#             return result
#         else:
#             return F.linear(x, T(self.weight), bias=self.bias)
        
# def transpose(weight, fan_in_fan_out):
#     return weight.T if fan_in_fan_out else weight  
     
# class Linear_MoE(nn.Linear, LoRALayer):
#     # Lora implemented in a dense layer
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         r: int = 0,
#         lora_alpha: int = 1,
#         lora_nums: int = 2,
#         blc_alpha: float = 0.0,
#         blc_weight: float = 0.0,
#         lora_dropout: float = 0.0,
#         fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
#         merge_weights: bool = True,
#         **kwargs,
#     ):
#         nn.Linear.__init__(self, in_features, out_features, **kwargs)
#         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

#         self.lora_num = lora_nums
#         self.blc_alpha = blc_alpha
#         self.blc_weight = blc_weight
        
#         self.fan_in_fan_out = fan_in_fan_out

#         # Actual trainable parameters
#         if r > 0:
#             self.lora_route = nn.Linear(in_features, self.lora_num, bias=False)
#             for i in range(self.lora_num):
#                 setattr(self, f"lora_A{i}", nn.Linear(in_features, r, bias=False))
#                 setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))

#             self.scaling = self.lora_alpha / self.r
#             # Freezing the pre-trained weight matrix
#             self.weight.requires_grad = False
#         self.reset_parameters()
#         if fan_in_fan_out:
#             self.weight.data = self.weight.data.T

#     def reset_parameters(self):
#         nn.Linear.reset_parameters(self)
        
#         if hasattr(self, "lora_A0"):
#             for i in range(self.lora_num):
#                 nn.init.kaiming_uniform_(getattr(self, f"lora_A{i}").weight, a=math.sqrt(5))
#                 nn.init.zeros_(getattr(self, f"lora_B{i}").weight)

#             nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

#     def train(self, mode: bool = True):
#         nn.Linear.train(self, mode)
#         self.lora_route.train(mode)
#         for i in range(self.lora_num):
#             getattr(self, f"lora_A{i}").train(mode)
#             getattr(self, f"lora_B{i}").train(mode)

#     def eval(self):
#         nn.Linear.eval(self)
#         self.lora_route.eval()
#         for i in range(self.lora_num):
#             getattr(self, f"lora_A{i}").eval()
#             getattr(self, f"lora_B{i}").eval()

#     def cv_squared(self, x):
#         """The squared coefficient of variation of a sample.
#         Useful as a loss to encourage a positive distribution to be more uniform.
#         Epsilons added for numerical stability.
#         Returns 0 for an empty Tensor.
#         Args:
#         x: a `Tensor`.
#         Returns:
#         a `Scalar`.
#         """
#         eps = 1e-10
#         if x.shape[0] == 1:
#             return torch.tensor([0], device=x.device, dtype=x.dtype)[0]
#         return x.float().var() / (x.float().mean()**2 + eps)

#     def forward(self, x: torch.Tensor, task_types=None):

#         if self.disable_adapters:
#             result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
#             raise ImportError(":(") 
#         elif self.r > 0 and not self.merged:
#             result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            
#             if self.r > 0:
#                 route_weight = nn.functional.softmax(self.lora_route(x), dim=-1, dtype=torch.float32).to(result.dtype)

#                 for i in range(self.lora_num):
#                     result = result + torch.unsqueeze(route_weight[:,:,i], -1) * getattr(self, f"lora_B{i}")(getattr(self, f"lora_A{i}")(self.lora_dropout(x))) * self.scaling

#         blcls = torch.zeros(1)[0].to(result)
#         if task_types != None:
#             if self.blc_weight != 0:
#                 task_types = task_types.view(-1, 1)
#                 blcls = self.cv_squared((
#                     route_weight.sum(dim=(1)) * torch.where(
#                         torch.concat(
#                             ((task_types==1).repeat(1, self.lora_num//2), (task_types==0).repeat(1, self.lora_num//2)), dim=-1
#                             ), 1.0+self.blc_alpha, 1.0-self.blc_alpha
#                         )
#                     ).flatten()
#                 ) * self.blc_weight
#         return result
#         # return result, blcls
    
    
class MultiheadAttention(nn.MultiheadAttention,LoRALayer):
    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            lora_q: bool = True,
            lora_k: bool = True,
            lora_v: bool = True,
            lora_out: bool = True,
            **kwargs
    ):
        nn.MultiheadAttention.__init__(self,embed_dim,num_heads,**kwargs)
        LoRALayer.__init__(self,r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        self.lora_q = lora_q
        self.lora_k = lora_k
        self.lora_v = lora_v
        self.lora_o = lora_out
        self.embed_dim = embed_dim
        if r > 0:
            if lora_q:
                self.lora_A_q = nn.Parameter(self.in_proj_weight.new_zeros((r, embed_dim)))
                self.lora_B_q = nn.Parameter(self.in_proj_weight.new_zeros((embed_dim, r)))
            else:
                self.register_parameter('lora_A_q', None)
                self.register_parameter('lora_B_q', None)
            if lora_k:
                self.lora_A_k = nn.Parameter(self.in_proj_weight.new_zeros((r, embed_dim)))
                self.lora_B_k = nn.Parameter(self.in_proj_weight.new_zeros((embed_dim, r)))
            else:
                self.register_parameter('lora_A_k', None)
                self.register_parameter('lora_B_k', None)
            if lora_v:
                self.lora_A_v = nn.Parameter(self.in_proj_weight.new_zeros((r, embed_dim)))
                self.lora_B_v = nn.Parameter(self.in_proj_weight.new_zeros((embed_dim, r)))
            else:
                self.register_parameter('lora_A_q', None)
                self.register_parameter('lora_B_q', None)
            if lora_out:
                self.lora_A_o = nn.Parameter(self.in_proj_weight.new_zeros((r,embed_dim)))
                self.lora_B_o = nn.Parameter(self.in_proj_weight.new_zeros((embed_dim,r)))
            else:
                self.register_parameter('lora_A_o', None)
                self.register_parameter('lora_B_o', None)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.in_proj_weight.requires_grad = False
            self.out_proj.weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        # nn.MultiheadAttention._reset_parameters(self)
        if hasattr(self, 'lora_A_q'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_q)
        if hasattr(self, 'lora_A_k'):
            if self.lora_A_k is not None:
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A_k, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_k)
        if hasattr(self, 'lora_A_v'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_v)
        if hasattr(self, 'lora_A_o'):
            if self.lora_A_o is not None:
                nn.init.kaiming_uniform_(self.lora_A_o, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_o)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.MultiheadAttention.train(self, mode)
        ###################还需要再改#################
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    if self.lora_q:
                        self.in_proj_weight.data[:self.embed_dim] -= T(self.lora_B_q @ self.lora_A_q) * self.scaling
                    if self.lora_k:
                        self.in_proj_weight.data[self.embed_dim:self.embed_dim*2] -= T(self.lora_B_k @ self.lora_A_k) * self.scaling
                    if self.lora_v:
                        self.in_proj_weight.data[self.embed_dim*2:] -= T(self.lora_B_v @ self.lora_A_v) * self.scaling
                    if self.lora_o:
                        self.out_proj.weight.data -= T(self.lora_B_o @ self.lora_A_o) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    if self.lora_q:
                        self.in_proj_weight.data[:self.embed_dim] += T(self.lora_B_q @ self.lora_A_q) * self.scaling
                    if self.lora_k:
                        self.in_proj_weight.data[self.embed_dim:self.embed_dim*2] += T(self.lora_B_k @ self.lora_A_k) * self.scaling
                    if self.lora_v:
                        self.in_proj_weight.data[self.embed_dim*2:] += T(self.lora_B_v @ self.lora_A_v) * self.scaling
                    if self.lora_o:
                        self.out_proj.weight.data += T(self.lora_B_o @ self.lora_A_o) * self.scaling
                self.merged = True



    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
    # def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
    #             need_weights: bool = True, attn_mask: Optional[Tensor] = None,
    #             average_attn_weights: bool = True):
    #
    #
        is_batched = query.dim() == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif attn_mask is not None:
            why_not_fast_path = "attn_mask was not None"
        elif query.is_nested and key_padding_mask is not None:
            why_not_fast_path = "key_padding_mask is not supported with NestedTensor input"
        elif self.num_heads % 2 == 1:
            why_not_fast_path = "num_heads is odd"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x is None or x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x is not None and x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    key_padding_mask if key_padding_mask is not None else attn_mask,
                    need_weights,
                    average_attn_weights,
                    1 if key_padding_mask is not None else 0 if attn_mask is not None else None)

        # any_nested = query.is_nested or key.is_nested or value.is_nested
        # assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
        #                         f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward_lora(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,lora_A_q=self.lora_A_q,
                lora_B_q=self.lora_B_q, lora_A_k=self.lora_A_k, lora_B_k=self.lora_B_k,
                lora_A_v=self.lora_A_v,lora_B_v=self.lora_B_v,lora_A_o=self.lora_A_o,lora_B_o=self.lora_B_o,
                merged=self.merged,r=self.r,scaling=self.scaling)
            # attn_output, attn_output_weights = F.multi_head_attention_forward_lora(
            #     query, key, value, self.embed_dim, self.num_heads,
            #     self.in_proj_weight, self.in_proj_bias,
            #     self.bias_k, self.bias_v, self.add_zero_attn,
            #     self.dropout, self.out_proj.weight, self.out_proj.bias,
            #     training=self.training,
            #     key_padding_mask=key_padding_mask, need_weights=need_weights,
            #     attn_mask=attn_mask, average_attn_weights=average_attn_weights, lora_A_q=self.lora_A_q,
            #     lora_B_q=self.lora_B_q, lora_A_k=self.lora_A_k, lora_B_k=self.lora_B_k,
            #     lora_A_v=self.lora_A_v, lora_B_v=self.lora_B_v, merged=self.merged, r=self.r, scaling=self.scaling)

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class MultiheadAttention_query(nn.MultiheadAttention,LoRALayer):
    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            lora_q: bool = True,
            lora_k: bool = True,
            lora_v: bool = True,
            lora_out: bool = True,
            lora_k_query = True,
            lora_q_query = True,
            **kwargs
    ):
        nn.MultiheadAttention.__init__(self,embed_dim,num_heads,**kwargs)
        LoRALayer.__init__(self,r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        self.lora_q = lora_q
        self.lora_k = lora_k
        self.lora_v = lora_v
        self.lora_o = lora_out
        self.lora_k_query = lora_k_query
        self.embed_dim = embed_dim
        if r > 0:
            if lora_q:
                self.lora_A_q = nn.Parameter(self.in_proj_weight.new_zeros((r, embed_dim)))
                self.lora_B_q = nn.Parameter(self.in_proj_weight.new_zeros((embed_dim, r)))
            else:
                self.register_parameter('lora_A_q', None)
                self.register_parameter('lora_B_q', None)
            if lora_k:
                self.lora_A_k = nn.Parameter(self.in_proj_weight.new_zeros((r, embed_dim)))
                self.lora_B_k = nn.Parameter(self.in_proj_weight.new_zeros((embed_dim, r)))
            else:
                self.register_parameter('lora_A_k', None)
                self.register_parameter('lora_B_k', None)
            if lora_v:
                self.lora_A_v = nn.Parameter(self.in_proj_weight.new_zeros((r, embed_dim)))
                self.lora_B_v = nn.Parameter(self.in_proj_weight.new_zeros((embed_dim, r)))
            else:
                self.register_parameter('lora_A_q', None)
                self.register_parameter('lora_B_q', None)
            if lora_out:
                self.lora_A_o = nn.Parameter(self.in_proj_weight.new_zeros((r,embed_dim)))
                self.lora_B_o = nn.Parameter(self.in_proj_weight.new_zeros((embed_dim,r)))
            else:
                self.register_parameter('lora_A_o', None)
                self.register_parameter('lora_B_o', None)
            if lora_k_query:

                self.lora_A_k_query = nn.Parameter(self.in_proj_weight.new_zeros((r, embed_dim)))
                self.lora_B_k_query = nn.Parameter(self.in_proj_weight.new_zeros((embed_dim, r)))
            else:
                self.register_parameter('lora_A_k_query', None)
                self.register_parameter('lora_B_k_query', None)
            if lora_q_query:
                self.lora_A_q_query = nn.Parameter(self.in_proj_weight.new_zeros((r, embed_dim)))
                self.lora_B_q_query = nn.Parameter(self.in_proj_weight.new_zeros((embed_dim, r)))
            else:
                self.register_parameter('lora_A_q_uery', None)
                self.register_parameter('lora_B_q_query',None)

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.in_proj_weight.requires_grad = False
            self.out_proj.weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        # nn.MultiheadAttention._reset_parameters(self)
        if hasattr(self, 'lora_A_q'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_q)
        if hasattr(self, 'lora_A_k'):
            if self.lora_A_k is not None:
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A_k, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_k)
        if hasattr(self, 'lora_A_v'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_v)
        if hasattr(self, 'lora_A_o'):
            if self.lora_A_o is not None:
                nn.init.kaiming_uniform_(self.lora_A_o, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_o)
        if hasattr(self, 'lora_A_k_query'):
            if self.lora_A_o is not None:
                nn.init.kaiming_uniform_(self.lora_A_k_query, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_k_query)
        if hasattr(self, 'lora_A_q_query'):
            if self.lora_A_o is not None:
                nn.init.kaiming_uniform_(self.lora_A_q_query, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_q_query)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.MultiheadAttention.train(self, mode)
        ###################还需要再改#################
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    if self.lora_q:
                        self.in_proj_weight.data[:self.embed_dim] -= T(self.lora_B_q @ self.lora_A_q) * self.scaling
                    if self.lora_k:
                        self.in_proj_weight.data[self.embed_dim:self.embed_dim*2] -= T(self.lora_B_k @ self.lora_A_k) * self.scaling
                    if self.lora_v:
                        self.in_proj_weight.data[self.embed_dim*2:] -= T(self.lora_B_v @ self.lora_A_v) * self.scaling
                    if self.lora_o:
                        self.out_proj.weight.data -= T(self.lora_B_o @ self.lora_A_o) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    if self.lora_q:
                        self.in_proj_weight.data[:self.embed_dim] += T(self.lora_B_q @ self.lora_A_q) * self.scaling
                    if self.lora_k:
                        self.in_proj_weight.data[self.embed_dim:self.embed_dim*2] += T(self.lora_B_k @ self.lora_A_k) * self.scaling
                    if self.lora_v:
                        self.in_proj_weight.data[self.embed_dim*2:] += T(self.lora_B_v @ self.lora_A_v) * self.scaling
                    if self.lora_o:
                        self.out_proj.weight.data += T(self.lora_B_o @ self.lora_A_o) * self.scaling
                self.merged = True



    def forward(self, query: Tensor, key: Tensor, value: Tensor, query_token: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
    # def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
    #             need_weights: bool = True, attn_mask: Optional[Tensor] = None,
    #             average_attn_weights: bool = True):
    #
    #
        is_batched = query.dim() == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif attn_mask is not None:
            why_not_fast_path = "attn_mask was not None"
        elif query.is_nested and key_padding_mask is not None:
            why_not_fast_path = "key_padding_mask is not supported with NestedTensor input"
        elif self.num_heads % 2 == 1:
            why_not_fast_path = "num_heads is odd"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x is None or x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x is not None and x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    key_padding_mask if key_padding_mask is not None else attn_mask,
                    need_weights,
                    average_attn_weights,
                    1 if key_padding_mask is not None else 0 if attn_mask is not None else None)

        # any_nested = query.is_nested or key.is_nested or value.is_nested
        # assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
        #                         f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        _, bsz, _ = query.shape

        query_token = query_token.unsqueeze(0).expand(bsz,-1,-1)
        query_token = query_token.transpose(1,0)


        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_output_query = multi_head_attention_forward_lora_query(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,lora_A_q=self.lora_A_q,
                lora_B_q=self.lora_B_q, lora_A_k=self.lora_A_k, lora_B_k=self.lora_B_k,
                lora_A_v=self.lora_A_v,lora_B_v=self.lora_B_v,lora_A_o=self.lora_A_o,lora_B_o=self.lora_B_o,
                lora_A_k_query=self.lora_A_k_query,lora_B_k_query=self.lora_B_k_query,
                lora_A_q_query=self.lora_A_q_query,lora_B_q_query=self.lora_B_q_query,query_token=query_token,
                merged=self.merged,r=self.r,scaling=self.scaling)
            # attn_output, attn_output_weights = F.multi_head_attention_forward_lora(
            #     query, key, value, self.embed_dim, self.num_heads,
            #     self.in_proj_weight, self.in_proj_bias,
            #     self.bias_k, self.bias_v, self.add_zero_attn,
            #     self.dropout, self.out_proj.weight, self.out_proj.bias,
            #     training=self.training,
            #     key_padding_mask=key_padding_mask, need_weights=need_weights,
            #     attn_mask=attn_mask, average_attn_weights=average_attn_weights, lora_A_q=self.lora_A_q,
            #     lora_B_q=self.lora_B_q, lora_A_k=self.lora_A_k, lora_B_k=self.lora_B_k,
            #     lora_A_v=self.lora_A_v, lora_B_v=self.lora_B_v, merged=self.merged, r=self.r, scaling=self.scaling)

        if self.batch_first and is_batched:
            return attn_output, attn_output_query
        else:
            return attn_output, attn_output_query

def _in_projection_packed_lora(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
        ##### have been modified##############
        lora_A_q: Optional[Tensor] = None,
        lora_B_q: Optional[Tensor] = None,
        lora_A_k: Optional[Tensor] = None,
        lora_B_k: Optional[Tensor] = None,
        lora_A_v: Optional[Tensor] = None,
        lora_B_v: Optional[Tensor] = None,

        r: int = 0,
        merged=False,
        scaling=1.0
) -> List[Tensor]:

    E = q.size(-1)
    if k is v:
        if q is k:
            ############################Modify####################

            w_q, w_k, w_v = w.chunk(3)
            if b is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = b.chunk(3)
            if r > 0 and not merged:

                query = F.linear(q, w_q, b_q)
                key = F.linear(k, w_k, b_k)
                value = F.linear(v, w_v, b_v)
                if lora_A_q is not None:
                    query += (q @ lora_A_q.transpose(0, 1) @ lora_B_q.transpose(0, 1)) * scaling
                if lora_A_k is not None:
                    key += (k @ lora_A_k.transpose(0, 1) @ lora_B_k.transpose(0, 1)) * scaling
                if lora_A_v is not None:
                    value += (v @ lora_A_v.transpose(0, 1) @ lora_B_v.transpose(0, 1)) * scaling
                return query, key, value
            else:
                return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)
            # else:
            #     # self-attention
            #     return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            # Modify
            w_q, w_k, w_v = w.chunk(3)
            if b is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = b.chunk(3)
            if r > 0 and not merged:

                query = F.linear(q, w_q, b_q)
                key = F.linear(k, w_k, b_k)
                value = F.linear(v, w_v, b_v)
                if lora_A_q is not None:
                    query += (q @ lora_A_q.transpose(0, 1) @ lora_B_q.transpose(0, 1)) * scaling
                if lora_A_k is not None:
                    key += (k @ lora_A_k.transpose(0, 1) @ lora_B_k.transpose(0, 1)) * scaling
                if lora_A_v is not None:
                    value += (v @ lora_A_v.transpose(0, 1) @ lora_B_v.transpose(0, 1)) * scaling
                return query, key, value
            else:
                return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)
            
            # w_q, w_kv = w.split([E, E * 2])
            # if b is None:
            #     b_q = b_kv = None
            # else:
            #     b_q, b_kv = b.split([E, E * 2])
            # return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.

    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.

    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.

        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn
def multi_head_attention_forward_lora(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    lora_A_q: Optional[Tensor] = None,
    lora_B_q: Optional[Tensor] = None,
    lora_A_k: Optional[Tensor] = None,
    lora_B_k: Optional[Tensor] = None,
    lora_A_v: Optional[Tensor] = None,
    lora_B_v: Optional[Tensor] = None,
    lora_A_o: Optional[Tensor] = None,
    lora_B_o: Optional[Tensor] = None,
    merged: bool = False,
    r: int = 0,
    scaling: float = 1.0
) -> Tuple[Tensor, Optional[Tensor]]:

    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    # if has_torch_function(tens_ops):
    #     return handle_torch_function(
    #         multi_head_attention_forward,
    #         tens_ops,
    #         query,
    #         key,
    #         value,
    #         embed_dim_to_check,
    #         num_heads,
    #         in_proj_weight,
    #         in_proj_bias,
    #         bias_k,
    #         bias_v,
    #         add_zero_attn,
    #         dropout_p,
    #         out_proj_weight,
    #         out_proj_bias,
    #         training=training,
    #         key_padding_mask=key_padding_mask,
    #         need_weights=need_weights,
    #         attn_mask=attn_mask,
    #         use_separate_proj_weight=use_separate_proj_weight,
    #         q_proj_weight=q_proj_weight,
    #         k_proj_weight=k_proj_weight,
    #         v_proj_weight=v_proj_weight,
    #         static_k=static_k,
    #         static_v=static_v,
    #     )

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        q, k, v = _in_projection_packed_lora(query, key, value, in_proj_weight, in_proj_bias, lora_A_q=lora_A_q,
                                             lora_B_q=lora_B_q, lora_A_k=lora_A_k, lora_B_k=lora_B_k, lora_A_v=lora_A_v,
                                             lora_B_v=lora_B_v, r=r, merged=merged, scaling=scaling)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to floata_A_q is n
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output_t = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

    ####################### Modify add lora in Wo##############################
    # if r > 0 and not merged:
    #     attn_output = F.linear(attn_output_t, out_proj_weight, out_proj_bias)
    #     attn_output += (attn_output_t @ lora_A_o.transpose(0, 1) @ lora_B_o.transpose(0, 1)) * scaling
    # else:
    #     attn_output = F.linear(attn_output_t,out_proj_weight,out_proj_bias)
    if lora_A_o is not None:
        if r > 0 and not merged:
            attn_output = F.linear(attn_output_t, out_proj_weight, out_proj_bias)
            attn_output += (attn_output_t @ lora_A_o.transpose(0, 1) @ lora_B_o.transpose(0, 1)) * scaling
        else:
            attn_output = F.linear(attn_output_t, out_proj_weight, out_proj_bias)
    else:
        attn_output = F.linear(attn_output_t, out_proj_weight, out_proj_bias)


    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

def _in_projection_packed_lora_query(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
        ##### have been modified##############
        lora_A_q: Optional[Tensor] = None,
        lora_B_q: Optional[Tensor] = None,
        lora_A_k: Optional[Tensor] = None,
        lora_B_k: Optional[Tensor] = None,
        lora_A_v: Optional[Tensor] = None,
        lora_B_v: Optional[Tensor] = None,
        lora_A_k_query: Optional[Tensor] = None,
        lora_B_k_query: Optional[Tensor] = None,
        lora_A_q_query: Optional[Tensor] = None,
        lora_B_q_query: Optional[Tensor] = None,
        query_tokens: Optional[Tensor] = None,
        r: int = 0,
        merged=False,
        scaling=1.0
) -> List[Tensor]:

    E = q.size(-1)
    # if k is v:
    #     if q is k:
            ############################Modify####################

    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    if r > 0 and not merged:

        query = F.linear(q, w_q, b_q)
        key = F.linear(k, w_k, b_k)
        value = F.linear(v, w_v, b_v)
        key_query = key
        q_query = F.linear(query_tokens,w_q,b_q)
        if lora_A_q is not None:
            query += (q @ lora_A_q.transpose(0, 1) @ lora_B_q.transpose(0, 1)) * scaling
        if lora_A_k is not None:
            key += (k @ lora_A_k.transpose(0, 1) @ lora_B_k.transpose(0, 1)) * scaling
        if lora_A_v is not None:
            value += (v @ lora_A_v.transpose(0, 1) @ lora_B_v.transpose(0, 1)) * scaling
        if lora_A_k_query is not None:
            key_query += (k @ lora_A_k_query.transpose(0, 1) @ lora_B_k_query.transpose(0, 1)) * scaling
        if lora_A_q_query is not None:
            q_query += (query_tokens @ lora_A_q_query.transpose(0, 1) @ lora_B_q_query.transpose(0, 1)) * scaling
        return query, key, value, key_query, q_query
    else:
        w_k_query = w_k - (lora_B_k @ lora_A_k) * scaling
        w_k_query += (lora_B_k_query @ lora_A_k_query) * scaling
        w_q_query = w_q - (lora_B_q @ lora_A_q) * scaling
        w_q_query += (lora_B_q_query @ lora_A_q_query) * scaling
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v), F.linear(k, w_k_query, b_k), F.linear(query_tokens, w_q_query,b_q)
            # else:
            #     # self-attention
            #     return linear(q, w, b).chunk(3, dim=-1)
    #     else:
    #         # encoder-decoder attention
    #         w_q, w_kv = w.split([E, E * 2])
    #         if b is None:
    #             b_q = b_kv = None
    #         else:
    #             b_q, b_kv = b.split([E, E * 2])
    #         return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    # else:
    #     w_q, w_k, w_v = w.chunk(3)
    #     if b is None:
    #         b_q = b_k = b_v = None
    #     else:
    #         b_q, b_k, b_v = b.chunk(3)
    #     return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def multi_head_attention_forward_lora_query(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    lora_A_q: Optional[Tensor] = None,
    lora_B_q: Optional[Tensor] = None,
    lora_A_k: Optional[Tensor] = None,
    lora_B_k: Optional[Tensor] = None,
    lora_A_v: Optional[Tensor] = None,
    lora_B_v: Optional[Tensor] = None,
    lora_A_o: Optional[Tensor] = None,
    lora_B_o: Optional[Tensor] = None,
    lora_A_k_query: Optional[Tensor] = None,
    lora_B_k_query: Optional[Tensor] = None,
    lora_A_q_query: Optional[Tensor] = None,
    lora_B_q_query: Optional[Tensor] = None,
    query_token = None,
    merged: bool = False,
    r: int = 0,
    scaling: float = 1.0
) -> Tuple[Tensor, Optional[Tensor]]:

    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    query_len, _, _ = query_token.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        q, k, v, k_query, query_token = _in_projection_packed_lora_query(query, key, value, in_proj_weight, in_proj_bias, lora_A_q=lora_A_q,
                                             lora_B_q=lora_B_q, lora_A_k=lora_A_k, lora_B_k=lora_B_k, lora_A_v=lora_A_v,
                                             lora_B_v=lora_B_v, lora_A_k_query=lora_A_k_query, lora_B_k_query=lora_B_k_query,
                                            lora_A_q_query=lora_A_q_query,lora_B_q_query=lora_B_q_query,query_tokens=query_token,
                                                   r=r, merged=merged, scaling=scaling)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k_query = k_query.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to floata_A_q is n
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    query_token = query_token.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output_query, attn_output_weights_query = _scaled_dot_product_attention(query_token,k_query,v)

    attn_output_t = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output_query_t = attn_output_query.transpose(0, 1).contiguous().view(query_len,bsz,embed_dim)

    ####################### Modify add lora in Wo##############################
    # if r > 0 and not merged:
    #     attn_output = F.linear(attn_output_t, out_proj_weight, out_proj_bias)
    #     attn_output += (attn_output_t @ lora_A_o.transpose(0, 1) @ lora_B_o.transpose(0, 1)) * scaling
    # else:
    #     attn_output = F.linear(attn_output_t,out_proj_weight,out_proj_bias)
    if lora_A_o is not None:
        if r > 0 and not merged:
            attn_output = F.linear(attn_output_t, out_proj_weight, out_proj_bias)
            attn_output_query = F.linear(attn_output_query_t, out_proj_weight, out_proj_bias)

            attn_output += (attn_output_t @ lora_A_o.transpose(0, 1) @ lora_B_o.transpose(0, 1)) * scaling
            attn_output_query += (attn_output_query_t @ lora_A_o.transpose(0, 1) @ lora_B_o.transpose(0, 1)) * scaling
        else:
            attn_output = F.linear(attn_output_t, out_proj_weight, out_proj_bias)
            attn_output_query = F.linear(attn_output_query_t, out_proj_weight, out_proj_bias)
    else:
        attn_output = F.linear(attn_output_t, out_proj_weight, out_proj_bias)


    if need_weights:
        # average attention weights over heads
        # attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_query
    else:
        return attn_output, attn_output_query
    
    
    
    
    
    




####################################################################################################

class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0), 
            self.lora_B.unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True        

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result

class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)

class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)

class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)

# Can Extend to other ones like this

class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)
        
