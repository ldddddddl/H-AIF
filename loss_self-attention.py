import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionLoss(nn.Module):
    def __init__(self, num_losses=5, embedding_dim=8):
        super(SelfAttentionLoss, self).__init__()
        self.num_losses = num_losses
        self.embedding_dim = embedding_dim
        
        # 为每个损失项创建嵌入层，以将其投影到固定的嵌入维度
        self.embedding = nn.Linear(1, embedding_dim)
        
        # 自注意力层（Query, Key, Value 共享同一线性层）
        self.attn_query = nn.Linear(embedding_dim, embedding_dim)
        self.attn_key = nn.Linear(embedding_dim, embedding_dim)
        self.attn_value = nn.Linear(embedding_dim, embedding_dim)
        
        # 用于将加权后的结果映射回标量形式
        self.output_proj = nn.Linear(embedding_dim, 1)
    
    def forward(self, losses):
        """
        参数:
        losses (torch.Tensor): 每个损失项构成的 tensor，形状为 [num_losses]
        
        返回:
        torch.Tensor: 加权后的总损失，标量
        """
        # 将损失项扩展为 [num_losses, 1] 形状
        losses = losses.view(self.num_losses, 1)
        
        # 嵌入每个损失项为 [num_losses, embedding_dim]
        embedded_losses = self.embedding(losses)
        
        # 计算 Query, Key, Value
        query = self.attn_query(embedded_losses)
        key = self.attn_key(embedded_losses)
        value = self.attn_value(embedded_losses)
        
        # 计算注意力分数并应用 softmax
        attn_scores = torch.matmul(query, key.transpose(0, 1)) / (self.embedding_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 根据注意力权重加权值向量
        weighted_values = torch.matmul(attn_weights, value)
        
        # 将加权后的结果投影回标量，并取均值作为最终损失
        weighted_losses = self.output_proj(weighted_values).squeeze(-1)
        total_loss = weighted_losses.mean()
        
        return total_loss
