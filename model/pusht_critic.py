import torch.nn as nn
import torch
import math


class Critic(nn.Module):
    def __init__(self, out_channels=2, dropout=0.1, config=None, *args, **kwargs) -> None:
        super(Critic, self).__init__()
        '''
        in_channel: actions:9 + sucker:2 = 11
        in_channel: actions:9 + sucker:2 + grip_feat:9 + side_feat:9 = 29
        '''
        
        self.config = config
        hidden_dim = config.critic.num_heads ** 2
        self.heads_dim = hidden_dim // config.critic.num_heads
        
        if config.olny_action_generate:
            self.fc1 = nn.Linear(2, hidden_dim, bias=True)
        else:
            self.fc1 = nn.Linear(6, hidden_dim, bias=True)
            self.grip_conv_1 = nn.Conv2d(3, 20, kernel_size=7, stride=5, padding=1, bias=True)
            self.grip_conv_2 = nn.Conv2d(20, config.per_image_with_signal_num, kernel_size=10, stride=7, padding=1, bias=True)
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.weight_fc = nn.Linear(28 - (20 - config.per_image_with_signal_num), out_channels)
        self.bias_fc = nn.Linear(8, out_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten(start_dim=2)
        self.batch_norm1 = nn.BatchNorm1d(config.per_image_with_signal_num)
        
        
    def forward(self, x):
        if self.config.olny_action_generate:
            # x include: [actions, sucker]
            x = torch.cat(x, dim=-1)
        else:
            # x include: [actions, sucker, grip_pred, side_pred]
            actions, obs_feat = x
            obs_feat = self.grip_conv_1(obs_feat.squeeze(1))
            obs_feat = self.grip_conv_2(obs_feat)
            obs_feat = self.flatten(obs_feat)
            x = torch.cat([actions, obs_feat], dim=-1)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        z1, z2 = torch.split(x, x.size(1)//2, dim=1)
        batchsize = z1.size(0)
        q = self.q(z1).view(batchsize, -1, self.heads_dim, self.config.critic.num_heads).transpose(1, 2)
        k = self.k(z1).view(batchsize, -1, self.heads_dim, self.config.critic.num_heads).transpose(1, 2)
        v = self.v(z2).view(batchsize, -1, self.heads_dim, self.config.critic.num_heads).transpose(1, 2)
        
        qk  = torch.matmul(q, k.transpose(-2, -1)) 
        scores = qk / math.sqrt(self.heads_dim)
        
        attn_w = self.softmax(scores)
        attn_w = self.dropout(attn_w)
        
        attnw_v = torch.matmul(attn_w, v)
        
        qk = qk.view(batchsize, self.config.per_image_with_signal_num, -1)
        attnw_v = attnw_v.view(batchsize, self.config.per_image_with_signal_num, -1)
        dp_out = torch.cat([qk, attnw_v], dim=-1)
        weights = self.weight_fc(dp_out)
        bias = self.bias_fc(attnw_v)
        
        return weights, bias
        