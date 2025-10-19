import torch.nn as nn
import torch
import math


class Critic(nn.Module):
    def __init__(
        self, out_channels=9, dropout=0.1, config=None, *args, **kwargs
    ) -> None:
        super(Critic, self).__init__()
        """
        in_channel: actions:9 + sucker:2 = 11
        in_channel: actions:9 + sucker:2 + grip_feat:9 + side_feat:9 = 29
        """

        self.config = config
        hidden_dim = config.critic.num_heads**2
        self.heads_dim = hidden_dim // config.critic.num_heads

        if config.olny_action_generate:
            self.fc1 = nn.Linear(11, hidden_dim, bias=True)
        else:
            self.fc1 = nn.Linear(29, hidden_dim, bias=True)
            self.grip_conv_1 = nn.Conv2d(
                3, 20, kernel_size=7, stride=5, padding=1, bias=True
            )
            self.grip_conv_2 = nn.Conv2d(
                20, 20, kernel_size=10, stride=7, padding=1, bias=True
            )
            self.side_conv_1 = nn.Conv2d(
                20, 20, kernel_size=7, stride=5, padding=1, bias=True
            )
            self.side_conv_2 = nn.Conv2d(
                20, 20, kernel_size=7, stride=5, padding=1, bias=True
            )
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.weight_fc = nn.Linear(216, out_channels)
        self.bias_fc = nn.Linear(16, out_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten(start_dim=2)
        self.batch_norm1 = nn.BatchNorm1d(config.horizon)

    def forward(self, x):
        if self.config.olny_action_generate:
            # x include: [actions, sucker]
            x = torch.cat(x, dim=-1)
        else:
            # x include: [actions, sucker, grip_pred, side_pred]
            actions, sucker, grip_pred, side_pred = x
            grip_feat = self.grip_conv_1(grip_pred)
            grip_feat = self.grip_conv_2(grip_feat)
            grip_feat = self.flatten(grip_feat)
            side_feat = self.grip_conv_1(side_pred)
            side_feat = self.grip_conv_2(side_feat)
            side_feat = self.flatten(side_feat)
            x = torch.cat([actions, sucker, grip_feat, side_feat], dim=-1)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        batchsize = x.size(0)
        q = (
            self.q(x)
            .view(batchsize, -1, self.config.critic.num_heads, self.heads_dim)
            .transpose(1, 2)
        )
        k = (
            self.k(x)
            .view(batchsize, -1, self.config.critic.num_heads, self.heads_dim)
            .transpose(1, 2)
        )
        v = (
            self.v(x)
            .view(batchsize, -1, self.config.critic.num_heads, self.heads_dim)
            .transpose(1, 2)
        )

        qk = torch.matmul(q, k.transpose(2, 3))
        scores = qk / math.sqrt(self.heads_dim)

        attn_w = self.softmax(scores)
        attnw_v = torch.matmul(attn_w, v).transpose(1, 2).reshape(batchsize, self.config.horizon, -1)
        dp_out = torch.cat([qk.transpose(1, 2).reshape(batchsize, self.config.horizon, -1), attnw_v], dim=-1)
        weights = self.weight_fc(dp_out)
        bias = self.bias_fc(attnw_v)

        return weights, bias
