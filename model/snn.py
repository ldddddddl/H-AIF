import torch.nn as nn
import torch

from collections import OrderedDict
from torch.distributions import Normal
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig
from .decoders import XLstmStack
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer, functional, rnn

        
class DownConvSNN(nn.Module):
    def __init__(self, embedding_dim, enc_out_dim, use_cupy=False, config=None):
        super(DownConvSNN, self).__init__()
        self.T = config.snn.T

        self.config = config
        self.conv_seq = nn.Sequential(
            layer.Conv3d(embedding_dim, 
                        enc_out_dim,
                        # enc_out_dim,
                        kernel_size=(3, 3, 3),
                        stride=(2, 2, 2),
                        padding=(1, 1, 1)
                                    ),
            layer.BatchNorm3d(enc_out_dim),
            # neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool3d((1, 2, 2)),
            NonSpikingLIFNode(tau=2.0, step_mode='s')
            
            # layer.Conv3d(embedding_dim * 2, 
            #             enc_out_dim,
            #             kernel_size=(3, 3, 3),
            #             stride=(1, 1, 1),
            #             padding=(1, 1, 1)),
            # layer.BatchNorm3d(enc_out_dim),
            # neuron.IFNode(surrogate_function=surrogate.ATan()),
            # layer.MaxPool3d((1, 2, 2))
            )
        functional.set_step_mode(self, step_mode='m')
        # functional.set_backend(self, backend='cupy')
    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x = self.conv_seq(x)  # 5, 8, 70, 1, 13, 13  # [5, 8, 70, 1, 6, 6]
        fr = x.mean(0)  
        # fr = fr.reshape(*self.out_shape)
        return fr


class CSNN(nn.Module):
    def __init__(self, T: int, channels: int, out_channel: int, use_cupy=False, config=None):
        super(CSNN, self).__init__()
        self.T = T
        self.config = config
        self.out_shape = [config.batchsize, config.past_img_num*config.future_img_num*2, -1]
        if config.snn.is_use_spike_dataset:
            self.conv = nn.Sequential(
                layer.Conv2d(9, 11, kernel_size=5, stride=2, padding=1, bias=False),
                layer.BatchNorm2d(11),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
                layer.MaxPool2d(4, 4),  # 14 * 14
            )
        else:
            self.conv = nn.Sequential(
                layer.Conv2d(9, channels * 12, kernel_size=3, padding=1, bias=False),
                layer.BatchNorm2d(channels * 12),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
                layer.MaxPool2d(2, 2),  # 14 * 14

                layer.Conv2d(channels * 12, channels * 12, kernel_size=3, padding=1, bias=False),
                layer.BatchNorm2d(channels * 12),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
                layer.MaxPool2d(2, 2),  # 7 * 7

                layer.Conv2d(channels * 12, 11, kernel_size=3, padding=1, bias=False),
                layer.BatchNorm2d(11),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
                layer.MaxPool2d(2, 2), 
            )

        if config.snn.is_use_spike_dataset:
            self.fc1 = nn.Sequential(
                layer.Linear(169, 50, bias=False),
                layer.BatchNorm1d(11),
                NonSpikingLIFNode(tau=2.0, step_mode='s'),
            )
        else:
            self.fc1 = nn.Sequential(
                layer.Linear(196, 50, bias=False),
                layer.BatchNorm1d(11),
                NonSpikingLIFNode(tau=2.0, step_mode='s'),
            )
        self.snn_flatten = layer.Flatten(start_dim=2)
        functional.set_step_mode(self, step_mode='m')
        # functional.set_backend(self, backend='cupy')

        
        
    def forward(self, x: torch.Tensor, error: torch.Tensor):
        # x.shape = [N, C, H, W]
        x = torch.cat([x, error], dim=1)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x = self.conv(x)
        x = self.snn_flatten(x)
        x = self.fc1(x)
        fr = x.mean(0)
        # fr = fr.reshape(*self.out_shape)
        fr = fr.transpose(1, -1)

        return fr
    
    def spiking_encoder(self):
        return self.conv_fc[0:3]

class SNNActDecoder(nn.Module):
    """ Learnable Population Coding Decoder """
    def __init__(self, act_dim, hidden_dim, std=0.0, *args, **kwargs):
        super().__init__()
        self.act_dim = act_dim
        self.config = kwargs["config"]
        device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.act_xlstm_layers = XLstmStack(self.config.snn.snn_dnc, device=device)
        self.norm = nn.BatchNorm1d(self.config.per_image_with_signal_num)
        # self.spike_lstm = rnn.SpikingLSTM(152, hidden_dim, self.config.snn.spike_lstm_layers)
        self.actor = nn.Sequential(
            layer.Linear(672, hidden_dim),
            layer.BatchNorm1d(self.config.per_image_with_signal_num),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            
            layer.Linear(hidden_dim, act_dim),
            layer.BatchNorm1d(self.config.per_image_with_signal_num), 
            NonSpikingLIFNode(tau=2.0, step_mode='s'),
        )
        self.log_std = nn.Parameter(torch.ones(self.config.per_image_with_signal_num, act_dim) * std)
        functional.set_step_mode(self, step_mode='m')
        # functional.set_backend(self, backend='cupy')
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor):
        x = self.act_xlstm_layers.xlstm(x.unsqueeze(-1).unsqueeze(-1))
        x = x.reshape(self.config.batchsize, self.config.per_image_with_signal_num, -1)
        x = self.norm(x)
        if self.config.snn.is_sampling:
            for t in range(self.config.snn.T):
                self.actor(x)
                
            mu  = self.actor[-1].v * 2
            std   = self.log_std.exp().expand_as(mu)
            dist  = Normal(mu, std)
            action = dist.sample().clamp(-2.0, 2.0)
        else:
            x = x.unsqueeze(0).repeat(self.config.snn.T, 1, 1, 1)
            x = self.actor(x)
            x = x.mean(0)
            action = self.tanh(x) * 2.0
            mu = None
            std = None
        return  mu, std, action

        

class NonSpikingLIFNode(neuron.LIFNode):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
        def single_step_forward(self, x: torch.Tensor):
            self.v_float_to_tensor(x)

            if self.training:
                self.v = self.neuronal_charge(x)
            else:
                if self.v_reset is None:
                    if self.decay_input:
                        self.v = self.neuronal_charge_decay_input_reset0(x, self.v, self.tau)
                    else:
                        self.v = self.neuronal_charge_no_decay_input_reset0(x, self.v, self.tau)
                    
                else:
                    if self.decay_input:
                        self.v = self.neuronal_charge_decay_input(x, self.v, self.v_reset, self.tau)
                    else:
                        self.v = self.neuronal_charge_no_decay_input(x, self.v, self.v_reset, self.tau)
            return self.v
        
        


class ActionSpikeEncode(nn.Module):
    def __init__(self, in_dim, act_dim, hidden_dim, T, config=None):
        super().__init__()
        self.in_dim = in_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.T = T
        self.config = config
        # Define Layers
        self.spike_conv1 = nn.Sequential(
            layer.Conv1d(in_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            layer.BatchNorm1d(hidden_dim),
            neuron.ParametricLIFNode(init_tau=2.0),
            layer.MaxPool1d(2),
            
            layer.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            layer.BatchNorm1d(hidden_dim),
            neuron.ParametricLIFNode(init_tau=2.0),
            layer.MaxPool1d(2),
        )
        self.spike_fc = nn.Sequential(
            layer.Linear(hidden_dim * 2, in_dim * act_dim),
            neuron.ParametricLIFNode(init_tau=2.0),
        )
        self.spike_flatten = layer.Flatten()
        
        functional.set_step_mode(self, step_mode='m')
        functional.set_backend(self, backend='torch')
        # functional.set_backend(self, backend='cupy')

    def forward(self, x:torch.Tensor):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1)
        x = self.spike_conv1(x)
        x = self.spike_flatten(x)
        x = self.spike_fc(x)
        x = x.mean(0)
        x = x.reshape(self.config.batchsize, self.in_dim, -1)
        return x 


    
class RGB2Spike(nn.Module):
    def __init__(self, in_channels: int, out_channel: int, use_cupy=False, config=None):
        super(RGB2Spike, self).__init__()
        self.T = config.snn.T
        self.config = config
        self.out_shape = [config.batchsize, config.past_img_num*config.future_img_num*2, -1]
        self.conv1 = nn.Sequential(
            layer.Conv3d(in_channels, out_channel, kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=(0, 1, 1)),
            layer.BatchNorm3d(out_channel),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool3d((1, 2, 2)),  # 14 * 14
        )
        functional.set_step_mode(self, step_mode='m')
        # functional.set_backend(self, backend='cupy')
    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x = self.conv1(x)
        fr = x.mean(0)
        # fr = fr.reshape(*self.out_shape)
        return fr



class SNNImagesGenerate(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SNNImagesGenerate, self).__init__(*args, **kwargs)
        self.deconv = snndeconv3d()


def snndeconv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=(1, 1, 1)):
    return nn.Sequential(
        layer.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=False),
        layer.BatchNorm3d(out_planes),
        NonSpikingLIFNode(tau=2.0),
    )  