import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Normal
from torch.nn.modules.utils import _triple, _pair


class DiffDecoder(nn.Module):
    def __init__(self, xlstm_cfg, in_channels, out_channels) -> None:
        super(DiffDecoder, self).__init__()
        self.tconv_1 = deconv(in_channels, 1, stride=(1, 2, 2))
        self.tconv_2 = deconv(10, 10, stride=(1, 2, 2))
        self.tconv_3 = deconv(10, 5, stride=(1, 2, 2))
        self.tconv_4 = deconv(5, 1, stride=(1, 2, 2))
        self.conv_1 = torch.nn.Conv3d(65, 10, (1, 3, 3), (1, 2, 2), (0, 1, 1))
        self.conv_2 = torch.nn.Conv3d(74, 10, (1, 3, 3), (1, 2, 2), (0, 1, 1))

    def forward(self, x:torch.Tensor, layers_out:list):
        layer_out1, layer_out2 = layers_out
        x = x.clone()
        x = self.tconv_1(x)
        temp_ = repeat_like(layer_out2, x, remain_dim=1)
        x = torch.cat([x, temp_], dim=1)
        x = self.conv_1(x)
        x = self.tconv_2(x)
        temp_ = repeat_like(layer_out1, x, remain_dim=1)
        x = torch.cat([x, temp_], dim=1)
        x = self.conv_2(x)
        x = self.tconv_3(x)
        x = self.tconv_4(x)
        
        return x

class CausalConv1D(nn.Conv1d):
    """A causal 1D convolution.
  """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, initailize_weights=False, device='cuda'
    ):
        self.__padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            bias=bias,
        )
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True).cuda()
        if initailize_weights:
            nn.init.kaiming_normal_(self.weight.data)
            if self.bias is not None:
                self.bias.data.zero_()
        
    def forward(self, x):
        res = super(CausalConv1D, self).forward(x)
        if self.__padding != 0:
            return self.leaky_relu(res[:, :, : -self.__padding]), res[:, :, : -self.__padding]
        return self.leaky_relu(res), res



class CausalConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, time_dim=-2, initailize_weights=True):
        '''
        input:Tensor[batch, seq, time_seq, signal]
        '''
        super(CausalConv2D, self).__init__()
        # 指定时间维度和填充量
        self.time_dim = time_dim
        if isinstance(kernel_size, tuple):
            assert kernel_size[time_dim] % 2 == 1, "Kernel size along time dimension must be odd for causality"
            self.pad = dilation * (kernel_size[time_dim] - 1)
        else:
            assert kernel_size % 2 == 1, "Kernel size must be odd for causality"
            self.pad = dilation * (kernel_size - 1)
        
        # 卷积层
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=0, dilation=dilation, bias=bias
        )
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, x):
        # 确保在指定时间维度上进行因果填充
        if self.time_dim == -2:  # 常用于二维卷积时时间维度是倒数第二维
            x = F.pad(x, (0, 0, 0, self.pad))  # 在宽度（时间）上填充右侧
        elif self.time_dim == -1:  # 如果时间维度是最后一维
            x = F.pad(x, (self.pad, 0, 0, 0))  # 在最后一维上进行因果填充
        else:
            raise ValueError("Unsupported time dimension index for padding")
        
        return self.leaky_relu(self.conv(x)), self.conv(x)


       
def conv3d_lerelu_maxpl(in_channels, out_channels, kernel_size=(1, 1, 1), padding=(1, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1), bias=True):
    '''
    return: seq[conv3d, leaky_relu, maxpooling]
    '''
    in_channels = in_channels
    out_channels = out_channels
    padding = padding
    stride = stride
    dilation = dilation
    bias = bias
    kernel_size = kernel_size
    # compute new filter size after dilation
    # and necessary padding for `same` output size
    dilated_kernel_size = (kernel_size[-1] - 1) * (dilation[-1] - 1) + kernel_size[-1]
    same_padding = (dilated_kernel_size - 1) // 2

    return nn.Sequential(
        nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(padding[1], same_padding, same_padding),
            dilation=dilation,
            bias=bias,
        ),
        nn.BatchNorm3d(out_channels),
        nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(0.1, inplace=True)        
    )
        
        
def gaussian_parameters(h, dim=-1):

    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def sample_gaussian(m, v, device, training_phase=None, z_attention=False, SelfAttentions=None):
    if training_phase == 'generate':
        m = torch.zeros_like(m)  #zero
        v = torch.ones_like(v)   #one
    # reparameterize
    epsilon = Normal(0, 1).rsample(m.size())
    z = m + torch.sqrt(v) * epsilon.to(device)  

    if z_attention and SelfAttentions is not None:
        #### add self-attentions to mu, var
        att_z = SelfAttentions(z)
        z_mix = torch.add(att_z, z)
    else:
        z_mix = z

    return z_mix

def downsampling(input, target):
    '''
    给一个张量，返回与目标张量相同形状的张量
    
    Param:
    input: Tensor
    target: Tensor or List,  
            if List --> [c, h, w]
    return: Tensor
    '''
    if type(target) is list:
        shape = target
    elif type(target) is torch.Tensor:
        shape = target.shape[2:]
        
    return  nn.functional.interpolate(input, size=shape, mode="trilinear", align_corners=False)

def crop_like(input, target):
    i_shape = input.shape
    t_shape = target.shape
    if input.size()[1:] == target.size()[1:]:
        return input
    elif len(target.shape) == 4:
        return input[:, 
                     (i_shape[1] - t_shape[1]) // 2: t_shape[1] + (i_shape[1] - t_shape[1]) // 2, 
                     (i_shape[2] - t_shape[2]) // 2: t_shape[2] + (i_shape[2] - t_shape[2]) // 2, 
                     (i_shape[3] - t_shape[3]) // 2: t_shape[3] + (i_shape[3] - t_shape[3]) // 2]
    elif len(target.shape) == 5:
        return input[:, 
                     (i_shape[1] - t_shape[1]) // 2: t_shape[1] + (i_shape[1] - t_shape[1]) // 2, 
                     (i_shape[2] - t_shape[2]) // 2: t_shape[2] + (i_shape[2] - t_shape[2]) // 2,
                     (i_shape[3] - t_shape[3]) // 2: t_shape[3] + (i_shape[3] - t_shape[3]) // 2,
                     (i_shape[4] - t_shape[4]) // 2: t_shape[4] + (i_shape[4] - t_shape[4]) // 2]
        
def repeat_like(input, target, remain_dim=None) -> torch.Tensor:
    assert len(input.shape) == len(target.shape), "the dim number of input and target must be eq."
    if len(input.shape) == 3:
        input = input.repeat(
            1, 
            1 if remain_dim == 1 else target.shape[1] // input.shape[1] + 1,
            1 if remain_dim == 2 else target.shape[2] // input.shape[2] + 1,
        )[:, 
          :target.shape[1] if remain_dim != 1 else input.shape[1], 
          :target.shape[2] if remain_dim != 2 else input.shape[2]]
    elif len(input.shape) == 4:
        input = input.repeat(
            1, 
            1 if remain_dim == 1 else target.shape[1] // input.shape[1] + 1,
            1 if remain_dim == 2 else target.shape[2] // input.shape[2] + 1,
            1 if remain_dim == 3 else target.shape[3] // input.shape[3] + 1,
        )[:, 
          :target.shape[1] if remain_dim != 1 else input.shape[1], 
          :target.shape[2] if remain_dim != 2 else input.shape[2], 
          :target.shape[3] if remain_dim != 3 else input.shape[3]]
    elif len(input.shape) == 5:
        input = input.repeat(
            1, 
            1 if remain_dim == 1 else target.shape[1] // input.shape[1] + 1,
            1 if remain_dim == 2 else target.shape[2] // input.shape[2] + 1,
            1 if remain_dim == 3 else target.shape[3] // input.shape[3] + 1,
            1 if remain_dim == 4 else target.shape[4] // input.shape[4] + 1,
        )[:, 
          :target.shape[1] if remain_dim != 1 else input.shape[1], 
          :target.shape[2] if remain_dim != 2 else input.shape[2], 
          :target.shape[3] if remain_dim != 3 else input.shape[3], 
          :target.shape[4] if remain_dim != 4 else input.shape[4]]

    return input

def deconv(in_planes, out_planes, kernel_size=(1, 3, 3), stride=(1, 1, 1)):
    return nn.Sequential(
        nn.ConvTranspose3d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=False
        ),
        nn.BatchNorm3d(out_planes),
        nn.LeakyReLU(0.1, inplace=True),
    )


def predict_flow(in_planes, out_planes=1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


def conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), padding=(1, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1), bias=True):
    """`same` convolution with LeakyReLU, i.e. output shape equals input shape.输出形状等于输入形状
  Args:
    in_planes (int): The number of input feature maps.
    out_planes (int): The number of output feature maps.
    kernel_size (int): The filter size.
    dilation (int): The filter dilation factor.
    stride (int): The filter stride.
  """
    # compute new filter size after dilation
    # and necessary padding for `same` output size
    dilated_kernel_size = (kernel_size[-1] - 1) * (dilation[-1] - 1) + kernel_size[-1]
    same_padding = (dilated_kernel_size - 1) // 2

    return nn.Sequential(
        nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(dilated_kernel_size, ) * 3,
            stride=stride,
            padding=(same_padding, same_padding, same_padding),
            dilation=dilation,
            bias=bias,
        ),
        nn.BatchNorm3d(out_channels),
        nn.LeakyReLU(0.1, inplace=True),
    )
def soft_pool2d(x, kernel_size=2, stride=None, force_inplace=False):

    kernel_size = _pair(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _pair(stride)
    # Get input sizes
    _, c, h, w = x.size()
    # Create exponential mask (should be similar to max-like pooling)
    e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
    e_x = torch.clamp(e_x , float(0), float('inf'))
    # Apply mask to input and pool and calculate the exponential sum
    # Tensor: [b x c x d] -> [b x c x d']
    x = F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))
    return torch.clamp(x , float(0), float('inf'))

def soft_pool3d(x, kernel_size=2, stride=None, force_inplace=False):
    if x.shape[2] < kernel_size:
        kernel_size = (1, 2, 2)
    else:
        kernel_size = _triple(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _triple(stride)
    # Get input sizes
    _, c, d, h, w = x.size()
    # Create exponential mask (should be similar to max-like pooling)
    e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
    e_x = torch.clamp(e_x , float(0), float('inf'))
    # Apply mask to input and pool and calculate the exponential sum
    # Tensor: [b x c x d x h x w] -> [b x c x d' x h' x w']
    x = F.avg_pool3d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool3d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))
    return torch.clamp(x , float(0), float('inf'))


def init_weights(modules, device='cuda'):
    """
    Weight initialization from original SensorFusion Code
    """
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()