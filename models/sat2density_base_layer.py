import torch
from torch import nn
from torch.nn import InstanceNorm2d, ReflectionPad2d
from torch.nn.utils import spectral_norm, weight_norm
from models.utils.eg3d_model_helper import Conv2dLayer

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none', inplace_nonlinearity=False,
                 order='CNA'):
        super(Conv2dBlock, self).__init__()

        # Initialize the convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups,
                              bias=bias, padding_mode=padding_mode)

        # Apply weight normalization if specified
        if weight_norm_type == 'spectral':
            self.conv = spectral_norm(self.conv)
        elif weight_norm_type == 'weight':
            self.conv = weight_norm(self.conv)
        # You would need to implement 'weight_demod' if required

        # Initialize the normalization layer based on activation_norm_type and arguments
        norm_layer = None
        if activation_norm_type == 'batch':
            norm_layer = nn.BatchNorm2d(out_channels, **(activation_norm_params or {}))
        elif activation_norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(out_channels, **(activation_norm_params or {}))
        elif activation_norm_type == 'layer':
            norm_layer = nn.LayerNorm(out_channels, **(activation_norm_params or {}))
        # Add other normalization types here as needed
        if norm_layer:
            self.add_module('norm', norm_layer)

        # Initialize the nonlinearity activation function
        nonlinearity_layer = None
        if nonlinearity == 'relu':
            nonlinearity_layer = nn.ReLU(inplace=inplace_nonlinearity)
        elif nonlinearity == 'leakyrelu':
            nonlinearity_layer = nn.LeakyReLU(0.2, inplace=inplace_nonlinearity)
        elif nonlinearity == 'prelu':
            nonlinearity_layer = nn.PReLU()
        elif nonlinearity == 'tanh':
            nonlinearity_layer = nn.Tanh()
        elif nonlinearity == 'sigmoid':
            nonlinearity_layer = nn.Sigmoid()
        elif nonlinearity == 'softmax':
            nonlinearity_layer = nn.Softmax(dim=1)
        # Add other activation functions here as needed
        if nonlinearity_layer:
            self.add_module('activation', nonlinearity_layer)

        # Add Gaussian noise module if apply_noise is True
        # if apply_noise:
        #     self.noise = GaussianNoise(out_channels)
        
        # Store the order of operation as a tuple based on the provided order string
        self.operations_order = tuple(order)

    def forward(self, x):
        # Apply operations according to the specified order
        for operation in self.operations_order:
            if operation == 'C':
                x = self.conv(x)
            elif operation == 'N' and hasattr(self, 'norm'):
                x = self.norm(x)
            elif operation == 'A' and hasattr(self, 'activation'):
                x = self.activation(x)
            # elif operation == "G" and hasattr(self, 'noise'):
            #     x = self.noise(x)
        return x

# class GaussianNoise(nn.Module):
#     """Gaussian noise regularizer.

#     Args:
#         std (float): Standard deviation of the Gaussian noise.
#     """
#     def __init__(self, std):
#         super(GaussianNoise, self).__init__()
#         self.std = std

#     def forward(self, x):
#         if self.training:
#             return x + torch.randn_like(x) * self.std
#         return x

    



class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 weight_norm_type=None, weight_norm_params=None,
                 activation_norm_type=None, activation_norm_params=None,
                 nonlinearity=None, inplace_nonlinearity=False,
                 order='CNA'):
        super(LinearBlock, self).__init__()

        layers = []

        # Define linear layer
        linear_layer = nn.Linear(in_features, out_features, bias=bias)

        # Apply weight normalization if specified
        if weight_norm_type == 'spectral':
            linear_layer = spectral_norm(linear_layer)
        elif weight_norm_type == 'weight':
            linear_layer = weight_norm(linear_layer, **(weight_norm_params or {}))
        # For 'weight_demod', implement the corresponding normalization if needed

        # Add linear layer according to the order
        if 'C' in order:
            layers.append(linear_layer)

        # Define normalization layer if specified
        norm_layer = None
        if activation_norm_type == 'instance':
            norm_layer = nn.InstanceNorm1d(out_features, **(activation_norm_params or {}))
        elif activation_norm_type == 'batch':
            norm_layer = nn.BatchNorm1d(out_features, **(activation_norm_params or {}))
        # ... Implement other normalization types as needed ...

        # Add normalization layer according to the order
        if norm_layer and 'N' in order:
            layers.append(norm_layer)

        # Add Gaussian noise layer if specified

        # Define nonlinearity layer if specified
        nonlinearity_layer = None
        if nonlinearity == 'relu':
            nonlinearity_layer = nn.ReLU(inplace=inplace_nonlinearity)
        elif nonlinearity == 'leakyrelu':
            nonlinearity_layer = nn.LeakyReLU(0.2, inplace=inplace_nonlinearity)
        elif not nonlinearity or nonlinearity == 'none':
            pass
        elif nonlinearity == 'tanh':
            nonlinearity_layer = nn.Tanh()
        else:
            raise NotImplementedError(f'nonlinearity {nonlinearity} not implemented')
        # ... Implement other nonlinearity types as needed ...

        # Add nonlinearity layer according to the order
        if nonlinearity_layer and 'A' in order:
            layers.append(nonlinearity_layer)

        # Combine layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)

        # Apply noise if necessary
        # if self.apply_noise:
        #     noise = torch.randn(x.size(0), 1, device=x.device) * self.noise_strength
        #     x = x + noise
        
        return x



class Res2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, hidden_channels_equal_out_channels=False,
                 order='CC'):
        super(Res2dBlock, self).__init__()

        if hidden_channels_equal_out_channels:
            hidden_channels = out_channels
        else:
            hidden_channels = min(in_channels, out_channels)


        # Build the residual link based on the specified order
        self.residual = self.build_residual(in_channels, hidden_channels, out_channels,
                                            kernel_size, order)

    def build_residual(self, in_channels, hidden_channels, out_channels,
                       kernel_size, order):
        layers = []
        i=0
        conv = Conv2dLayer(in_channels if i == 0 else hidden_channels,
                            out_channels if i == len(order) - 1 else hidden_channels,
                            kernel_size, activation='lrelu')
        layers.append(conv)
        i=1
        conv = Conv2dLayer(in_channels if i == 0 else hidden_channels,
                                out_channels if i == len(order) - 1 else hidden_channels,
                                kernel_size)
        layers.append(conv)
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = self.residual(x)
        return x + residual






if __name__ == '__main__':
    pass

