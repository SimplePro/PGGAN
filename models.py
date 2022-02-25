# models file was referred to "https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/ProGAN".

import torch
import torch.nn as nn
import torch.nn.functional as F

channel_list = [128, 128, 128, 128, 64, 32, 16]



class WSConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5

        # self.bias.shape: (out_channels)
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    
    def forward(self, x):
        out = self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)
        return out



class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8


    def forward(self, x):
        # (batch_size, C, H, W) / (batch_size, 1, H, W)
        out = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
        return out



class UpDownSampling(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = size


    def forward(self, x):
        out = F.interpolate(x, scale_factor=self.size, mode="nearest")
        return out



class GeneratorConvBlock(nn.Module):

    def __init__(self, step, scale_size):
        super().__init__()
        self.up_sampling = UpDownSampling(size=scale_size)

        # (C_(step-1), H, W) -> (C_step, H, W)
        self.conv1 = WSConv2d(in_channels=channel_list[step-1], out_channels=channel_list[step], kernel_size=3, stride=1, padding=1)

        # (C_step, H, W) -> (C_step, H, W)
        self.conv2 = WSConv2d(in_channels=channel_list[step], out_channels=channel_list[step], kernel_size=3, stride=1, padding=1)

        self.leakyrelu = nn.LeakyReLU(0.2)

        self.pn = PixelNorm()


    def forward(self, x):
        self.scaled = self.up_sampling(x)
        
        out = self.conv1(self.scaled)
        out = self.leakyrelu(out)
        out = self.pn(out)

        out = self.conv2(out)
        out = self.leakyrelu(out)
        out = self.pn(out)

        return out



class Generator(nn.Module):

    def __init__(self, steps):
        super().__init__()

        self.steps = steps

        self.init = nn.Sequential(
            PixelNorm(),

            # (z_dim, 1, 1) -> (C_0, 4, 4)
            nn.ConvTranspose2d(in_channels=channel_list[0], out_channels=channel_list[0], kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),

            # (C_0, 4, 4) -> (C_0, 4, 4)
            WSConv2d(in_channels=channel_list[0], out_channels=channel_list[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

        self.init_torgb = WSConv2d(in_channels=channel_list[0], out_channels=3, kernel_size=1, stride=1, padding=0)

        self.prog_blocks = nn.ModuleList([self.init])
        self.torgb_layers = nn.ModuleList([self.init_torgb])
        
        # append blocks that are not init block.
        for step in range(1, self.steps+1):
            self.prog_blocks.append(GeneratorConvBlock(step, scale_size=2))
            self.torgb_layers.append(WSConv2d(in_channels=channel_list[step], out_channels=3, kernel_size=1, stride=1, padding=0))


    def fade_in(self, alpha, upsampling, generated):
        return alpha * generated + (1 - alpha) * upsampling


    def forward(self, x, alpha):
        out = self.prog_blocks[0](x)

        if self.steps == 0:
            return self.torgb_layers[0](out)

        for step in range(1, self.steps+1):
            out = self.prog_blocks[step](out)

        upsampling = self.torgb_layers[step-1](self.prog_blocks[step].scaled)
        generated = self.torgb_layers[step](out)

        return self.fade_in(alpha, upsampling, generated)



class DiscriminatorConvBlock(nn.Module):

    def __init__(self, step):
        super().__init__()

        # (C_step, H, W) -> (C_step, H, W)
        self.conv1 = WSConv2d(in_channels=channel_list[step], out_channels=channel_list[step], kernel_size=3, stride=1, padding=1)

        # (C_step, H, W) -> (C_(step-1), H, W)
        self.conv2 = WSConv2d(in_channels=channel_list[step], out_channels=channel_list[step-1], kernel_size=3, stride=1, padding=1)

        # (C_(step-1), H/2, W/2)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.leakyrelu = nn.LeakyReLU(0.2)

    
    def forward(self, x):
        out = self.conv1(x)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.leakyrelu(out)

        out = self.downsample(out)

        return out



class MinibatchStd(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, x):
        # mean of minibatch's std
        # (1) -> (batch_size, 1, H, W)
        batch_statistics = (torch.std(x, dim=0).mean().repeat(x.size(0), 1, x.size(2), x.size(3)))

        # (batch_size, C, H, W) -> (batch_size, C+1, H, W)
        return torch.cat((x, batch_statistics), dim=1)



class Discriminator(nn.Module):

    def __init__(self, steps):
        super().__init__()
        # progressive growing blocks
        self.prog_blocks = nn.ModuleList([])

        # fromrgb layers
        self.fromrgb_layers = nn.ModuleList([])

        self.leakyrelu = nn.LeakyReLU(0.2)

        self.steps = steps
        
        # append blocks that are not final block.
        for step in range(steps, 0, -1):
            self.prog_blocks.append(DiscriminatorConvBlock(step))
            self.fromrgb_layers.append(WSConv2d(in_channels=3, out_channels=channel_list[step], kernel_size=1, stride=1, padding=0))

        # append final block
        self.fromrgb_layers.append(
            WSConv2d(in_channels=3, out_channels=channel_list[0], kernel_size=1, stride=1, padding=0)
        )

        # append final block
        self.prog_blocks.append(
            nn.Sequential(
                MinibatchStd(),
                WSConv2d(in_channels=channel_list[0]+1, out_channels=channel_list[0], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                WSConv2d(in_channels=channel_list[0], out_channels=channel_list[0], kernel_size=4, stride=1, padding=0),
                nn.LeakyReLU(0.2),
                WSConv2d(in_channels=channel_list[0], out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
        )

        # downsample
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    
    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled

    
    def forward(self, x, alpha):
        # (3, H, W) -> (C, H, W)
        out = self.leakyrelu(self.fromrgb_layers[0](x))

        if self.steps == 0: # i.e, image size is 4x4
            
            # (C, 4, 4) -> (1, 1, 1)
            out = self.prog_blocks[-1](out)

            # (1, 1, 1) -> (1)
            # out.size(0) = batch_size
            return out.view(out.size(0), -1)
        

        downscaled = self.leakyrelu(self.fromrgb_layers[1](self.avgpool(x)))
        out = self.prog_blocks[0](out)

        out = self.fade_in(alpha, downscaled, out)
        
        for i in range(1, self.steps+1):
            out = self.prog_blocks[i](out)

        return out.view(out.size(0), -1)



if __name__ == '__main__':
    g_out_shapes = [(1, 3, 4, 4), (1, 3, 8, 8), (1, 3, 16, 16), (1, 3, 32, 32), (1, 3, 64, 64), (1, 3, 128, 128), (1, 3, 256, 256)]
    d_out_shapes = (1, 1)

    for steps in range(7):
        z_vector = torch.randn((1, 128, 1, 1))
        generator = Generator(steps)
        discriminator = Discriminator(steps)
        g_out = generator(z_vector, alpha=0.5)
        d_out = discriminator(g_out, alpha=0.5)

        if g_out.shape == g_out_shapes[steps] and d_out.shape == d_out_shapes:
            print("SUCCESS steps:", steps)
