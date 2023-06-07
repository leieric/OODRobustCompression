import torch
import torch.nn as nn
import torch.autograd
import torch.nn.functional as F

class AutoencoderQ(nn.Module):
    def __init__(self, encoder, decoder, quantizer):
        super(AutoencoderQ, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.quantizer(x)
        x = self.decoder(x)
        return x

class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        super(Generator, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.feature_sizes = (int(self.img_size[0] / 16), int(self.img_size[1] / 16))

        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, 8 * dim * self.feature_sizes[0] * self.feature_sizes[1]),
            nn.ReLU()
        )

        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim),
            nn.ConvTranspose2d(4 * dim, 2 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),
            nn.ConvTranspose2d(2 * dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, self.img_size[2], 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(input_data)
        # Reshape
        x = x.view(-1, 8 * self.dim, self.feature_sizes[0], self.feature_sizes[1])
        # Return generated image
        return self.features_to_image(x)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))


class Discriminator(nn.Module):
    def __init__(self, img_size, dim):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Discriminator, self).__init__()

        self.img_size = img_size

        self.image_to_features = nn.Sequential(
            nn.Conv2d(self.img_size[2], dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )

        # 4 convolutions of stride 2, i.e. halving of size everytime
        # So output size will be 8 * (img_size / 2 ^ 4) * (img_size / 2 ^ 4)
        output_size = int(8 * dim * (img_size[0] / 16) * (img_size[1] / 16))
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1)
        )

    def forward(self, input_data):
        batch_size = input_data.size()[0]
        x = self.image_to_features(input_data)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)
    
class Encoder(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Encoder, self).__init__()

        self.img_size = img_size

        self.image_to_features = nn.Sequential(
            nn.Conv2d(self.img_size[2], dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
#             nn.Dropout2d(0.1),
            nn.BatchNorm2d(2 * dim),
            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
#             nn.Dropout2d(0.1),
            nn.BatchNorm2d(4 * dim),
            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            nn.Sigmoid()
        )

        # 4 convolutions of stride 2, i.e. halving of size everytime
        # So output size will be 8 * (img_size / 2 ^ 4) * (img_size / 2 ^ 4)
        output_size = int(8 * dim * (img_size[0] / 16) * (img_size[1] / 16))
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, latent_dim),
            nn.Tanh()
        )

    def forward(self, input_data):
        batch_size = input_data.size()[0]
        x = self.image_to_features(input_data)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)
    
class Quantizer(nn.Module):
    """
    Scalar Quantizer module
    Source: https://github.com/mitscha/dplc
    """
    def __init__(self, levels=[-1.0, 1.0], sigma=1.0):
        super(Quantizer, self).__init__()
        self.levels = levels
        self.sigma = sigma

    def forward(self, x):
        levels = x.data.new(self.levels)
        xsize = list(x.size())

        # Compute differentiable soft quantized version
        x = x.view(*(xsize + [1]))
        level_var = torch.autograd.Variable(levels)
        dist = torch.pow(x-level_var, 2)
        val = torch.nn.functional.softmax(-self.sigma*dist, dim=-1)
        val2 = level_var.clone() * val
        output = torch.sum(val2, dim=-1)

        # Compute hard quantization (invisible to autograd)
        _, symbols = torch.min(dist.data, dim=-1, keepdim=True)
        for _ in range(len(xsize)): levels.unsqueeze_(0)
        levels = levels.expand(*(xsize + [len(self.levels)]))

        quant = levels.gather(-1, symbols.long()).squeeze_(dim=-1)

        # Replace activations in soft variable with hard quantized version
        output.data = quant

        return output
    
class Noise(nn.Module):
    def __init__(self, L, device):
        super().__init__()
        self.a = 2/(L-1)
        self.device = device

    def forward(self, din):
        return din + torch.autograd.Variable((self.a*torch.rand(din.size())-(self.a/2)*torch.ones(din.size())).to(self.device))
