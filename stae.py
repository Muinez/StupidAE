import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm2d(nn.Module):
	def __init__(self, channels, eps=1e-8, affine=True):
		super().__init__()
		self.eps = eps
		self.affine = affine
		if affine:
			self.weight = nn.Parameter(torch.ones(channels))
		else:
			self.register_parameter("weight", None)

	def forward(self, x):
		norm = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
		x = x * norm
		if self.affine:
			x = x * self.weight[:, None, None]
		return x

class ConvMlp(nn.Module):
	def __init__(self, in_features, hidden_features=None, out_features=None):
		super().__init__()
		self.model = nn.Sequential(
			nn.Conv2d(in_channels=in_features, out_channels=hidden_features, kernel_size=1),
			nn.GELU(),
			nn.Conv2d(in_channels=hidden_features, out_channels=out_features, kernel_size=1),
		)

	def forward(self, x):
		return self.model(x)

import torch
import torch.nn as nn
class GegluMlp(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv_up = nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=1)
        self.conv_down = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, x):
        x = self.conv_up(x)
        x_gate, x_act = torch.chunk(x, 2, dim=1)
        x = self.activation(x_act) * x_gate
        x = self.conv_down(x)
        
        return x

class EncoderBlock(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.norm = RMSNorm2d(channels)
		hidden_dim = channels

		self.mlp = GegluMlp(hidden_dim)
	   
	def forward(self, x):
		norm = self.norm(x)
		mlp_out = self.mlp(norm)
		x = x + mlp_out

		return x

class DecoderBlock(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.norm = RMSNorm2d(channels)

		self.mlp = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=1),
			nn.GELU(approximate="tanh"),
			nn.Conv2d(channels, channels, kernel_size=3, padding=1),
		)
		
	def forward(self, x):
		norm = self.norm(x)
		mlp_out = self.mlp(norm)
		x = x + mlp_out

		return x

class StupidEncoder(nn.Module):
	def __init__(self,
				 hidden_dim,
				 in_channels,
				 out_channels,
				 patch_size,
				 num_blocks):
		super().__init__()

		self.initial = nn.Sequential(
			nn.Conv2d(in_channels, hidden_dim, patch_size, padding=0, stride=patch_size),
		)

		self.blocks = nn.ModuleList(EncoderBlock(hidden_dim) for _ in range(num_blocks))
		self.out = ConvMlp(hidden_dim, hidden_dim, out_channels)

	def forward(self, x):
		x = self.initial(x)

		for block in self.blocks:
			x = block(x)

		x = self.out(x)
		return x

class NerfHead(nn.Module):
	def __init__(self, patch_dim, mlp_dim):
		super().__init__()
		self.mlp_dim = mlp_dim
		self.param_gen = nn.Linear(patch_dim, self.mlp_dim*self.mlp_dim*2)
		self.norm = nn.RMSNorm(self.mlp_dim)

	def forward(self, pixels, patches):
		bs = pixels.shape[0]
		params = self.param_gen(patches)
		layer1, layer2 = params.chunk(2, dim=-1)
		layer1 = layer1.view(bs, self.mlp_dim, self.mlp_dim)
		layer2 = layer2.view(bs, self.mlp_dim, self.mlp_dim)

		layer1 = torch.nn.functional.normalize(layer1, dim=-2)

		res_x = pixels
		pixels = self.norm(pixels)
		pixels = torch.bmm(pixels, layer1)
		pixels = torch.nn.functional.silu(pixels)
		pixels = torch.bmm(pixels, layer2)
		pixels = pixels + res_x
		return pixels

class StupidDecoder(nn.Module):
	def __init__(self,
				 hidden_dim,
				 in_channels,
				 out_channels,
				 patch_size,
				 num_blocks,
				 nerf_blocks,
				 mlp_dim):
		super().__init__()
		
		self.out_channels = out_channels

		self.patch_size = patch_size
		self.conv_in = ConvMlp(in_channels, hidden_dim, hidden_dim)
		self.blocks = []
		for _ in range(num_blocks):
			self.blocks.append(DecoderBlock(hidden_dim))
			self.blocks.append(EncoderBlock(hidden_dim))
		self.blocks = nn.ModuleList(self.blocks)

		self.nerf = nn.ModuleList(NerfHead(hidden_dim, mlp_dim) for _ in range(nerf_blocks))
		self.positions = nn.Parameter(torch.randn(1, self.patch_size**2, mlp_dim))
		self.last = nn.Linear(mlp_dim, self.out_channels)

	def forward(self, x):
		B, C, H, W = x.shape
		x = self.conv_in(x)
		for block in self.blocks:
			x = block(x)

		patches = x.flatten(2).transpose(1,2) # B C H W -> B (HW) C 
		patch_count = H*W
		total_len = x.shape[0] * patch_count
		patches = patches.reshape(total_len, -1)
		x = self.positions.repeat(total_len, 1, 1)

		for block in self.nerf:
			x = block(x, patches) # B * patch_count, ps*ps, C
		x = self.last(x)
		x = x.transpose(1,2) # [B * patch_count, ps*ps, C] -> [B*patch_count, C, ps*ps]
		x = x.reshape(B, patch_count, -1) # [B*patch_count, C, ps*ps] -> [B, patch_count, ps*ps*3]
		x = x.transpose(1,2) # [B, patch_count, ps*ps*3] -> [B, ps*ps*3, patch_count]
		x = torch.nn.functional.fold(x.contiguous(),
                                     (H*self.patch_size, W*self.patch_size),
                                     kernel_size=self.patch_size,
                                     stride=self.patch_size)

		return x

class SimpleStupidDecoder(nn.Module):
	def __init__(self,
				 hidden_dim,
				 in_channels,
				 out_channels,
				 patch_size,
				 num_blocks):
		super().__init__()
		
		self.out_channels = out_channels
		self.patch_size = patch_size

		self.conv_in = ConvMlp(in_channels, hidden_dim, hidden_dim)
		self.blocks = nn.ModuleList(DecoderBlock(hidden_dim) for _ in range(num_blocks))

		self.last = nn.Sequential(
			ConvMlp(hidden_dim, hidden_dim, out_channels * patch_size * patch_size),
			nn.PixelShuffle(patch_size)
		)

	def forward(self, x):
		x = self.conv_in(x)
		for block in self.blocks:
			x = block(x)

		return self.last(x)

class StupidAE(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.encoder = nn.Sequential(
			StupidEncoder(in_channels=3, out_channels=16, hidden_dim=512, patch_size=8, num_blocks=2),
		)
		self.decoder = nn.Sequential(
			StupidDecoder(in_channels=16, out_channels=3, hidden_dim=512, patch_size=8, num_blocks=2, nerf_blocks=1, mlp_dim=32)
		)
	
	def encode(self, x):
		return self.encoder(x)
	
	def decode(self, x):
		return self.decoder(x)
	
	def forward(self, x):
		x = self.encode(x)
		x = self.decode(x)
		return x
