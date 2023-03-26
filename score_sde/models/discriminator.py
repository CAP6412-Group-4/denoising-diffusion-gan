# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np

from model.mdm import InputProcess, OutputProcess, PositionalEncoding, TimestepEmbedder
import clip

from . import up_or_down_sampling
from . import dense_layer
from . import layers

dense = dense_layer.dense
conv2d = dense_layer.conv2d
get_sinusoidal_positional_embedding = layers.get_timestep_embedding

# class TimestepEmbedding(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
#         super().__init__()

#         self.embedding_dim = embedding_dim
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim

#         self.main = nn.Sequential(
#             dense(embedding_dim, hidden_dim),
#             act,
#             dense(hidden_dim, output_dim),
#         )

#     def forward(self, temp):
#         temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
#         temb = self.main(temb)
#         return temb
#%%

class MotionDiscriminator(nn.Module):
    def __init__(self, t_emb_dim: int):
        super().__init__()

        self.clip_version = 'ViT-B/32'
        self.clip_model = self.load_and_freeze_clip(self.clip_version)
        self.position_encoder = PositionalEncoding(t_emb_dim)

        self.input_process = InputProcess(data_rep='rot6d', input_feats=263, latent_dim=t_emb_dim)
        self.timestep_embedder = TimestepEmbedder(t_emb_dim, self.position_encoder)
        self.text_embedder = nn.Linear(512, t_emb_dim)

        self.act = nn.GELU()
        self.conv1 = nn.Conv1d(263, 200, kernel_size=5)
        self.conv2 = nn.Conv1d(200, 150, kernel_size=3)
        self.conv3 = nn.Conv1d(150, 75, kernel_size=5)

        self.output_process = OutputProcess(data_rep='rot6d', input_feats=263, latent_dim=t_emb_dim, njoints=263, nfeats=1)

        self.fc = nn.Linear(263, 1)

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
  
    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 # if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        else:
            return cond

    def forward(self, x, t, x_t, y):
        emb = self.timestep_embedder(t)
        input = self.input_process(x)

        encoded_text = self.encode_text(y['text'])
        emb += self.text_embedder(self.mask_cond(encoded_text, force_mask=False))
        
        input = self.position_encoder(torch.cat(emb, x), axis=0)

        output = self.act(self.conv1(input))
        output = self.act(self.conv2(input))
        output = self.act(self.conv3(input))

        output = self.output_process(output)
        output = self.fc(output)

        return torch.sigmoid(output)


class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        t_emb_dim = 128,
        downsample=False,
        act = nn.LeakyReLU(0.2),
        fir_kernel=(1, 3, 3, 1)
    ):
        super().__init__()
     
        
        self.fir_kernel = fir_kernel
        self.downsample = downsample
        
        self.conv1 = nn.Sequential(
                    conv2d(in_channel, out_channel, kernel_size, padding=padding),
                    )

        
        self.conv2 = nn.Sequential(
                    conv2d(out_channel, out_channel, kernel_size, padding=padding,init_scale=0.)
                    )
        self.dense_t1= dense(t_emb_dim, out_channel)


        self.act = act
        
            
        self.skip = nn.Sequential(
                    conv2d(in_channel, out_channel, 1, padding=0, bias=False),
                    )
        
            

    def forward(self, input, t_emb):
        
        out = self.act(input)
        out = self.conv1(out)
        out += self.dense_t1(t_emb)[..., None, None]
       
        out = self.act(out)
       
        if self.downsample:
            out = up_or_down_sampling.downsample_2d(out, self.fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, self.fir_kernel, factor=2)
        out = self.conv2(out)
        
        
        skip = self.skip(input)
        out = (out + skip) / np.sqrt(2)


        return out
    
class Discriminator_small(nn.Module):
  """A time-dependent discriminator for small images (CIFAR10, StackMNIST)."""

  def __init__(self, nc = 3, ngf = 64, t_emb_dim = 512, act=nn.LeakyReLU(0.2), downsample=True):
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.act = act
    
    self.t_embed = TimestepEmbedder(
        t_emb_dim,
        PositionalEncoding(self.t_emb_dim)
    )
    
    
    
     
     
     
    # Encoding layers where the resolution decreases
    # self.start_conv = conv2d(nc,ngf*2,1, padding=0)
    # self.conv1 = DownConvBlock(ngf*2, ngf*2, t_emb_dim = t_emb_dim,downsample=downsample,act=act)
    
    # self.conv2 = DownConvBlock(ngf*2, ngf*4,  t_emb_dim = t_emb_dim, downsample=downsample,act=act)
    
    
    # self.conv3 = DownConvBlock(ngf*4, ngf*8,  t_emb_dim = t_emb_dim, downsample=downsample,act=act)
    
    
    self.conv4 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=downsample,act=act)

    self.conv4 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=downsample,act=act)
    
    # self.conv4 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=downsample,act=act)
    
    
    # self.final_conv = conv2d(ngf*8 + 1, ngf*8, 3,padding=1, init_scale=0.)
    # self.end_linear = dense(ngf*8, 1)
    
    # self.stddev_group = 4
    # self.stddev_feat = 1
        
  def forward(self, x, t, x_t, y):
    t_embed = self.act(self.t_embed(t) + self.encode_text(y['text']))
    
    input_x = torch.cat((x, x_t), dim = 1)
    
    h0 = self.start_conv(input_x)
    h1 = self.conv1(h0,t_embed)    
    
    h2 = self.conv2(h1,t_embed)   
   
    h3 = self.conv3(h2,t_embed)
   
    
    out = self.conv4(h3,t_embed)
    
    batch, channel, height, width = out.shape
    group = min(batch, self.stddev_group)
    stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
    stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
    stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
    stddev = stddev.repeat(group, 1, height, width)
    out = torch.cat([out, stddev], 1)
    
    out = self.final_conv(out)
    out = self.act(out)
   
    
    out = out.view(out.shape[0], out.shape[1], -1).sum(2)
    out = self.end_linear(out)
    
    return out


class Discriminator_large(nn.Module):
  """A time-dependent discriminator for large images (CelebA, LSUN)."""

  def __init__(self, nc = 1, ngf = 32, t_emb_dim = 512, act=nn.LeakyReLU(0.2), downsample=True):
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.act = act
    self.clip_version = 'ViT-B/32'
    self.clip_model = self.load_and_freeze_clip(self.clip_version)
    
    self.position_encoder = PositionalEncoding(t_emb_dim)
    self.t_embed = TimestepEmbedder(t_emb_dim, self.position_encoder)
      
    self.start_conv = conv2d(nc,ngf*2,1, padding=0)
    self.conv1 = DownConvBlock(ngf*2, ngf*4, t_emb_dim = t_emb_dim, downsample = downsample, act=act)
    
    self.conv2 = DownConvBlock(ngf*4, ngf*8,  t_emb_dim = t_emb_dim, downsample=downsample,act=act)

    self.conv3 = DownConvBlock(ngf*8, ngf*8,  t_emb_dim = t_emb_dim, downsample=downsample,act=act)
    
    
    self.conv4 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=downsample,act=act)
    self.conv5 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=downsample,act=act)
    self.conv6 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=downsample,act=act)

  
    self.final_conv = conv2d(ngf*8 + 1, ngf*8, 3,padding=1)
    self.end_linear = dense(ngf*8, 1)
    
    self.stddev_group = 4
    self.stddev_feat = 1
    
  def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
  
  def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 # if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
        
  def forward(self, x, t, x_t, y):
    t_embed = self.act(torch.cat((self.t_embed(t), self.encode_text(y['text']).unsqueeze(0))))  
    
    input_x = torch.cat((x, x_t), dim = 1)
    
    h = self.start_conv(input_x)
    h = self.conv1(h,t_embed)    
   
    h = self.conv2(h,t_embed)
   
    h = self.conv3(h,t_embed)
    h = self.conv4(h,t_embed)
    h = self.conv5(h,t_embed)
   
    
    out = self.conv6(h,t_embed)
    
    batch, channel, height, width = out.shape
    group = min(batch, self.stddev_group)
    stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
    stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
    stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
    stddev = stddev.repeat(group, 1, height, width)
    out = torch.cat([out, stddev], 1)
    
    out = self.final_conv(out)
    out = self.act(out)
    
    out = out.view(out.shape[0], out.shape[1], -1).sum(2)
    out = self.end_linear(out)
    
    return out

