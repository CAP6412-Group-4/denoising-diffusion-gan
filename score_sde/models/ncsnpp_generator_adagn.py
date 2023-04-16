# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the Score SDE library
# which was released under the Apache License.
#
# Source:
# https://github.com/yang-song/score_sde_pytorch/blob/main/models/layerspp.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_Apache). The modifications
# to this file are subject to the same Apache License.
# ---------------------------------------------------------------

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
''' Codes adapted from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnpp.py
'''

from . import utils, layers, layerspp, dense_layer
import torch.nn as nn
# import functools
import torch
from .mdm import PositionalEncoding, TimestepEmbedder, EmbedAction, InputProcess, OutputProcess
import clip

# ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp_Adagn
# ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp_Adagn
# ResnetBlockBigGAN_one = layerspp.ResnetBlockBigGANpp_Adagn_one
# Combine = layerspp.Combine
# conv3x3 = layerspp.conv3x3
# conv1x1 = layerspp.conv1x1
# get_act = layers.get_act
# default_initializer = layers.default_init
dense = dense_layer.dense

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


@utils.register_model(name='ncsnpp')
class NCSNpp(nn.Module):
  """NCSN++ model"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    # self.not_use_tanh = config.not_use_tanh
    self.act = nn.SiLU()
    self.z_emb_dim = 512
    
    # self.nf = nf = config.num_channels_dae
    # ch_mult = config.ch_mult
    # self.num_res_blocks = num_res_blocks = config.num_res_blocks
    # self.attn_resolutions = attn_resolutions = config.attn_resolutions
    # dropout = config.dropout
    # resamp_with_conv = config.resamp_with_conv
    # self.num_resolutions = num_resolutions = len(ch_mult)
    # self.all_resolutions = all_resolutions = [config.image_size // (2 ** i) for i in range(num_resolutions)]

    # self.conditional = conditional = config.conditional  # noise-conditional
    # fir = config.fir
    # fir_kernel = config.fir_kernel
    # self.skip_rescale = skip_rescale = config.skip_rescale
    # self.resblock_type = resblock_type = config.resblock_type.lower()
    # self.progressive = progressive = config.progressive.lower()
    # self.progressive_input = progressive_input = config.progressive_input.lower()
    # self.embedding_type = embedding_type = config.embedding_type.lower()
    # init_scale = 0.
    # assert progressive in ['none', 'output_skip', 'residual']
    # assert progressive_input in ['none', 'input_skip', 'residual']
    # assert embedding_type in ['fourier', 'positional']
    # combine_method = config.progressive_combine.lower()
    # combiner = functools.partial(Combine, method=combine_method)
    self.input_process = InputProcess(data_rep='rot6d', input_feats=263, latent_dim=self.z_emb_dim)
    self.output_process = OutputProcess(data_rep='rot6d', input_feats=263, latent_dim=self.z_emb_dim, njoints=263, nfeats=1)
    self.position_encoder = PositionalEncoding(self.z_emb_dim)
    self.embed_timestep = TimestepEmbedder(self.z_emb_dim, self.position_encoder)
    self.cond_mask_prob = 0.1

    seqTransEncoderLayer = nn.TransformerEncoderLayer(
      d_model=self.z_emb_dim,
      nhead=4,
      dim_feedforward=1024,
      dropout=0.1,
      activation="gelu",
    )

    self.seqTransEncoder = nn.TransformerEncoder(encoder_layer=seqTransEncoderLayer, num_layers=8)
    

    # modules: list[nn.Module] = []
    # # timestep/noise_level embedding; only for continuous training
    # if embedding_type == 'fourier':
    #   # Gaussian Fourier features embeddings.
    #   #assert config.training.continuous, "Fourier features are only used for continuous training."

    #   modules.append(layerspp.GaussianFourierProjection(
    #     embedding_size=nf, scale=config.fourier_scale
    #   ))
    #   embed_dim = 2 * nf

    # elif embedding_type == 'positional':
    #   embed_dim = nf

    # else:
    #   raise ValueError(f'embedding type {embedding_type} unknown.')

      # # conditional timestep embeddings
      # modules.append(nn.Linear(embed_dim, nf * 4))
      # modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      # nn.init.zeros_(modules[-1].bias)
      # modules.append(nn.Linear(nf * 4, nf * 4))
      # modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      # nn.init.zeros_(modules[-1].bias)

    # Conditional Text Embeddings
    self.embed_text = nn.Linear(512, self.z_emb_dim)
    print('EMBED TEXT')
    print('Loading CLIP...')
    self.clip_version = 'ViT-B/32'
    self.clip_model = self.load_and_freeze_clip(self.clip_version)
    self.embedding_type = "positional"
    self.conditional = True
      # Conditional action embeddings

      # TODO Number of actions should be the equivalent of `dataset.num_actions`
    # self.embed_action = EmbedAction(5, self.z_emb_dim)
    # print('EMBED ACTION')

      # modules.append(nn.Linear(embed_dim, nf * 4))
      # modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      # nn.init.zeros_(modules[-1].bias)
      # modules.append(nn.Linear(nf * 4, nf * 4))
      # modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      # nn.init.zeros_(modules[-1].bias)  

    # AttnBlock = functools.partial(layerspp.AttnBlockpp,
    #                               init_scale=init_scale,
    #                               skip_rescale=skip_rescale)

    # Upsample = functools.partial(layerspp.Upsample,
    #                              with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    # if progressive == 'output_skip':
    #   self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    # elif progressive == 'residual':
    #   pyramid_upsample = functools.partial(layerspp.Upsample,
    #                                        fir=fir, fir_kernel=fir_kernel, with_conv=True)

    # Downsample = functools.partial(layerspp.Downsample,
    #                                with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    # if progressive_input == 'input_skip':
    #   self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    # elif progressive_input == 'residual':
    #   pyramid_downsample = functools.partial(layerspp.Downsample,
    #                                          fir=fir, fir_kernel=fir_kernel, with_conv=True)

    # if resblock_type == 'ddpm':
    #   ResnetBlock = functools.partial(ResnetBlockDDPM,
    #                                   act=act,
    #                                   dropout=dropout,
    #                                   init_scale=init_scale,
    #                                   skip_rescale=skip_rescale,
    #                                   temb_dim=nf * 4,
    #                                   zemb_dim = z_emb_dim)

    # elif resblock_type == 'biggan':
    #   ResnetBlock = functools.partial(ResnetBlockBigGAN,
    #                                   act=act,
    #                                   dropout=dropout,
    #                                   fir=fir,
    #                                   fir_kernel=fir_kernel,
    #                                   init_scale=init_scale,
    #                                   skip_rescale=skip_rescale,
    #                                   temb_dim=nf * 4,
    #                                   zemb_dim = z_emb_dim)
    # elif resblock_type == 'biggan_oneadagn':
    #   ResnetBlock = functools.partial(ResnetBlockBigGAN_one,
    #                                   act=act,
    #                                   dropout=dropout,
    #                                   fir=fir,
    #                                   fir_kernel=fir_kernel,
    #                                   init_scale=init_scale,
    #                                   skip_rescale=skip_rescale,
    #                                   temb_dim=nf * 4,
    #                                   zemb_dim = z_emb_dim)

    # else:
    #   raise ValueError(f'resblock type {resblock_type} unrecognized.')

    # Downsampling block

    # channels = config.num_channels
    # if progressive_input != 'none':
    #   input_pyramid_ch = channels

    # modules.append(conv3x3(channels, nf))
    # hs_c = [nf]

    # in_ch = nf
    # for i_level in range(num_resolutions):
    #   # Residual blocks for this resolution
    #   for i_block in range(num_res_blocks):
    #     out_ch = nf * ch_mult[i_level]
    #     modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
    #     in_ch = out_ch

    #     if all_resolutions[i_level] in attn_resolutions:
    #       modules.append(AttnBlock(channels=in_ch))
    #     hs_c.append(in_ch)

    #   if i_level != num_resolutions - 1:
    #     if resblock_type == 'ddpm':
    #       modules.append(Downsample(in_ch=in_ch))
    #     else:
    #       modules.append(ResnetBlock(down=True, in_ch=in_ch))

    #     if progressive_input == 'input_skip':
    #       modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
    #       if combine_method == 'cat':
    #         in_ch *= 2

    #     elif progressive_input == 'residual':
    #       modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
    #       input_pyramid_ch = in_ch

    #     hs_c.append(in_ch)

    # in_ch = hs_c[-1]
    # modules.append(ResnetBlock(in_ch=in_ch))
    # modules.append(AttnBlock(channels=in_ch))
    # modules.append(ResnetBlock(in_ch=in_ch))

    # pyramid_ch = 0
    # # Upsampling block
    # for i_level in reversed(range(num_resolutions)):
    #   for i_block in range(num_res_blocks + 1):
    #     out_ch = nf * ch_mult[i_level]
    #     modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
    #                                out_ch=out_ch))
    #     in_ch = out_ch

    #   if all_resolutions[i_level] in attn_resolutions:
    #     modules.append(AttnBlock(channels=in_ch))

    #   if progressive != 'none':
    #     if i_level == num_resolutions - 1:
    #       if progressive == 'output_skip':
    #         modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
    #                                     num_channels=in_ch, eps=1e-6))
    #         modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
    #         pyramid_ch = channels
    #       elif progressive == 'residual':
    #         modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
    #                                     num_channels=in_ch, eps=1e-6))
    #         modules.append(conv3x3(in_ch, in_ch, bias=True))
    #         pyramid_ch = in_ch
    #       else:
    #         raise ValueError(f'{progressive} is not a valid name.')
    #     else:
    #       if progressive == 'output_skip':
    #         modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
    #                                     num_channels=in_ch, eps=1e-6))
    #         modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
    #         pyramid_ch = channels
    #       elif progressive == 'residual':
    #         modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
    #         pyramid_ch = in_ch
    #       else:
    #         raise ValueError(f'{progressive} is not a valid name')

    #   if i_level != 0:
    #     if resblock_type == 'ddpm':
    #       modules.append(Upsample(in_ch=in_ch))
    #     else:
    #       modules.append(ResnetBlock(in_ch=in_ch, up=True))

    # assert not hs_c

    # if progressive != 'output_skip':
    #   modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
    #                               num_channels=in_ch, eps=1e-6))
    #   modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

    # self.all_modules = nn.ModuleList(modules)
    
    
    mapping_layers = [
      nn.Linear(128, 400),
      self.act,
      nn.Linear(400, 512),
      self.act, 
    ]
    # for _ in range(config.n_mlp):
    #     mapping_layers.append(dense(self.z_emb_dim, self.z_emb_dim))
    #     mapping_layers.append(self.act)
    self.z_transform = nn.Sequential(*mapping_layers)

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
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

  def forward(self, x, t, y, z):
    # timestep/noise_level embedding; only for continuous training
    zemb = self.z_transform(z).unsqueeze(0)
    # print(f"zemb={zemb.shape}")

    zemb += self.embed_timestep(t)
   
    # if self.embedding_type == 'fourier':
    #   # Gaussian Fourier features embeddings.
    #   used_sigmas = time_cond
    #   temb = modules[m_idx](torch.log(used_sigmas))
    #   m_idx += 1

    x = self.input_process(x)

    if self.embedding_type == 'positional' and self.conditional:

      # force_masky.get('uncond', False), y.keys())

      # text embeddings
      encoded_text = self.encode_text(y['text'])
      toAdd = self.embed_text(self.mask_cond(encoded_text, force_mask=False))

      # print(toAdd.size(), encoded_text.size(), len(y['text']), y.get('uncond', False))

      zemb += toAdd


      # action embeddings
      # action_embedding = self.embed_action(action_cond)
      # zemb += self.mask_cond(action_embedding, force_mask=True)

    else:
      raise ValueError(f'embedding type {self.embedding_type} unknown.')

    # adding the timestep embed
    xseq = torch.cat((zemb, x), axis=0)  # [seqlen+1, bs, d]
    xseq = self.position_encoder(xseq)  # [seqlen+1, bs, d]
    output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

    output = self.output_process(output)
    return output

    # MDM handles the forward() passes in it's own models
    # if self.conditional:
    #   temb = modules[m_idx](temb)
    #   m_idx += 1
    #   temb = modules[m_idx](self.act(temb))
    #   m_idx += 1
    # else:
    #   temb = None

    # if not self.config.centered:
    #   # If input data is in [0, 1]
    #   x = 2 * x - 1.

    # Downsampling block


    # input_pyramid = None
    # if self.progressive_input != 'none':
    #   input_pyramid = x

    # hs = [modules[m_idx](x)]
    # m_idx += 1
    # for i_level in range(self.num_resolutions):
    #   # Residual blocks for this resolution
    #   for i_block in range(self.num_res_blocks):
    #     h = modules[m_idx](hs[-1], temb, zemb)
    #     m_idx += 1
    #     if h.shape[-1] in self.attn_resolutions:
    #       h = modules[m_idx](h)
    #       m_idx += 1

    #     hs.append(h)

    #   if i_level != self.num_resolutions - 1:
    #     if self.resblock_type == 'ddpm':
    #       h = modules[m_idx](hs[-1])
    #       m_idx += 1
    #     else:
    #       h = modules[m_idx](hs[-1], temb, zemb)
    #       m_idx += 1

    #     if self.progressive_input == 'input_skip':
    #       input_pyramid = self.pyramid_downsample(input_pyramid)
    #       h = modules[m_idx](input_pyramid, h)
    #       m_idx += 1

    #     elif self.progressive_input == 'residual':
    #       input_pyramid = modules[m_idx](input_pyramid)
    #       m_idx += 1
    #       if self.skip_rescale:
    #         input_pyramid = (input_pyramid + h) / np.sqrt(2.)
    #       else:
    #         input_pyramid = input_pyramid + h
    #       h = input_pyramid

    #     hs.append(h)

    # h = hs[-1]
    # h = modules[m_idx](h, temb, zemb)
    # m_idx += 1
    # h = modules[m_idx](h)
    # m_idx += 1
    # h = modules[m_idx](h, temb, zemb)
    # m_idx += 1

    # pyramid = None

    # # Upsampling block
    # for i_level in reversed(range(self.num_resolutions)):
    #   for i_block in range(self.num_res_blocks + 1):
    #     h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, zemb)
    #     m_idx += 1

    #   if h.shape[-1] in self.attn_resolutions:
    #     h = modules[m_idx](h)
    #     m_idx += 1

    #   if self.progressive != 'none':
    #     if i_level == self.num_resolutions - 1:
    #       if self.progressive == 'output_skip':
    #         pyramid = self.act(modules[m_idx](h))
    #         m_idx += 1
    #         pyramid = modules[m_idx](pyramid)
    #         m_idx += 1
    #       elif self.progressive == 'residual':
    #         pyramid = self.act(modules[m_idx](h))
    #         m_idx += 1
    #         pyramid = modules[m_idx](pyramid)
    #         m_idx += 1
    #       else:
    #         raise ValueError(f'{self.progressive} is not a valid name.')
    #     else:
    #       if self.progressive == 'output_skip':
    #         pyramid = self.pyramid_upsample(pyramid)
    #         pyramid_h = self.act(modules[m_idx](h))
    #         m_idx += 1
    #         pyramid_h = modules[m_idx](pyramid_h)
    #         m_idx += 1
    #         pyramid = pyramid + pyramid_h
    #       elif self.progressive == 'residual':
    #         pyramid = modules[m_idx](pyramid)
    #         m_idx += 1
    #         if self.skip_rescale:
    #           pyramid = (pyramid + h) / np.sqrt(2.)
    #         else:
    #           pyramid = pyramid + h
    #         h = pyramid
    #       else:
    #         raise ValueError(f'{self.progressive} is not a valid name')

    #   if i_level != 0:
    #     if self.resblock_type == 'ddpm':
    #       h = modules[m_idx](h)
    #       m_idx += 1
    #     else:
    #       h = modules[m_idx](h, temb, zemb)
    #       m_idx += 1

    # assert not hs

    # if self.progressive == 'output_skip':
    #   h = pyramid
    # else:
    #   h = self.act(modules[m_idx](h))
    #   m_idx += 1
    #   h = modules[m_idx](h)
    #   m_idx += 1

    # assert m_idx == len(modules)
    
    # if not self.not_use_tanh:

    #     return torch.tanh(h)
    # else:
    #     return h
