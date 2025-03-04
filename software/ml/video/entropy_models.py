# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai._CXX import pmf_to_quantized_cdf
from compressai.ops.bound_ops import LowerBound
from compressai import ans

class EntropyBottleneck(nn.Module):
    _offset: Tensor

    def __init__(
        self, 
        channels: int, 
        tail_mass: float = 1e-9, 
        init_scale: float = 10, 
        filters: Tuple[int, ...] = (3, 3, 3, 3), 
        likelihood_bound: float = 1e-9, 
        entropy_coder_precision: int = 16
    ):
        super().__init__()

        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        self._encoder = ans.RansEncoder()
        self._decoder = ans.RansDecoder()

        self.entropy_coder_precision = int(entropy_coder_precision)
        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

        # to be filled on update()
        self.register_buffer("_offset", torch.IntTensor())
        self.register_buffer("_quantized_cdf", torch.IntTensor())
        self.register_buffer("_cdf_length", torch.IntTensor())

        # Create parameters
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels

        self.matrices = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.factors = nn.ParameterList()

        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self.matrices.append(nn.Parameter(matrix))

            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.biases.append(nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.factors.append(nn.Parameter(factor))

        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)

        target = np.log(2 / self.tail_mass - 1)
        self.register_buffer("target", torch.Tensor([-target, 0, target]))

    def update(self, force: bool = False, update_quantiles: bool = False) -> bool:
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if self._offset.numel() > 0 and not force:
            return False

        if update_quantiles:
            self._update_quantiles()

        medians = self.quantiles[:, 0, 1]

        minima = medians - self.quantiles[:, 0, 0]
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0)

        maxima = self.quantiles[:, 0, 2] - medians
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)

        self._offset = -minima

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1

        max_length = pmf_length.max().item()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)
        samples = samples[None, :] + pmf_start[:, None, None]

        lower = self._logits_cumulative(samples - 0.5, True)
        upper = self._logits_cumulative(samples + 0.5, True)
        pmf = torch.sigmoid(upper) - torch.sigmoid(lower)

        pmf = pmf[:, 0, :]
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])

        quantized_cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = torch.IntTensor(pmf_to_quantized_cdf(prob.tolist(), self.entropy_coder_precision))
            quantized_cdf[i, : _cdf.size(0)] = _cdf

        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return True

    def loss(self) -> Tensor:
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = torch.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs: Tensor, stop_gradient: bool) -> Tensor:
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = self.matrices[i]
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)

            bias = self.biases[i]
            if stop_gradient:
                bias = bias.detach()
            logits = logits + bias

            if i < len(self.filters):
                factor = self.factors[i]
                if stop_gradient:
                    factor = factor.detach()
                logits = logits + torch.tanh(factor) * torch.tanh(logits)
        return logits

    def forward(self, x: Tensor, training: bool = True) -> Tuple[Tensor, Tensor]:
        # x from B x C x ... to C x B x ...
        perm = torch.cat((
            torch.tensor([1, 0], dtype=torch.long, device=x.device),
            torch.arange(2, x.ndim, dtype=torch.long, device=x.device),
        ))
        inv_perm = perm

        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)

        # Add noise or quantize
        if training:
            outputs = values + torch.empty_like(values).uniform_(-0.5, 0.5)
        else:
            medians = self.quantiles[:, :, 1:2]
            outputs = torch.round(values.clone() - medians) + medians

        # Compute likelihood
        lower = self._logits_cumulative(outputs - 0.5, False)
        upper = self._logits_cumulative(outputs + 0.5, False)
        likelihood = torch.sigmoid(upper) - torch.sigmoid(lower)

        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):
        N, C = size[:2]

        view_dims = np.ones((len(size),), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(N, 1, *size[2:])

    @torch.no_grad()
    def _update_quantiles(self, search_radius=1e5, rtol=1e-4, atol=1e-3):
        """Fast quantile update via bisection search.

        Often faster and much more precise than minimizing aux loss.
        """
        device = self.quantiles.device
        shape = (self.channels, 1, 1)
        low = torch.full(shape, -search_radius, device=device)
        high = torch.full(shape, search_radius, device=device)

        def f(y, self=self):
            return self._logits_cumulative(y, stop_gradient=True)

        for i in range(len(self.target)):
            assert (low <= high).all()
            low = torch.where(self.target[i] <= f(high), low, high)
            high = torch.where(f(low) <= self.target[i], high, low)
            while not torch.isclose(low, high, rtol=rtol, atol=atol).all():
                mid = (low + high) / 2
                f_mid = f(mid)
                low = torch.where(f_mid <= self.target[i], mid, low)
                high = torch.where(f_mid >= self.target[i], mid, high)
            q_i = (low + high) / 2
            self.quantiles[:, :, i] = q_i[:, :, 0]

    def compress(self, x):
        indexes = self._build_indexes(x.size())
        medians = self.quantiles[:, :, 1:2].detach()
        spatial_dims = len(x.size()) - 2
        medians = medians.reshape(-1, *([1] * spatial_dims)) if spatial_dims > 0 else medians.reshape(-1)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))

        symbols = torch.round(x.clone() - medians).int()

        strings = []
        for i in range(symbols.size(0)):
            rv = self._encoder.encode_with_indexes(
                symbols[i].reshape(-1).int().tolist(),
                indexes[i].reshape(-1).int().tolist(),
                self._quantized_cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            strings.append(rv)

        return strings

    def decompress(self, strings, size):
        output_size = (len(strings), self._quantized_cdf.size(0), *size)
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        quantiles = self.quantiles[:, :, 1:2].detach()
        medians = quantiles.reshape(-1, *([1] * len(size))) if len(size) > 0 else quantiles.reshape(-1)
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))

        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size())

        for i, s in enumerate(strings):
            values = self._decoder.decode_with_indexes(
                s,
                indexes[i].reshape(-1).int().tolist(),
                cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            outputs[i] = torch.tensor(values, device=outputs.device, dtype=outputs.dtype).reshape(outputs[i].size())
        outputs = outputs.to(medians.dtype) + medians
        return outputs


class GaussianConditional(nn.Module):
    def __init__(
            self, scale_table: Optional[Union[List, Tuple]], 
            scale_bound: float = 0.11, 
            tail_mass: float = 1e-9,
            likelihood_bound: float = 1e-9,
            entropy_coder_precision: int = 16,
        ):
        super().__init__()
        
        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            scale_bound = self.scale_table[0]
        self.lower_bound_scale = LowerBound(scale_bound)

        self.register_buffer("scale_table", torch.Tensor(tuple(float(s) for s in scale_table)) if scale_table else torch.Tensor())
        self.register_buffer("scale_bound", torch.Tensor([float(scale_bound)]) if scale_bound is not None else None)

        self._encoder = ans.RansEncoder()
        self._decoder = ans.RansDecoder()

        self.entropy_coder_precision = int(entropy_coder_precision)
        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

        # to be filled on update()
        self.register_buffer("_offset", torch.IntTensor())
        self.register_buffer("_quantized_cdf", torch.IntTensor())
        self.register_buffer("_cdf_length", torch.IntTensor())

    def update_scale_table(self, scale_table, force=False):
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditonal model is
        # updated.
        if self._offset.numel() > 0 and not force:
            return False
        self.scale_table = torch.Tensor(tuple(float(s) for s in scale_table), device=self.scale_table.device)
        self.update()
        return True

    def update(self):
        multiplier = -scipy.stats.norm.ppf(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.abs(torch.arange(max_length, device=device).int() - pmf_center[:, None])
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = 0.5 * torch.erfc(-(2**-0.5) * ((0.5 - samples) / samples_scale))
        lower = 0.5 * torch.erfc(-(2**-0.5) * ((-0.5 - samples) / samples_scale))
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = torch.IntTensor(pmf_to_quantized_cdf(prob.tolist(), self.entropy_coder_precision))
            quantized_cdf[i, : _cdf.size(0)] = _cdf
        
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def forward(self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None, training: bool = True) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training

        # Add noise or quantize
        if training:
            outputs = inputs + torch.empty_like(inputs).uniform_(-0.5, 0.5)
        else:
            outputs = torch.round(inputs.clone() - means) + means

        # Compute likelihood
        values = outputs if means is None else outputs - means
        scales = self.lower_bound_scale(scales)

        values = torch.abs(values)
        upper = 0.5 * torch.erfc(-(2**-0.5) * ((0.5 - values) / scales))
        lower = 0.5 * torch.erfc(-(2**-0.5) * ((-0.5 - values) / scales))
        likelihood = upper - lower

        # Apply likelihood bound
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)

        return outputs, likelihood

    def build_indexes(self, scales: Tensor) -> Tensor:
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes
        
    def compress(self, inputs, indexes, means=None):
        symbols = torch.round(inputs.clone() - means).int()
        strings = []
        for i in range(symbols.size(0)):
            rv = self._encoder.encode_with_indexes(
                symbols[i].reshape(-1).int().tolist(),
                indexes[i].reshape(-1).int().tolist(),
                self._quantized_cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            strings.append(rv)
        return strings

    def decompress(self, strings: str, indexes: torch.IntTensor, dtype: torch.dtype = torch.float, means: torch.Tensor = None):
        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size())

        for i, s in enumerate(strings):
            values = self._decoder.decode_with_indexes(
                s,
                indexes[i].reshape(-1).int().tolist(),
                cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            outputs[i] = torch.tensor(values, device=outputs.device, dtype=outputs.dtype).reshape(outputs[i].size())
        outputs = outputs.to(dtype) + means
        return outputs
