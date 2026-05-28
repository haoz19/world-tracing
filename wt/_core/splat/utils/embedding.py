# from https://github.com/lab4d-org/lab4d/

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


def frameid_to_vid(
    fid: Float[Tensor, "nframes"],
    frame_offset: Float[Tensor, "nvideos + 1"],
) -> Float[Tensor, "nframes"]:
    """Given absolute frame ids [0, ..., N], compute the video id of each frame.

    Args:
        fid: Absolute frame ids
          e.g. [0, 1, 2, 3, 100, 101, 102, 103, 200, 201, 202, 203]
        frame_offset: Offset of each video
          e.g., [0, 100, 200, 300]
    Returns:
        vid: Maps idx to video id
        tid: Maps idx to relative frame id
    """
    vid = torch.zeros_like(fid)
    for i in range(frame_offset.shape[0] - 1):
        assign = torch.logical_and(fid >= frame_offset[i], fid < frame_offset[i + 1])
        vid[assign] = i
    return vid


class TimeInfo(nn.Module):
    """Stores constant information about the time embedding

    Args:
        frame_info (Dict): Metadata about the frames in a dataset
    """

    def __init__(self, frame_info, time_scale=1.0):
        super().__init__()
        self.frame_offset = frame_info["frame_offset"]
        self.frame_offset_raw = frame_info["frame_offset"]
        self.num_frames = self.frame_offset[-1]
        self.num_vids = len(self.frame_offset) - 1

        frame_offset_raw = np.asarray(frame_info["frame_offset"])

        max_ts = (frame_offset_raw[1:] - frame_offset_raw[:-1]).max()
        raw_fid = torch.arange(0, frame_offset_raw[-1])
        raw_fid_to_vid = frameid_to_vid(raw_fid, frame_offset_raw)
        raw_fid_to_vstart = torch.tensor(frame_offset_raw[raw_fid_to_vid]).reshape(-1)
        raw_fid_to_vidend = torch.tensor(frame_offset_raw[raw_fid_to_vid + 1]).reshape(
            -1
        )
        raw_fid_to_vidlen = raw_fid_to_vidend - raw_fid_to_vstart

        self.register_buffer("raw_fid_to_vid", raw_fid_to_vid, persistent=False)
        self.register_buffer("raw_fid_to_vidlen", raw_fid_to_vidlen, persistent=False)
        self.register_buffer("raw_fid_to_vstart", raw_fid_to_vstart, persistent=False)

        # a function, make it more/less senstiive to time
        def frame_to_tid_fn(frame_id):
            if torch.is_tensor(frame_id):
                device = frame_id.device
            else:
                frame_id = torch.tensor(frame_id)
                device = "cpu"
            frame_id = frame_id.to(self.raw_fid_to_vid.device)
            vid_len = self.raw_fid_to_vidlen[frame_id.long()]
            tid_sub = frame_id - self.raw_fid_to_vstart[frame_id.long()]
            tid = (tid_sub - vid_len / 2) / max_ts * 2  # [-1, 1]
            tid = tid * time_scale
            tid = tid.to(device)
            return tid

        self.frame_to_tid = frame_to_tid_fn


class TimeEmbedding(TimeInfo):
    """A learnable feature embedding per frame

    Args:
        num_freq_t (int): Number of frequencies in time embedding
        frame_info (Dict): Metadata about the frames in a dataset
        out_channels (int): Number of output channels
    """

    def __init__(self, num_freq_t, frame_info, out_channels=128, time_scale=1.0):
        super().__init__(frame_info, time_scale=time_scale)
        self.fourier_embedding = PosEmbedding(1, num_freq_t)
        t_channels = self.fourier_embedding.out_channels
        self.out_channels = out_channels

        self.mapping = nn.Linear(t_channels, out_channels)

    def forward(self, frame_id=None):
        """
        Args:
            frame_id: (...,) Frame id to evaluate at, or None to use all frames
        Returns:
            t_embed (..., self.W): Output time embeddings
        """
        # pylint: disable=C2801
        device = self.parameters().__next__().device
        if frame_id is None:
            inst_id, t_sample = self.frame_to_vid, self.frame_to_tid(self.frame_mapping)
        else:
            if not torch.is_tensor(frame_id):
                frame_id = torch.tensor(frame_id, device=device)
            inst_id = self.raw_fid_to_vid[frame_id.long()]
            t_sample = self.frame_to_tid(frame_id)

        if inst_id.ndim == 1:
            inst_id = inst_id[..., None]  # (N, 1)
            t_sample = t_sample[..., None]  # (N, 1)

        coeff = self.fourier_embedding(t_sample)
        t_embed = self.mapping(coeff)
        return t_embed

    def get_mean_embedding(self, device):
        """Compute the mean time embedding over all frames

        Args:
            device (torch.device): Output device
        """
        t_embed = self.forward(self.frame_mapping).mean(0, keepdim=True)
        if t_embed.device != device:
            t_embed = t_embed.to(device)
        return t_embed


class PosEmbedding(nn.Module):
    """A Fourier embedding that maps x to (x, sin(2^k x), cos(2^k x), ...)
    Adapted from https://github.com/kwea123/nerf_pl/blob/master/models/nerf.py

    Args:
        in_channels (int): Number of input channels (3 for both xyz, direction)
        n_freqs (int): Number of frequency bands
        logscale (bool): If True, construct frequency bands in log-space
    """

    def __init__(self, in_channels, n_freqs, logscale=True):
        super().__init__()
        self.n_freqs = n_freqs
        self.in_channels = in_channels
        self.logscale = logscale

        # no embedding
        if n_freqs == -1:
            self.out_channels = 0
            return

        self.funcs = [torch.sin, torch.cos]
        self.nfuncs = len(self.funcs)
        self.out_channels = in_channels * (len(self.funcs) * n_freqs + 1)

        self.set_alpha(None)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_freq_bands(self, device: str = "cuda") -> Float[Tensor, "n_freqs"]:
        """Get the frequency bands

        Returns:
            freq_bands: (n_freqs,) Frequency bands
        """
        if self.logscale:
            freq_bands = 2 ** torch.linspace(
                0, self.n_freqs - 1, self.n_freqs, device=device
            )
        else:
            freq_bands = torch.linspace(
                1, 2 ** (self.n_freqs - 1), self.n_freqs, device=device
            )
        return freq_bands

    def set_alpha(self, alpha):
        """Set the alpha parameter for the annealing window

        Args:
            alpha (float or None): 0 to 1
        """
        self.alpha = alpha

    def forward(self, x):
        """Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Args:
            x: (B, self.in_channels)
        Returns:
            out: (B, self.out_channels)
        """
        if self.n_freqs == -1:
            return torch.zeros_like(x[..., :0])

        # cosine features
        if self.n_freqs > 0:
            shape = x.shape
            device = x.device
            input_dim = shape[-1]
            output_dim = input_dim * (1 + self.n_freqs * self.nfuncs)
            out_shape = shape[:-1] + ((output_dim),)

            # assign input coordinates to the first few output channels
            x = x.reshape(-1, input_dim)
            out = torch.empty(x.shape[0], output_dim, dtype=x.dtype, device=device)
            out[:, :input_dim] = x

            # assign fourier features to the remaining channels
            out_bands = out[:, input_dim:].view(
                -1, self.n_freqs, self.nfuncs, input_dim
            )
            freq_bands = self.get_freq_bands(device=device)
            for i, func in enumerate(self.funcs):
                # (B, nfreqs, input_dim) = (1, nfreqs, 1) * (B, 1, input_dim)
                out_bands[:, :, i] = func(freq_bands[None, :, None] * x[:, None, :])

            self.apply_annealing(out_bands)

            out = out.view(out_shape)
        else:
            out = x
        return out

    def apply_annealing(self, out_bands):
        """Apply the annealing window w = 0.5*( 1+cos(pi + pi clip(alpha-j)) )

        Args:
            out_bands: (..., n_freqs, nfuncs, in_channels) Frequency bands
        """
        device = out_bands.device
        if self.alpha is not None:
            alpha_freq = self.alpha * self.n_freqs
            window = alpha_freq - torch.arange(self.n_freqs).to(device)
            window = torch.clamp(window, 0.0, 1.0)
            window = 0.5 * (1 + torch.cos(np.pi * window + np.pi))
            window = window.view(1, -1, 1, 1)
            out_bands[:] = window * out_bands

    def get_mean_embedding(self, device):
        """Compute the mean Fourier embedding

        Args:
            device (torch.device): Output device
        """
        mean_embedding = torch.zeros(self.out_channels, device=device)
        return mean_embedding
