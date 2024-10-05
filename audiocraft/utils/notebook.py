# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

try:
    import IPython.display as ipd  # type: ignore
except ImportError:
    # Note in a notebook...
    pass


import torch


import IPython.display as ipd

def display_audio(samples: torch.Tensor, sample_rate: int, descriptions: list = None):
    """Renders an audio player for the given audio samples, with descriptions above each player.

    Args:
        samples (torch.Tensor): a Tensor of decoded audio samples
            with shapes [B, C, T] or [C, T]
        sample_rate (int): sample rate audio should be displayed with.
        descriptions (list): a list of descriptions (titles) for each audio player.
            Length of descriptions should match the number of audio samples.
    """
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    if descriptions is not None:
        assert len(descriptions) == samples.size(0), "Descriptions length must match number of audio samples."

    for i, audio in enumerate(samples):
        if descriptions is not None:
            print(f"Title: {descriptions[i]}")
        ipd.display(ipd.Audio(audio, rate=sample_rate))