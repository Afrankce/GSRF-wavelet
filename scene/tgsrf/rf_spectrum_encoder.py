import torch
from torch import nn

from .encoding import fourier_encode, make_angle_grid


class RFSpectrumEncoder(nn.Module):
    """Encode a NeRF2-style angular RF spectrum into transformer tokens."""

    def __init__(self, hidden_dim=128, patch_stride=8, txrx_freqs=4):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.patch_stride = int(patch_stride)
        self.txrx_freqs = int(txrx_freqs)

        self.stem = nn.Sequential(
            nn.Conv2d(5, hidden_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=patch_stride, stride=patch_stride),
            nn.SiLU(inplace=True),
        )
        txrx_dim = 6 * (1 + 2 * self.txrx_freqs)
        self.txrx_proj = nn.Sequential(
            nn.Linear(txrx_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, spectrum, tx, rx):
        if spectrum.ndim == 2:
            spectrum = spectrum[None, None]
        elif spectrum.ndim == 3:
            spectrum = spectrum[:, None]
        if spectrum.shape[0] != 1:
            raise ValueError("TGS-RF V1 expects one RF spectrum at a time.")

        spectrum = spectrum.float()
        device, dtype = spectrum.device, spectrum.dtype
        _, _, height, width = spectrum.shape
        az, el, _ = make_angle_grid(height, width, device, dtype)
        angle = torch.stack(
            [torch.sin(az), torch.cos(az), torch.sin(el), torch.cos(el)],
            dim=0,
        ).unsqueeze(0)
        image = torch.cat([spectrum, angle], dim=1)

        feature_map = self.stem(image)
        tokens = feature_map.flatten(2).transpose(1, 2).contiguous()

        txrx = torch.cat([tx.reshape(1, 3), rx.reshape(1, 3)], dim=-1).to(device=device, dtype=dtype)
        txrx_emb = self.txrx_proj(fourier_encode(txrx, self.txrx_freqs))
        tokens = self.norm(tokens + txrx_emb[:, None, :])
        global_token = tokens.mean(dim=1)
        return {
            "tokens": tokens,
            "global": global_token,
            "feature_map": feature_map,
        }

