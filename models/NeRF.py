import torch
import torch.nn as nn
import tinycudann as tcnn  # Import tcnn

# Positional embedding
class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        super(Embedding, self).__init__()

        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = self.in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(freq * x))
        return torch.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 in_channels_xyz=63,
                 in_channels_dir=27):
        """
        hidden_dim: number of hidden units
        in_channels_xyz: input channels for xyz (default: 63)
        in_channels_dir: input channels for direction (default: 27)
        """
        super(NeRF, self).__init__()

        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.hidden_dim = hidden_dim

        self.block1 = tcnn.Network(in_channels_xyz, hidden_dim, {
                                    "otype": "FullyFusedMLP",
                                    "activation": "ReLU",
                                    "output_activation": "ReLU",
                                    "n_neurons": hidden_dim,
                                    "n_hidden_layers": 4
                                })

        self.block2 = tcnn.Network(hidden_dim + in_channels_xyz, hidden_dim, {
                                    "otype": "FullyFusedMLP",
                                    "activation": "ReLU",
                                    "output_activation": "ReLU",
                                    "n_neurons": hidden_dim,
                                    "n_hidden_layers": 2
                                })

        # # Final MLP for xyz encoding
        # self.xyz_encoding_final = tcnn.Network(hidden_dim, hidden_dim, {
        #     "otype": "FullyFusedMLP",
        #     "activation": "None",
        #     "output_activation": "None",
        #     "n_neurons": hidden_dim,
        #     "n_hidden_layers": 1
        # })

        # # Direction encoding MLP
        # self.dir_network = tcnn.Network(hidden_dim + in_channels_dir, hidden_dim // 2, {
        #     "otype": "FullyFusedMLP",
        #     "activation": "ReLU",
        #     "output_activation": "None",
        #     "n_neurons": hidden_dim // 2,
        #     "n_hidden_layers": 1
        # })

        # Sigma (density) MLP
        self.sigma_network = tcnn.Network(hidden_dim, 1, {
            "otype": "FullyFusedMLP",
            "activation": "None",
            "output_activation": "None",
            "n_neurons": hidden_dim,
            "n_hidden_layers": 1
        })

        # RGB output MLP
        self.rgb_network = tcnn.Network(hidden_dim + in_channels_dir, 3, {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "Sigmoid",  # For RGB values
            "n_neurons": hidden_dim // 2,
            "n_hidden_layers": 1
        })

    def forward(self, x, sigma_only=False):
        """
        Forward pass through NeRF MLPs

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
            sigma_only: if True, only return sigma (density)

        Outputs:
            if sigma_only: sigma (B, 1)
            else: rgb (B, 3) and sigma (B, 1)
        """
        if not sigma_only:
            input_xyz, input_dir = torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        # Forward pass through xyz encoding with skip connections
        xyz_encoded = input_xyz
        xyz_encoded = self.block1(xyz_encoded)
        xyz_encoded = torch.cat([input_xyz, xyz_encoded], dim=-1)
        xyz_encoded = self.block2(xyz_encoded)

        sigma = self.sigma_network(xyz_encoded)
        if sigma_only:
            return sigma

        # Final layer of xyz encoding
        # xyz_encoding_final = self.xyz_encoding_final(xyz_encoded)

        # Concatenate with directional input
        dir_encoded = torch.cat([xyz_encoded, input_dir], dim=-1)

        # Forward pass through directional MLP
        # dir_encoded = self.dir_network(dir_encoding_input)

        # RGB output
        rgb = self.rgb_network(dir_encoded)

        # Concatenate RGB and sigma for final output
        out = torch.cat([rgb, sigma], dim=-1)

        return out
