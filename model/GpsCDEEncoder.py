import torch
import torch.nn as nn
import torchcde


## Continuity-preserving Encoding
class NeuralCDEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_adjoint=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_adjoint = use_adjoint

        self.linear = nn.Linear(hidden_dim, hidden_dim * input_dim)
        self.out_linear = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.z0_proj = nn.Linear(input_dim, hidden_dim)

    def cde_neural(self, t, z):
        out = self.linear(z)  # (batch, hidden_dim * input_dim)
        out = out.view(z.size(0), self.hidden_dim, self.input_dim)
        return out

    def forward(self, traj, traj_length):
        batch_size, max_seq_len, input_dim = traj.shape
        coeffs = torchcde.natural_cubic_spline_coeffs(traj)
        X = torchcde.CubicSpline(coeffs)
        integration_times = torch.linspace(0, 1, max_seq_len, device=traj.device)
        z0 = self.z0_proj(traj[:, 0, :])
        z = torchcde.cdeint(
            X=X,
            func=self.cde_neural,
            z0=z0,
            t=integration_times,
            method="euler",
            options={"step_size": 0.05},
            rtol=1e-5,
            atol=1e-5,
        )  # (Eq. 13 in paper)
        traj_repr = self.out_linear(z)  # (Eq. 14 in paper)
        return traj_repr
