from typing import Union

import torch


class TrunkNet(torch.nn.Module):
    def __init__(self, p, d, activation):
        super(TrunkNet, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(p, d))
        self.z = torch.nn.Parameter(torch.randn(p))
        self.activation = activation

    def forward(self, y):
        """
        Take points y of shape (b, n, d) and return trunk output of shape (b, p)
        """
        b, n, d = (
            y.shape
        )  # b is the batch size, n is the number of points, d is the input dimension
        return torch.einsum("bnd,dp->bnp", y, self.W.T) + self.z.unsqueeze(
            0
        )  # Should return a tensor of shape (b, n, p)


class BranchNet(torch.nn.Module):
    def __init__(self, p, m, n, activation):
        super(BranchNet, self).__init__()
        self.C = torch.nn.Parameter(torch.randn(p, n))
        self.Z = torch.nn.Parameter(torch.randn(p, n, m))
        self.Theta = torch.nn.Parameter(torch.randn(p, n))
        self.activation = activation

    def forward(self, u):
        """
        Take function evaluations u of shape (b, m, c) and return branch output of shape (b, p)
        """
        b, m, c = u.shape
        branch_output = (
            self.activation(
                (torch.einsum("bmc,pnm->bpn", u, self.Z) + self.Theta.unsqueeze(0))
            )
            * self.C.unsqueeze(0)
        ).sum(dim=-1)
        return branch_output  # Should return a tensor of shape (b, p)


class DeepONet(torch.nn.Module):
    def __init__(self, p, m, n, d, activation):
        """
        p = number of basis functions (trunk coefficients t_k and branch coefficients b_k, k=1,...,p)
        m = number of function evaluations (u(x_j) for j=1,...,m)
        n = number of points where the output is evaluated (y_i for i=1,...,n)
        d = dimension of the input points (y_i in R^d)
        """
        super(DeepONet, self).__init__()
        self.branch_net = BranchNet(p, m, n, activation)
        self.trunk_net = TrunkNet(p, d, activation)

    def attention(self, u, y):
        b, m, c = u.shape
        b, n, d = y.shape
        branch_output = self.branch_net(u)  # Should be of shape (b, p)
        trunk_output = self.trunk_net(y)  # Should be of shape (b, n, p)
        return torch.einsum("bp,bnp->bnp", branch_output, trunk_output)

    def forward(self, u, y):
        return self.attention(u, y).sum(dim=-1)  # Final output of shape (b, n)


class MixDeepONet(torch.nn.Module):
    def __init__(self, p, m, n, d, activation):
        """
        p = number of basis functions (trunk coefficients t_k and branch coefficients b_k, k=1,...,p)
        m = number of function evaluations (u(x_j) for j=1,...,m)
        n = number of points where the output is evaluated (y_i for i=1,...,n)
        d = dimension of the input points (y_i in R^d)
        """
        super(MixDeepONet, self).__init__()
        self.branch_net = BranchNet(p, m, n, activation)
        self.trunk_net = TrunkNet(p, d, activation)
        self.mix_tensor = torch.nn.Parameter(torch.randn(p, p, p))

    def attention(self, u, y):
        b, m, c = u.shape
        b, n, d = y.shape
        branch_output = self.branch_net(u)  # Should be of shape (b, p)

        mid_output = torch.einsum(
            "ijp, bp->bij", self.mix_tensor, branch_output
        )  # (b, p, p)

        trunk_output = self.trunk_net(y)  # Should be of shape (b, n, p)
        return torch.einsum(
            "bni, bij ->bnj",
            trunk_output,
            mid_output,
        )  # Should return a tensor of shape (b, n, p)

    def forward(self, u, y):
        return self.attention(u, y).sum(dim=-1)  # Final output of shape (b, n)


class LayeredDeepONet(torch.nn.Module):
    def __init__(
        self,
        p,
        m,
        n,
        d,
        activation,
        num_layers,
        deeponet_cls: Union[DeepONet, MixDeepONet],
    ):
        """
        p = number of basis functions (trunk coefficients t_k and branch coefficients b_k, k=1,...,p)
        m = number of function evaluations (u(x_j) for j=1,...,m)
        n = number of points where the output is evaluated (y_i for i=1,...,n)
        d = dimension of the input points (y_i in R^d)
        num_layers = number of DeepONet layers
        deeponet_cls = class of the DeepONet layer (either DeepONet or MixDeepONet)
        """
        super(LayeredDeepONet, self).__init__()

        self.deeponet_layers = torch.nn.ModuleList(
            [deeponet_cls(p, m, n, d, activation)]
            + [deeponet_cls(p, m, n, p, activation) for _ in range(num_layers - 1)]
        )

    def forward(self, u, y):
        b, m, c = u.shape
        b, n, d = y.shape

        for i, layer in enumerate(self.deeponet_layers):
            y = layer.attention(u, y)  # Update y for the next layer

        return y.sum(dim=-1)  # Final output of shape (b, n)


if __name__ == "__main__":
    p = 16
    m = 64
    n = 64
    d = 1
    u = torch.randn(32, m, n)  # batch size of 32
    y = torch.randn(32, n, d)  # batch size of 32
    activation = torch.nn.ReLU()

    # Single Layer DeepONet
    model = DeepONet(p, m, n, d, activation)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {pytorch_total_params}")
    output = model(u, y)
    print(output.shape)  # Should print torch.Size([32, 64])

    # Single Layer MixDeepONet
    model = MixDeepONet(p, m, n, d, activation)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {pytorch_total_params}")
    output = model(u, y)
    print(output.shape)  # Should print torch.Size([32, 64])

    # Multiple Layer DeepONet
    num_layers = 3
    model = LayeredDeepONet(p, m, n, d, activation, num_layers, DeepONet)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {pytorch_total_params}")
    output = model(u, y)
    print(output.shape)  # Should print torch.Size([32, 64])
