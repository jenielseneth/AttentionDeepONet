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
        p = number of basis functions
        m = number of function evaluations
        n = number of points where the output is evaluated
        d = dimension of the input points
        """
        super(DeepONet, self).__init__()
        self.branch_net = BranchNet(p, m, n, activation)
        self.trunk_net = TrunkNet(p, d, activation)

    def forward(self, u, y):
        b, m, c = u.shape
        b, n, d = y.shape
        branch_output = self.branch_net(u)  # Should be of shape (b, p)
        trunk_output = self.trunk_net(y)  # Should be of shape (b, n, p)
        return torch.einsum("bp,bnp->bn", branch_output, trunk_output)


class MixDeepONet(torch.nn.Module):
    def __init__(self, p, m, n, d, activation):
        """
        p = number of basis functions
        m = number of function evaluations
        n = number of points where the output is evaluated
        d = dimension of the input points
        """
        super(MixDeepONet, self).__init__()
        self.branch_net = BranchNet(p, m, n, activation)
        self.trunk_net = TrunkNet(p, d, activation)
        self.mix_tensor = torch.nn.Parameter(torch.randn(p, p, p))

    def forward(self, u, y):
        b, m, c = u.shape
        b, n, d = y.shape
        branch_output = self.branch_net(u)  # Should be of shape (b, p)

        mid_output = torch.einsum(
            "bp,ijp->bp", branch_output, self.mix_tensor
        )  # (b, p)

        trunk_output = self.trunk_net(y)  # Should be of shape (b, n, p)
        return torch.einsum("bp,bnp->bn", mid_output, trunk_output)


if __name__ == "__main__":
    p = 16
    m = 64
    n = 64
    d = 1
    activation = torch.nn.ReLU()
    model = DeepONet(p, m, n, d, activation)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {pytorch_total_params}")
    u = torch.randn(32, m, n)  # batch size of 32
    y = torch.randn(32, n, d)  # batch size of 32
    output = model(u, y)
    print(output.shape)  # Should print torch.Size([32, 64])

    model = MixDeepONet(p, m, n, d, activation)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {pytorch_total_params}")
    u = torch.randn(32, m, n)  # batch size of 32
    y = torch.randn(32, n, d)  # batch size of 32
    output = model(u, y)
    print(output.shape)  # Should print torch.Size([32, 64])
