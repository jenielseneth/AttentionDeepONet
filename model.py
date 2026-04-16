import torch


class TrunkNet(torch.nn.Module):
    def __init__(self, p, d, activation):
        super(TrunkNet, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(p, d))
        self.z = torch.nn.Parameter(torch.randn(p))
        self.activation = activation

    def forward(self, y):
        """
        Take points y of shape (b, d) and return trunk output of shape (b, p)
        """
        n, d = y.shape  # n is the number of points, d is the input dimension
        return torch.matmul(y, self.W) + self.z


class BranchNet(torch.nn.Module):
    def __init__(self, p, m, n, activation):
        super(BranchNet, self).__init__()
        self.C = torch.nn.Parameter(torch.randn(p, n))
        self.Z = torch.nn.Parameter(torch.randn(p, n, m))
        self.Theta = torch.nn.Parameter(torch.randn(p, n))
        self.activation = activation

    def forward(self, u):
        """
        Take function evaluations u of shape (m,) and return branch output of shape (p, n)
        """
        m = u.shape
        return (
            self.activation((torch.einsum("m,pnm->pn", u, self.Z) + self.Theta))
            * self.C
        ).sum(
            dim=-1
        )  # Should return a tensor of shape (p,)


class DeepONet(torch.nn.Module):
    def __init__(self, p, m, n, d, activation):
        super(DeepONet, self).__init__()
        self.branch_net = BranchNet(p, m, n, activation)
        self.trunk_net = TrunkNet(p, d, activation)

    def forward(self, y, u):
        branch_output = self.branch_net(u)
        trunk_output = self.trunk_net(y)
        return torch.einsum("p,bp->b", branch_output, trunk_output)
