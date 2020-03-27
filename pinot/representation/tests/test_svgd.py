import pinot
import torch

def test_min():
    x = torch.tensor(torch.distributions.normal.Normal(
            loc=torch.zeros((128, 64)),
            scale=torch.ones((128, 64))).sample(), requires_grad=True)

    def l(y):
        return torch.sum((y-torch.tensor(1)) ** 2 + (y-torch.tensor(5)) ** 2)

    opt = torch.optim.Adam([x], 1e-1)

    for _ in range(1000):
        loss = l(x)
        
        opt.zero_grad()
        loss.backward()
        
        x.grad = pinot.models.svgd.svgd_grad(x, x.grad)
        
        print(loss)
        print(x)

        opt.step()

test_min()
