import torch


class mySoftmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):

        N = torch.exp(x)
        D = torch.sum(N, dim=1, keepdim=True)

        output = N / D

        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_outputs):

        output, = ctx.saved_tensors
        _, d = output.shape
        grad_x = None

        J = []
        for idx, o in enumerate(output):
            J.append(- torch.ger(o, o) * (1 - torch.eye(d).to('cuda')) + torch.diag(o * (1-o)))
        J = torch.stack(J)

        grad_x = []
        for jac, grad in zip(J, grad_outputs):
            grad_x.append(jac.mm(grad.unsqueeze(1)).squeeze(1))

        grad_x = torch.stack(grad_x)
        return grad_x
