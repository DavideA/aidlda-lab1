import torch


class myRelu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):

        # save
        ctx.save_for_backward(x)
        return torch.clamp(x, min=0)

    @staticmethod
    def backward(ctx, grad_outputs):
        x, = ctx.saved_tensors

        grad_inputs = (x >= 0).float() * grad_outputs
        return grad_inputs
