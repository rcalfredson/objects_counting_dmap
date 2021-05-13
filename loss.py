import torch

# New loss functions
def meanAbsoluteError(result, label):
    # necessary to iterate explicitly? I think yes.
    # this will give the mean over
    return torch.mean(torch.cat([torch.sum()]))