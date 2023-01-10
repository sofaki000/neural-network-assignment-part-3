import torch

def get_random_mask():
    # the mask should have the size of the original image
    # 3 rgb channels , 1024 (32*23) pixels
    # we want 784 (28*28) untouched pixels
    unmasked = torch.ones(784)

    # the masked pixels
    masked = torch.zeros(240)

    mask = torch.concatenate((masked, unmasked))

    # we get random indexes from the mask to shuffle masked and unmasked pixels
    idx = torch.randperm(mask.shape[0])

    mask_shuffled = mask[idx]

    return mask_shuffled