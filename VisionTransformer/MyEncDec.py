from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import torch.nn as nn
from VisionTransformer.cifar10Utilities import image_to_patch, get_cifar_loaders
from VisionTransformer.tutorial.model import AttentionBlock

num_channels = 3
patch_size = 4
embed_dim = 192
hidden_dim = 128
num_heads = 3
dropout = 0.2
num_layers = 6

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(num_channels * (patch_size ** 2), embed_dim)
        self.transformer = nn.Sequential(
            *[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout)
              for _ in range(num_layers)])
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, input):
        patch_input = image_to_patch(input, patch_size)

        lin_output = self.input_layer(patch_input)

        transformed = self.transformer(lin_output) #[batch_size, 64, 192]

        return self.layer_norm(transformed)

def patch_to_image(patch):
    batch_size = patch.size(0)
    image = patch.reshape(batch_size,3,32,32)
    return image

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Sequential(
            *[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout)
              for _ in range(num_layers)])
        self.linear = nn.Linear(in_features=embed_dim, out_features=3*(patch_size)**2)

    def forward(self, input):

        # [batch_size, t, c]
        transformed = self.transformer(input)

        decoded = self.linear(transformed)

        image = patch_to_image(decoded)

        return image


class AutoEncoder(nn.Module):
    def __init__(self, enc, dec):

        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, input):
        encoded = self.enc(input)

        return self.dec(encoded)


if __name__ == '__main__':
    train_loader  = get_cifar_loaders(return_only_train=True)


    enc = Encoder()
    dec = Decoder()
    model = AutoEncoder(enc, dec)

    losses = []
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    lambda_group1 = lambda epoch: epoch // 30
    scheduler = LambdaLR(optimizer, lr_lambda=[lambda_group1])

    epochs = 5
    loss_function = torch.nn.MSELoss()

    fake_batch = torch.rand(64,3,32,32)
    for i in range(epochs):
        for image,idx in train_loader:
            #image = fake_batch
            optimizer.zero_grad()
            prediction = model(image)
            loss = loss_function(prediction, image)

            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())


            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            print(f'Loss:{loss.detach().item()}, lr:{lr}')


    plt.clf()
    plt.plot(losses)
    plt.savefig(f"loss.png")