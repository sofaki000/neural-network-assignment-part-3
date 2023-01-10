from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from VisionTransformer.MyEncDec import Encoder,Decoder,AutoEncoder
from VisionTransformer.cifar10Utilities import  get_cifar_loaders
from config import get_datetime_for_filename
from data.data_utilities import get_dataset_for_developing, load_datasets

if __name__ == '__main__':
    #train_loader = get_cifar_loaders(return_only_train=True)
    train_dataloader, test_dataloader = load_datasets(data_num=5,
                                                      percentage_of_data_to_keep=0.5,
                                                      return_loader=True,
                                                      number_of_data_to_keep=100)

    enc = Encoder()
    dec = Decoder()
    model = AutoEncoder(enc, dec)

    losses = []
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    lambda_group1 = lambda epoch: epoch // 30
    # scheduler = LambdaLR(optimizer, lr_lambda=[lambda_group1])

    epochs = 200
    loss_function = torch.nn.MSELoss()

    for i in range(epochs):
        epoch_loss = 0
        for image, idx in train_dataloader:
            optimizer.zero_grad()
            prediction = model(image)
            loss = loss_function(prediction, image)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()

        # epoch finished
        if i == epochs - 1: # sto teleutaio epoch trabame mia photo
            fig, axarr = plt.subplots(1, 2)
            image = image.reshape(-1, 32, 32)
            prediction = prediction.detach().numpy().reshape(-1, 32, 32)
            time = get_datetime_for_filename()
            axarr[0].imshow(image.T)
            axarr[1].imshow(prediction.T)
            plt.savefig(f"reconstructed{time}.png")

        epoch_loss = epoch_loss/len(train_dataloader)
        losses.append(epoch_loss)
        print(f'Loss:{epoch_loss}, epoch:{i}')


    plt.clf()
    plt.plot(losses)
    plt.savefig(f"loss_with_scheduler{time}.png")