import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from config import get_datetime_for_filename
from data.data_utilities import load_datasets
from Autoencoder.models.ConvolutionalAutoEncoder import CnnAutoEncoder
import torch.optim as optim
import time
epochs = 400
batch_size = 64
# loss_function = torch.nn.MSELoss()
# BCELoss thelei na exeis [0,1] ta images
# opote sto last decoder step ebala mia sigmoid
def recostruction_loss(x, x_hat):
    loss = F.mse_loss(x, x_hat, reduction="none")
    # loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
    return loss

#loss_function = F.mse_loss
loss_function = torch.nn.BCELoss()

use_previous_model = False
hidden_size = 2048
model = CnnAutoEncoder()
model_path = "autoencoder.h5"
f = open("autoencoder_convolutional_results.txt", "a")
f.write(f'Epochs:{epochs}, loss func:{loss_function._get_name()}\n')
#f.write(f'Epochs:{epochs}, loss func:mse_loss\n')
experiment_name = "relu"
f.write(f'{experiment_name}\n')

start_time = time.time()

optimizer = optim.Adam(model.parameters(), lr=0.001)

if use_previous_model:
    print("using pretrained model...")
    model.load_state_dict(torch.load(model_path))



train_dataloader, test_dataloader = load_datasets(data_num=5,
                                                  batch_size = batch_size,
                                                  percentage_of_data_to_keep=0.5,
                                                  return_loader=True,
                                                  number_of_data_to_keep=100)

losses = []

time_for_files = get_datetime_for_filename()
f.write(time_for_files)
experiments_name_folder = f'images\\cnn\\{experiment_name}_{time_for_files}\\'
os.makedirs(experiments_name_folder, exist_ok=True)

for i in range(epochs):
    for image,idx in train_dataloader:
        optimizer.zero_grad()

        prediction = model(image)

        loss = loss_function(prediction, image)
        loss.backward()
        optimizer.step()

        if i%10==0:
            print(f'Epoch:{i}: loss={loss.detach().item()}')
            losses.append(loss.detach().item()/10)

        # sto teleutaio epoch kanoume print ti eftiaxe o autoencoder
        if i==epochs-1:
            # compare_imgs(image, prediction)

            image = image[0].reshape(-1, 32, 32)
            plt.imshow(image.T)
            plt.savefig(f"{experiments_name_folder}og_img_v2.png")
            prediction = prediction.detach().numpy()[0].reshape(-1, 32, 32)
            plt.imshow(prediction.T)
            plt.savefig(f"{experiments_name_folder}reconstructed_v2.png")

            # plt.imshow(image.permute([1, 2, 0]))
            # plt.savefig(f"{experiments_name_folder}maybe_better_v2.png")


# note down training time
training_time = time.time() - start_time
training_time_content = f"--- {training_time} seconds ---\n"
f.write(training_time_content)

plt.clf()
plt.plot(losses)
plt.savefig(f"{experiments_name_folder}loss.png")


# we save trained model
torch.save(model.state_dict(),model_path)