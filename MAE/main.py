import os
import time
import matplotlib.pyplot as plt
import torch
from MAE.masks.mask_utilities import get_mask_for_1024_pixels, show_two_images
from MAE.masks.random_patches import get_random_mask
from MAE.model import MAE
from config import get_datetime_for_filename
from data.data_utilities import load_datasets
import torch.optim as optim

hidden_size = 2048

model = MAE()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 300
batch_size = 1
loss_function = torch.nn.MSELoss() #torch.nn.BCELoss()
f = open("mae_results.txt", "a")
f.write(f'Epochs:{epochs}, loss func:{loss_function._get_name()}\n')
start_time = time.time()
experiment_name = "pooling_layers_only"
f.write(f'{experiment_name}\n')
time_for_files = get_datetime_for_filename()
f.write(time_for_files)
experiments_name_folder = f'images\\{experiment_name}_{time_for_files}\\'
os.makedirs(experiments_name_folder, exist_ok=True)
train_dataloader, test_dataloader = load_datasets(data_num=5,
                                                  batch_size = batch_size,
                                                  percentage_of_data_to_keep=0.5,
                                                  return_loader=True,
                                                  number_of_data_to_keep=100)
# mask = torch.randint(0,2, (32*32,)) # poia pixels kratame poia oxi
# mask =  get_random_mask() #get_mask_for_1024_pixels()

def remove_masked_pixels(input, mask):
    pixels = input.view(3, -1) # edw exoume 1024 pixels
    # to kanoume transpose wste to pixels[0] na anaparista oti input_features exei to prwto pixel
    pixels = pixels.transpose(1,0)

    # anti na pairnoume to mask apo edw, to mask prepei na einai GENIKO
    # mask = pixels.sum(dim=1) > 0.5 # an oi times enos pixel exoun athroisma panw apo 0.5 tis kratame

    indices_to_keep = torch.nonzero(mask)

    pixels_we_want = pixels[indices_to_keep].squeeze(1)

    return pixels_we_want


losses = []

# mask = get_mask_for_pixels_greater_than(image, 0.2)

counter = 0
took_photo = False
for i in range(epochs):
    for image,idx in train_dataloader:

        mask = get_random_mask()

        optimizer.zero_grad()

        masked_img = remove_masked_pixels(image, mask).reshape(3, 28, 28) # 3*28*28 (ta ligotera pixels,xwris ta masked pixels)

        prediction = model(masked_img, mask)

        loss = loss_function(prediction, image)

        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())

        if counter < 3:
            # take some pics for the random mask
            masked_img = (image.view(3,-1) * mask).reshape(3,32,32)
            show_two_images(image.squeeze(0).T, masked_img.T, img_name=f"random_mask{counter}")
            counter+=1
            break
        if i==epochs-1 and took_photo is False:
            took_photo = True
            fig, axarr = plt.subplots(1, 2)
            image = image.reshape(-1, 32, 32)
            prediction = prediction.detach().numpy().reshape(-1, 32, 32)

            axarr[0].imshow(image.T)
            axarr[1].imshow(prediction.T)
            plt.savefig(f"{experiments_name_folder}reconstructed.png")



# note down training time
training_time = time.time() - start_time
training_time_content = f"--- {training_time} seconds ---\n"
f.write(training_time_content)

plt.clf()
plt.plot(losses)
plt.savefig(f"{experiments_name_folder}loss.png")

