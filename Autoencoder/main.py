import matplotlib.pyplot as plt
import torch
from config import get_datetime_for_filename
from data.data_utilities import load_datasets
from Autoencoder.models.LinearAutoEncoder import LinearAutoEncoder
import torch.optim as optim

hidden_size = 2048

model = LinearAutoEncoder(hidden_size=hidden_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 100
batch_size = 2
percentage_of_data_to_keep = 1
number_of_data_to_keep = 10
train_dataloader, test_dataloader = load_datasets(data_num=5,
                                                  percentage_of_data_to_keep=0.5,
                                                  return_loader=True,
                                                  number_of_data_to_keep=100)
# loss_function = torch.nn.MSELoss()
# BCELoss thelei na exeis [0,1] ta images
# opote sto last decoder step ebala mia sigmoid
loss_function = torch.nn.BCELoss()
losses = []

for i in range(epochs):
    for image,idx in train_dataloader:
        optimizer.zero_grad()

        image = image.reshape(-1, 32 * 32)
        prediction = model(image)

        loss = loss_function(prediction, image)

        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())

        if i==epochs-1:
            image = image.reshape(-1, 32, 32)
            plt.imshow(image.T)

            time = get_datetime_for_filename()
            plt.savefig(f"images\\og_img_v2{time}.png")
            prediction = prediction.detach().numpy().reshape(-1, 32, 32)
            plt.imshow(prediction.T)
            plt.savefig(f"images\\reconstructed_v2{time}.png")

            plt.imshow(image.permute([1, 2, 0]))
            plt.savefig(f"images\\maybe_better_v2{time}.png")


plt.clf()
plt.plot(losses)
plt.savefig(f"training_results\\losses{time}.png")