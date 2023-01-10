import torchvision.transforms as transforms
from keras.utils import np_utils
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
import numpy as np

import config

cifar_data_fila_path = config.cifar_path
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_cifar(data_num=5, num_features=3072):
    X_train, y_train, X_test, y_test = load_datasets(data_num=5)
    X_train = X_train.reshape(-1, num_features)
    X_test = X_test.reshape(-1, num_features)
    pca = PCA(0.9)
    # train_img_pca = pca.fit_transform(X_train)
    # test_img_pca = pca.transform(X_test)
    # y_train = np_utils.to_categorical(y_train)
    # y_test = np_utils.to_categorical(y_test)

    return X_train, X_test, y_train, y_test


def get_cifar_binary(data_num=5, percentage_of_data_to_keep =0.5, num_features =3072):
    X_train, y_train, X_test, y_test = load_datasets(data_num=data_num, percentage_of_data_to_keep=percentage_of_data_to_keep)

    a_class = 5
    b_class = 4

    for i in np.array([i for i, x in enumerate(y_train) if y_train[i] == a_class]):
        y_train[i] = 1
    for i in np.array([i for i, x in enumerate(y_train) if y_train[i] == b_class ]):
        y_train[i] = -1

    for i in np.array([i for i, x in enumerate(y_test) if y_test[i] == a_class]):
        y_test[i] = 1
    for i in np.array([i for i, x in enumerate(y_test) if y_test[i] == b_class ]):
        y_test[i] = -1


    for i in range(len(X_train)):
        X_train[i] = X_train[i].flatten().reshape(-1, num_features)

    for i in range(len(X_test)):
        X_test[i]= X_test[i].flatten().reshape(-1, num_features)

    num_samples_train = len(X_train)
    X_train = np.asarray(X_train).reshape(num_samples_train, num_features)

    num_samples_test = len(X_test)
    X_test = np.asarray(X_test).reshape(num_samples_test, num_features)

    y_train = np.asarray(y_train)
    y_test =np.asarray(y_test)

    print(f'Number of A class on train ds: {np.count_nonzero(np.array(y_train)==1)}')
    print(f'Number of A class on test ds: {np.count_nonzero(np.array(y_test)==1)}')
    print(f'Number of B class on train ds: {np.count_nonzero(np.array(y_train)==-1)}')
    print(f'Number of B class on test ds: {np.count_nonzero(np.array(y_test)==-1)}')

    return X_train, X_test, y_train, y_test


def get_dataset_for_developing(transform, percentage_of_data_to_keep=0.5, number_of_data_to_keep=100):
    ds = CIFAR10(cifar_data_fila_path, train=True, download=False , transform=transform)
    dog_indices, deer_indices, airplane_indices, automobile_indices, ship_indices, truck_indices, bird_indices, frog_indices, horse_indices, cat_indices = [], [], [], [], [], [], [], [], [], []
    dog_idx, deer_idx = ds.class_to_idx['dog'], ds.class_to_idx['deer']
    airplane_idx, automobile_idx = ds.class_to_idx['airplane'], ds.class_to_idx['automobile']
    ship_idx, truck_idx = ds.class_to_idx['ship'], ds.class_to_idx['truck']
    bird_idx, frog_idx = ds.class_to_idx['bird'], ds.class_to_idx['frog']
    horse_idx, cat_idx = ds.class_to_idx['horse'], ds.class_to_idx['cat']

    counter = 0
    for i in range(len(ds)):
        counter+=1
        current_class = ds[i][1]
        if current_class == dog_idx:
            dog_indices.append(i)
        elif current_class == deer_idx:
            deer_indices.append(i)
        elif current_class == airplane_idx:
            airplane_indices.append(i)
        elif current_class == automobile_idx:
            automobile_indices.append(i)
        elif current_class == ship_idx:
            ship_indices.append(i)
        elif current_class == truck_idx:
            truck_indices.append(i)
        elif current_class == bird_idx:
            bird_indices.append(i)
        elif current_class == frog_idx:
            frog_indices.append(i)
        elif current_class == horse_idx:
            horse_indices.append(i)
        elif current_class == cat_idx:
            cat_indices.append(i)

        if number_of_data_to_keep==counter:
            break

    dog_indices = dog_indices[:int(percentage_of_data_to_keep * len(dog_indices))]
    deer_indices = deer_indices[:int(percentage_of_data_to_keep * len(deer_indices))]


    dog_indices_test = dog_indices[int(percentage_of_data_to_keep * len(dog_indices)):]
    deer_indices_test = deer_indices[int(percentage_of_data_to_keep * len(deer_indices)):]
    # airplane_indices = airplane_indices[:int(percentage_of_data_to_keep * len(airplane_indices))]
    # automobile_indices = automobile_indices[:int(percentage_of_data_to_keep * len(automobile_indices))]
    # ship_indices = ship_indices[:int(percentage_of_data_to_keep * len(ship_indices))]
    # truck_indices = truck_indices[:int(percentage_of_data_to_keep * len(truck_indices))]
    # bird_indices = bird_indices[:int(percentage_of_data_to_keep * len(bird_indices))]
    # frog_indices = frog_indices[:int(percentage_of_data_to_keep * len(frog_indices))]
    # horse_indices = horse_indices[:int(percentage_of_data_to_keep * len(horse_indices))]
    # cat_indices = cat_indices[:int(0.5 * len(cat_indices))]
    new_dataset = Subset(ds, dog_indices + deer_indices)
    test_dataset = Subset(ds, dog_indices_test + deer_indices_test)

    #+ airplane_indices + automobile_indices + ship_indices + truck_indices + bird_indices + frog_indices + horse_indices + cat_indices)
    return new_dataset,test_dataset


def load_datasets(data_num=5, batch_size = 1, percentage_of_data_to_keep=0.5,return_loader=True,number_of_data_to_keep=100):
    # transform = transforms.Compose([transforms.Resize((32, 32)),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0,), (1,)), nn.Flatten()])
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0,), (1,))])

    trainset,testset = get_dataset_for_developing(transform,
                                                  percentage_of_data_to_keep=percentage_of_data_to_keep,
                                                  number_of_data_to_keep=number_of_data_to_keep)
    # Define indexes and get the subset random sample of each.

    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    if return_loader:
        return (train_dataloader, test_dataloader)
    # Convert data to tensors.
    x_test = []
    y_test = []
    x_train = []
    y_train = []
    load_for_development = True

    if load_for_development:  # we dont load the whole cifar
        counter = 0
        for idx, (data, tar) in enumerate(test_dataloader):

            x_test.append(data.squeeze().numpy())
            y_test.append(tar.squeeze().item())
            counter += 1

            if counter == data_num:
                break

        counter = 0
        for idx, (data, tar) in enumerate(train_dataloader):
            x_train.append(data.squeeze().numpy())
            y_train.append(tar.squeeze().item())
            counter = counter + 1
            if counter == data_num:
                break
    # else:
    #     for idx, (data, tar) in enumerate(test_dataloader):
    #         x_test = data.squeeze()
    #         y_test = tar.squeeze()
    #     for idx, (data, tar) in enumerate(train_dataloader):
    #         x_train = data.squeeze()
    #         y_train = tar.squeeze()

    return x_train, y_train, x_test, y_test


def get_train_cifar_data_quick(data_num=10, train_data_percentage=0.7):
    from torchvision.datasets import CIFAR10
    from torch.utils.data import Subset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)

    # ds = CIFAR10('~/.torch/data/', train=True, download=True)
    ds = CIFAR10(cifar_data_fila_path, train=True, download=False, transform=transform)
    dog_indices, deer_indices, other_indices = [], [], []
    plane_indices, car_indices = [], []
    frog_indices, ship_indices = [], []
    dog_idx, deer_idx = ds.class_to_idx['dog'], ds.class_to_idx['deer']
    plane_idx, car_idx = ds.class_to_idx['airplane'], ds.class_to_idx['automobile']
    frog_idx, ship_idx = ds.class_to_idx['frog'], ds.class_to_idx['ship']

    counter = 0
    for i in range(len(ds)):
        counter+=1
        current_class = ds[i][1]
        if current_class == dog_idx:
            dog_indices.append(i)
        elif current_class == deer_idx:
            deer_indices.append(i)
        elif current_class == plane_idx:
            plane_indices.append(i)
        elif current_class == car_idx:
            car_indices.append(i)
        elif current_class == frog_idx:
            frog_indices.append(i)
        elif current_class == ship_idx:
            ship_indices.append(i)
        else:
            other_indices.append(i)
        if counter== data_num:
            break

    dog_split_indices = (int)(len(dog_indices) * train_data_percentage)
    # dog_indices_train = dog_indices[:int(0.5* len(dog_indices))]
    # dog_indices_test = dog_indices[int(0.5 * len(dog_indices)):int(0.8 * len(dog_indices))]
    dog_indices_train = dog_indices[:dog_split_indices]
    dog_indices_test = dog_indices[dog_split_indices:]

    plane_split_indices = (int)(len(plane_indices) * train_data_percentage)
    plane_indices_train = deer_indices[:plane_split_indices]
    plane_indices_test = deer_indices[plane_split_indices:]

    deer_split_indices = (int)(len(deer_indices) * train_data_percentage)
    deer_indices_train = deer_indices[:deer_split_indices]
    deer_indices_test = deer_indices[deer_split_indices:]

    ship_split_indices = (int)(len(ship_indices) * train_data_percentage)
    ship_indices_train = ship_indices[:ship_split_indices]
    ship_indices_test = ship_indices[ship_split_indices:]

    frog_split_indices = (int)(len(frog_indices) * train_data_percentage)
    frog_indices_train = frog_indices[:frog_split_indices]
    frog_indices_test = frog_indices[frog_split_indices:]

    car_split_indices = (int)(len(car_indices) * train_data_percentage)
    car_indices_train = car_indices[:car_split_indices]
    car_indices_test = car_indices[car_split_indices:]

    train_dataset = Subset(ds, dog_indices_train + deer_indices_train)  # + other_indices
    # test dataset
    test_dataset = Subset(ds, dog_indices_test + deer_indices_test)  # + other_indices

    print(f'Number of set: {len(dog_indices) + len(deer_indices)}')

    return train_dataset, test_dataset


def get_train_cifar_data(train_data_percentage=0.7):
    from torchvision.datasets import CIFAR10
    from torch.utils.data import Subset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)

    # ds = CIFAR10('~/.torch/data/', train=True, download=True)
    ds = CIFAR10(cifar_data_fila_path, train=True, download=False, transform=transform)
    dog_indices, deer_indices, other_indices = [], [], []
    plane_indices, car_indices = [], []
    frog_indices, ship_indices = [], []
    dog_idx, deer_idx = ds.class_to_idx['dog'], ds.class_to_idx['deer']
    plane_idx, car_idx = ds.class_to_idx['airplane'], ds.class_to_idx['automobile']
    frog_idx, ship_idx = ds.class_to_idx['frog'], ds.class_to_idx['ship']

    for i in range(len(ds)):
        current_class = ds[i][1]
        if current_class == dog_idx:
            dog_indices.append(i)
        elif current_class == deer_idx:
            deer_indices.append(i)
        elif current_class == plane_idx:
            plane_indices.append(i)
        elif current_class == car_idx:
            car_indices.append(i)
        elif current_class == frog_idx:
            frog_indices.append(i)
        elif current_class == ship_idx:
            ship_indices.append(i)
        else:
            other_indices.append(i)

    dog_split_indices = (int)(len(dog_indices) * train_data_percentage)
    # dog_indices_train = dog_indices[:int(0.5* len(dog_indices))]
    # dog_indices_test = dog_indices[int(0.5 * len(dog_indices)):int(0.8 * len(dog_indices))]
    dog_indices_train = dog_indices[:dog_split_indices]
    dog_indices_test = dog_indices[dog_split_indices:]

    plane_split_indices = (int)(len(plane_indices) * train_data_percentage)
    plane_indices_train = deer_indices[:plane_split_indices]
    plane_indices_test = deer_indices[plane_split_indices:]

    deer_split_indices = (int)(len(deer_indices) * train_data_percentage)
    deer_indices_train = deer_indices[:deer_split_indices]
    deer_indices_test = deer_indices[deer_split_indices:]

    ship_split_indices = (int)(len(ship_indices) * train_data_percentage)
    ship_indices_train = ship_indices[:ship_split_indices]
    ship_indices_test = ship_indices[ship_split_indices:]

    frog_split_indices = (int)(len(frog_indices) * train_data_percentage)
    frog_indices_train = frog_indices[:frog_split_indices]
    frog_indices_test = frog_indices[frog_split_indices:]

    car_split_indices = (int)(len(car_indices) * train_data_percentage)
    car_indices_train = car_indices[:car_split_indices]
    car_indices_test = car_indices[car_split_indices:]

    train_dataset = Subset(ds,
                           dog_indices_train + deer_indices_train + car_indices_train + frog_indices_train + ship_indices_train + plane_indices_train)  # + other_indices
    # test dataset
    test_dataset = Subset(ds,
                          dog_indices_test + deer_indices_test + car_indices_test + frog_indices_test + ship_indices_test + plane_indices_test)  # + other_indices

    print(
        f'Number of set: {len(dog_indices) + len(deer_indices) + len(frog_indices_train) + len(car_indices_train) + len(ship_indices_train) + len(plane_indices_train)}')

    return train_dataset, test_dataset