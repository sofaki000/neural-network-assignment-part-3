
from torchvision import transforms

import matplotlib.pyplot as plt
import torch

from data.data_utilities import load_datasets, get_dataset_for_developing


def show_two_images(img1, img2, img_name):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img1)
    axarr[1].imshow(img2)
    plt.savefig(f"{img_name}.png")


def get_mask_for_pixels_greater_than(img, num, return_tensor=True):
    mask_2 = img > num
    if return_tensor:
        return mask_2.int()
    return mask_2.detach().numpy().astype(int)


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
percentage_of_data_to_keep = 1
number_of_data_to_keep = 10
trainset, testset = get_dataset_for_developing(transform,
                                               percentage_of_data_to_keep=percentage_of_data_to_keep,
                                               number_of_data_to_keep=number_of_data_to_keep)

def get_mask_for_1024_pixels():
    mask = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1,   1, 0, 1, 1, 1, 0, 0, 0, 1, 1,
        1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,   1, 0, 1, 0, 1, 0, 0, 1, 0,
        1, 0, 0, 0, 1, 0, 0, 0,   1, 1, 1,1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
        0, 1, 0, 0, 1, 0, 1, 1,   1, 1, 1, 1, 1, 1, 1,  1,  1, 1, 1,
        0, 0, 1, 0, 1, 1,  1,   1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1,
        0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0,
        0, 1, 0, 1, 0, 1, 1,   1, 0, 1, 1, 1, 1,   1, 0, 0, 0, 0, 1,
        1, 0, 0, 1, 1, 1, 0, 1,   1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1,   1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0,
        1, 1, 0, 1, 0, 1, 1,   1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
        0, 0, 0, 1, 0, 0, 1,  1,  1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
        0, 1, 1,  1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0,
        0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1,   1, 1, 1,
        0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1,  1,
        1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1,   0, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1,   1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,   1, 0, 1, 0, 1, 0,
        1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1,   1, 0, 1, 0, 0, 1, 0, 0, 1,
        0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1,   1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0,
        0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0,
        0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1,
        1, 1, 1, 0,   0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1,
        1, 1, 1, 1,   1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

    # mask_for_all_pixesl = torch.concat((mask,mask, mask)).reshape(3,32,32)
    return mask


# mask_stable = get_mask_for_1024_pixels()
# img = trainset[0][0]
# masked_img = (img.view(3,-1)*mask_stable).reshape(3,32,32)
# show_two_images(img.T, masked_img.T, img_name="correct_mask2")
# #

# # to problhma me autes tis maskes: h kathe eikona tha exei diaforetiko masked_output size.
# # synepws exoume problhma ti input features na baloume ston encoder mas.
# img = trainset[0][0]
# mask = torch.randint(0,2, (3,32,32))
# show_two_images(img.T, (img*mask).T, img_name="mask_zero_ones")
#
# # 0.1
# mask_1 =  get_mask_for_pixels_greater_than(img, 0.1)
# show_two_images(img.T, (img*mask_1).T, img_name="mask_0.1_less_zero")
# # 0.2
# mask_2 =  get_mask_for_pixels_greater_than(img, 0.2)
# show_two_images(img.T, (img*mask_2).T, img_name="mask_0.2_less_zero")
#
# # 0.3
# mask_3 =  get_mask_for_pixels_greater_than(img, 0.3)
# show_two_images(img.T, (img*mask_3).T, img_name="mask_03_less_zero")
#
# # 0.4
# mask_4 = get_mask_for_pixels_greater_than(img, 0.4)
# show_two_images(img.T, (img*mask_4).T, img_name="mask_04_less_zero")
#
# mask_5 =  get_mask_for_pixels_greater_than(img, 0.5)
# show_two_images(img.T, (img*mask_5).T, img_name="mask_0.5_less_zero")
#
# mask_6 =  get_mask_for_pixels_greater_than(img, 0.6)
# show_two_images(img.T, (img*mask_6).T, img_name="mask_0.6_less_zero")
#
# mask_7 =  get_mask_for_pixels_greater_than(img, 0.7)
# show_two_images(img.T, (img*mask_7).T, img_name="mask_0.7_less_zero")

