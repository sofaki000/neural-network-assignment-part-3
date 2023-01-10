import torch
import torch.nn as nn
import torch.nn.functional as F

activation_function= F.relu
#activation_function = F.tanh

class Encoder(nn.Module):
    def __init__(self, output_channels, hidden_1=16, hidden_2=24):
        super().__init__()
        num_features = 3
        kernel_size=3
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(in_channels=num_features,
                               out_channels=hidden_1,
                               kernel_size=kernel_size,
                               padding=1) # 1*16*32*32

        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(in_channels=hidden_1,
                               out_channels=output_channels,
                               kernel_size=3,padding=1)

        # self.conv3 = nn.Conv2d(in_channels=hidden_2,
        #                        out_channels=output_channels,
        #                        kernel_size=3,
        #                     padding=1)

    def forward(self, input):
        # input prepei na einai se morfh:
        # [ num_features, batch_size, sequence_length(?)]
        # [1, 3, 32, 32]
        output = self.conv1(input)
        output = activation_function(output)
        output = self.conv2(output)
        output = activation_function(output)
        # output = self.conv3(output)
        return output


class Decoder(nn.Module):
    def __init__(self,output_channels):
        super().__init__()
        hidden_1 = 16
        hidden_2 = 16
        self.t_conv1 = nn.ConvTranspose2d(output_channels, hidden_1, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(hidden_1, 3, 2, stride=2)
        # self.t_conv3 = nn.ConvTranspose2d(hidden_2, 3, 2, stride=2)

    def forward(self, input):
        # input prepei na einai se morfh:
        # [ num_features, batch_size, sequence_length(?)]
        # [1, 1, 32, 32]
        output = self.t_conv1(input) # 1*4*16*16->1*16*32*32
        output = activation_function(self.t_conv2(output)) # 1*16*64*64
        #output = activation_function(self.t_conv3(output)) # 1*3*128*128
        return output # F.softmax(output)


class CnnAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        output_channels = 4
        self.enc = Encoder(output_channels=output_channels, hidden_1=12, hidden_2=12)
        self.dec = Decoder(output_channels=output_channels)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)

        self.linear = nn.Linear(4096, 32*32) #nn.Linear(in_features=62*62, out_features=32*32)
    def forward(self,input):
        batch_size = input.size(0)
        enc_output = self.enc(input) # 1*3*32*32->
        enc_output = self.pool1(enc_output) # converts 32 by 32 to 16 by 16
        dec_output = self.dec(enc_output)
        # output = self.pool2(dec_output) # converts 62 by 62 to 32 by 32
        output = self.linear(dec_output.view(batch_size,3,-1)).view(batch_size, 3,32, 32)
        # assert output.size
        return F.softmax(output)


# model = CnnAutoEncoder()
# trainable_params = sum( p.numel() for p in model.parameters() if p.requires_grad )
# print(trainable_params)