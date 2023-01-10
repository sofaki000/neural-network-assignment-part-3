import torch
import torch.nn as nn
import torch.nn.functional as F

activation_function= F.relu
#activation_function = F.tanh

class Encoder(nn.Module):
    ''' o encoder mas peirazei MONO TA UNMASKED PIXELS!!!'''
    def __init__(self, output_channels, hidden_1=16, hidden_2=24):
        super().__init__()
        num_features = 3
        kernel_size=3
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(in_channels=num_features, out_channels=hidden_1, kernel_size=kernel_size, padding=1) # 1*16*32*32

        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.linear = nn.Linear(hidden_1 , output_channels) #nn.Linear(in_features=hidden_1, out_features=output_channels)
        self.conv2 = nn.Conv2d(in_channels=hidden_1, out_channels=output_channels,  kernel_size=3,padding=1)

    def forward(self, input):
        output = self.conv1(input)
        # 1*12 *28*28 ->  1*3*28*28
        embedded_output = self.linear(output.reshape(28, 28, 12))
        return embedded_output.reshape(1,3,28,28)


class Decoder(nn.Module):
    def __init__(self,output_channels):
        super().__init__()
        hidden_1 = 16
        self.t_conv1 = nn.ConvTranspose2d(output_channels, hidden_1, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(hidden_1, 3, 2, stride=2)

    def forward(self, input):
        # input prepei na einai se morfh: [ num_features, batch_size, sequence_length(?)]
        # [1, 1, 32, 32]
        output = self.t_conv1(input) # 1*4*16*16->1*16*32*32
        output = activation_function(self.t_conv2(output)) # 1*16*64*64

        return output

def add_masked_pixels(encoded, mask):

    # gia osa pixels exoume value
    enc_output = encoded.view(3,-1).transpose(1,0)
    encoded_pixels_num = enc_output.size(1)

    # briskoume to index poiwn pixels den kaname mask prin
    indices_to_encoded= mask.nonzero()

    # antikathistoume ta arxika pixels pou den eginan mask, me ta encoded pixels
    decoder_input = torch.zeros(1024,3)

    indices_to_encoded = indices_to_encoded.squeeze().detach().numpy()
    for i in range(encoded_pixels_num):

        # an to i einai mesa sto indices_to_encoded to pairnoume apo to encoder output
        if i in indices_to_encoded:
            decoder_input[i] = enc_output[i]

    # to apotelesma einai 3*32*32 encoded image INCLUDING ta masked pixels
    return decoder_input.transpose(1,0)


class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        output_channels = 3
        self.enc = Encoder(output_channels=output_channels, hidden_1=12, hidden_2=12)
        self.dec = Decoder(output_channels=output_channels)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.linear = nn.Linear(4096, 32*32) #nn.Linear(in_features=62*62, out_features=32*32)
    def forward(self,input,mask):
        input = input.unsqueeze(0)
        enc_output = self.enc(input)
        # twra to 3*28*28 encoded output mas, tou prosthetoume ta masked indexes apo prin
        output_with_masked_pixels = add_masked_pixels(enc_output, mask)

        output_with_masked_pixels = output_with_masked_pixels.reshape(1, 3, 32, 32)
        dec_output = self.dec(output_with_masked_pixels)

        output = self.pool2(dec_output) #.view(batch_size, 3,32, 32)# converts 62 by 62 to 32 by 32

        assert output.size(3)==32 # theloume o decoder na thn epistrefei oloklhrh
        assert output.size(2)==32
        return F.softmax(output)


# model = CnnAutoEncoder()
# trainable_params = sum( p.numel() for p in model.parameters() if p.requires_grad )
# print(trainable_params)