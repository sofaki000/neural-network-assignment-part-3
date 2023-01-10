import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(in_features=32*32, out_features=hidden_size)
    def forward(self, input):
        output = self.layer1(input)
        return output


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(in_features=hidden_size, out_features=32*32)
    def forward(self, input):
        output = self.layer1(input)
        return F.sigmoid(output)

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, input):
        encoder_output = self.encoder(input)
        output = self.decoder(encoder_output)

        return output
    
    
class LinearAutoEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.enc = Encoder(hidden_size=hidden_size)
        self.dec = Decoder(hidden_size=hidden_size)
        
        self.autoencoder = AutoEncoder(self.enc, self.dec)

    def forward(self,input):
        return self.autoencoder(input)


# xrhsimopoiw ton linearAutoEncoder anti gia:
# enc = Encoder(hidden_size=hidden_size)
# dec = Decoder(hidden_size=hidden_size)
# AutoEncoder(enc, dec)