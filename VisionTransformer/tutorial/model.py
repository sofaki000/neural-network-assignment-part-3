
import torch.nn.functional as F
import pytorch_lightning
import torch
import torch.nn as nn
from VisionTransformer.cifar10Utilities import image_to_patch

# SXOLIO: den exw grapsei auta ta classes

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        '''
        :param embed_dim: dimensionality of input and attention feature vectors
        :param hidden_dim: dimensionality of hidden layer in feed forward network (usually 2-4x larger than embed dim)
        :param num_heads: number of heads to use in the multi-head attention block
        :param dropout: amount of dropout to apply in the feed-forward network
        '''
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.linear = nn.Sequential(
            nn.Linear(embed_dim,  hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        inp_x = self.layer_norm1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm2(x))
        return x


class VisionTransformerModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_channels,num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        '''
        :param embed_dim: dimensionality of the input feature vectors to the transformer
        :param hidden_dim: dimensionality of the hidden layer in the feed-forward networks within the transformer
        :param num_channels: number of channels of the input (3 for RBG)
        :param num_layers: number of layers to use in the Transformer
        :param num_classes: number of classes to predict
        :param patch_size: number of pixels that the patches have per dimension
        :param num_patches: maximum number of patches an image can have
        :param dropout: aamount of dropout to apply in the feed-forward network and on the input encoding
        '''

        super().__init__()
        self.patch_size = patch_size

        # layers/network
        self.input_layer = nn.Linear(num_channels*(patch_size**2) , embed_dim)

        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

        # parameters/embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches, embed_dim))

    def forward(self, x):
        # preprocess input: x: [batch_size, channels, height, width]
        x = image_to_patch(x, self.patch_size)

        b, t, _ = x.shape # [batch_size,64, 48]

        x = self.input_layer(x)

        # add cls token and positional encoding
        #cls_token = self.cls_token.repeat(b,1,1)
        #x = torch.cat([cls_token, x], dim=1)

        x = x + self.pos_embedding[:,:t+1]
        # apply transformer
        x = self.dropout(x)
        x = x.transpose(0,1) #[_ , batch_size, 256]
        x = self.transformer(x) # ! gyrnaei same size!

        # perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out


