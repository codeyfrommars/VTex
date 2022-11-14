import torch
import torch.nn as nn


from encoder.densenet import DenseNet
from embedding.transformer_embedding import EncoderEmbedding

class Encoder(nn.Module):
    """
    Encoder for VTex transformer. Input is an image. Output goes to decoder.
    """
    def __init__(self, growth_rate, block_depth, compression, dropout, height, width, dim_model):
        super(Encoder, self).__init__()

        # CNN to extract image features
        self.cnn = DenseNet(growth_rate, block_depth, compression, dropout)

        # 1x1 convolution to reshape CNN's output to transformer's d_model dimensions
        self.reshape_conv = nn.Conv2d(self.cnn.out_features, dim_model, kernel_size=1)
        self.reshape_norm = nn.BatchNorm2d(dim_model)
        self.reshape_relu = nn.ReLU(inplace=True)

        # Image positional encoding
        self.embed = EncoderEmbedding(height//16, width//16, dim_model)
        self.dim_model = dim_model

    def forward(self, img):
        """
        img [batch_size, channels=1, height, width]
        output [batch_size, (height*width)//16, dim_model]
        """
        batch_size, _, height, width = img.size()
        # Feature extraction
        # [batch_size, out_features, height/16, width/16]
        features = self.cnn(img)

        # Feature reshape to [batch_size, dim_model, height/16, width/16]
        features = self.reshape_conv(features)
        features = self.reshape_norm(features)
        features = self.reshape_relu(features)

        # Rearrange to [batch_size, height/16, width/16, dim_model]
        features = torch.permute(features, (0, 2, 3, 1))

        assert (features.size() == (batch_size, height//16, width//16, self.dim_model)), "feature reshape incorrect shape"

        # Image positional encoding
        features = self.embed(features)

        # Reshape from 2D to 1D
        # before: [batch_size, height/16, width/16, dim_model]
        # after: [batch_size, height/16 * width/16, dim_model]
        features = torch.flatten(features, start_dim=1, end_dim=2)

        assert (features.size() == (batch_size, (height//16) * (width//16), self.dim_model)), "Encoder output incorrect shape"

        return features

