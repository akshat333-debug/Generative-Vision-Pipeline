import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Self-Attention module for GAN architecture (SAGAN implementation mapping).
    Allows the model to focus on pertinent spatial areas of the image.
    """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # Learnable scale parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Reshape to (batch_size, C/8, N) where N = width * height
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1) # B x N x C/8
        key = self.key_conv(x).view(batch_size, -1, width * height)                      # B x C/8 x N
        
        # Energy and attention mapping
        energy = torch.bmm(query, key) # B x N x N
        attention = F.softmax(energy, dim=-1) # B x N x N
        
        value = self.value_conv(x).view(batch_size, -1, width * height) # B x C x N
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        # Scaled residual connection
        out = self.gamma * out + x
        return out


class ConditionalGenerator(nn.Module):
    """
    Conditional Generator (CGAN). Maps textual embeddings + latent noise Z to visuals.
    """
    def __init__(self, z_dim=100, embed_dim=768, feature_maps=64, out_channels=3):
        super(ConditionalGenerator, self).__init__()
        
        # Project conditioning embedding to match spatial needs
        self.embed_proj = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(True)
        )
        
        # Deconvolutional trunk taking concatenated (Z + Projected Embedding)
        self.main = nn.Sequential(
            # Input: Z_dim + 256. Generate 4x4
            nn.ConvTranspose2d(z_dim + 256, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            
            # 8x8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            # 16x16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            # inject Self-Attention block
            SelfAttention(feature_maps * 2),
            
            # 32x32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            # Output: 64x64
            nn.ConvTranspose2d(feature_maps, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, embedding):
        # embedding: B x embed_dim
        # noise: B x Z_dim x 1 x 1
        x_embed = self.embed_proj(embedding)
        x_embed = x_embed.view(-1, 256, 1, 1) # reshape for concatenation
        
        # concatenate conditioning block
        z_concat = torch.cat([noise, x_embed], dim=1)
        return self.main(z_concat)


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator mapping generated/real visuals + textual embeddings to probabilities.
    Includes Self Attention.
    """
    def __init__(self, embed_dim=768, in_channels=3, feature_maps=64):
        super(ConditionalDiscriminator, self).__init__()
        
        self.embed_proj = nn.Sequential(
            nn.Linear(embed_dim, 1 * 64 * 64), # Project to spatial map of 64x64
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Input channels + defined 1 channel embedding projection
        self.main = nn.Sequential(
            # Input: 64x64 x (in_channels + 1)
            nn.Conv2d(in_channels + 1, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Self-Attention block
            SelfAttention(feature_maps * 2),
            
            # 16x16
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 Output to 1 probability scoring
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, embedding):
        B = img.size(0)
        
        # Map conditioning to standard spatial structure
        embed_mapped = self.embed_proj(embedding)
        embed_mapped = embed_mapped.view(B, 1, img.size(2), img.size(3))
        
        # Inject textual metadata into raw image pixel data
        c_concat = torch.cat([img, embed_mapped], dim=1)
        return self.main(c_concat).view(-1, 1).squeeze(1)


# Simple block check locally
if __name__ == "__main__":
    _embed = torch.randn(2, 768)
    _noise = torch.randn(2, 100, 1, 1)
    
    G = ConditionalGenerator()
    D = ConditionalDiscriminator()
    
    _gen = G(_noise, _embed)
    _prob = D(_gen, _embed)
    
    print("Self Attention CGAN verified")
    print(f"Generator output shape: {_gen.shape}")
    print(f"Discriminator prob mapping: {_prob.shape}")
