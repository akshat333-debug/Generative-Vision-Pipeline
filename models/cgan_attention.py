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


class CrossAttention(nn.Module):
    """
    Cross-Attention module. 
    Allows image spatial features (Queries) to attend to textual sequence features (Keys/Values).
    This directly aligns specific image regions with specific words/tokens.
    """
    def __init__(self, in_channels, context_dim):
        super(CrossAttention, self).__init__()
        self.in_channels = in_channels
        self.context_dim = context_dim
        
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_proj = nn.Linear(context_dim, in_channels // 8)
        self.value_proj = nn.Linear(context_dim, in_channels)
        
        # Learnable scale parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, context):
        # x: B x C x W x H
        # context: B x SeqLen x context_dim
        batch_size, C, width, height = x.size()
        
        # Query from image
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1) # B x N x C/8
        
        # Key, Value from text context
        key = self.key_proj(context).permute(0, 2, 1) # B x C/8 x SeqLen
        value = self.value_proj(context).permute(0, 2, 1) # B x C x SeqLen
        
        # Attention map: Image regions attending to Text tokens
        energy = torch.bmm(query, key) # B x N x SeqLen
        attention = F.softmax(energy, dim=-1) # B x N x SeqLen
        
        # Output synthesis
        out = torch.bmm(value, attention.permute(0, 2, 1)) # B x C x N
        out = out.view(batch_size, C, width, height)
        
        # Scaled residual connection
        return self.gamma * out + x


class ConditionalGenerator(nn.Module):
    """
    Conditional Generator (CGAN). Maps textual embedding sequences + latent noise Z to visuals.
    Features integrated Cross-Attention for text-image alignment.
    """
    def __init__(self, z_dim=100, embed_dim=512, feature_maps=64, out_channels=3):
        super(ConditionalGenerator, self).__init__()
        
        # Project pooled embedding to match spatial needs for the initial concat
        self.embed_proj = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(True)
        )
        
        # Block 1: 1x1 -> 4x4
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim + 256, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True)
        )
        
        # Block 2: 4x4 -> 8x8
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True)
        )
        
        # Block 3: 8x8 -> 16x16
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True)
        )
        
        # Attention Blocks at 16x16
        self.self_attn = SelfAttention(feature_maps * 2)
        self.cross_attn = CrossAttention(feature_maps * 2, context_dim=embed_dim)
        
        # Block 4: 16x16 -> 32x32
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True)
        )
        
        # Block 5: 32x32 -> 64x64
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, context):
        # context: B x SeqLen x embed_dim
        # noise: B x Z_dim x 1 x 1
        
        # Mean pool the context sequence just for the initial generative seed
        pooled_context = context.mean(dim=1) # B x embed_dim
        
        x_embed = self.embed_proj(pooled_context)
        x_embed = x_embed.view(-1, 256, 1, 1) # reshape for concatenation
        
        # Initial concatenation
        out = torch.cat([noise, x_embed], dim=1)
        
        # Spatial upsampling
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        
        # Apply Self-Attention (spatial coherence)
        out = self.self_attn(out)
        
        # Apply Cross-Attention (text alignment)
        out = self.cross_attn(out, context)
        
        # Final upsampling
        out = self.block4(out)
        out = self.block5(out)
        
        return out


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator. Scores visuals using Cross-Attention to check alignment with text.
    """
    def __init__(self, embed_dim=512, in_channels=3, feature_maps=64):
        super(ConditionalDiscriminator, self).__init__()
        
        self.embed_proj = nn.Sequential(
            nn.Linear(embed_dim, 1 * 64 * 64), 
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Block 1: 64x64 -> 32x32
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels + 1, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Block 2: 32x32 -> 16x16
        self.block2 = nn.Sequential(
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Attention Blocks at 16x16
        self.self_attn = SelfAttention(feature_maps * 2)
        self.cross_attn = CrossAttention(feature_maps * 2, context_dim=embed_dim)
        
        # Block 3: 16x16 -> 8x8
        self.block3 = nn.Sequential(
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Block 4: 8x8 -> 4x4
        self.block4 = nn.Sequential(
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Block 5: 4x4 -> 1x1
        self.block5 = nn.Sequential(
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, context):
        B = img.size(0)
        
        # Mean pool context for initial spatial map
        pooled_context = context.mean(dim=1)
        embed_mapped = self.embed_proj(pooled_context)
        embed_mapped = embed_mapped.view(B, 1, img.size(2), img.size(3))
        
        # Inject textual metadata into raw image pixel data
        out = torch.cat([img, embed_mapped], dim=1)
        
        out = self.block1(out)
        out = self.block2(out)
        
        # Attention checks
        out = self.self_attn(out)
        out = self.cross_attn(out, context)
        
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        
        return out.view(-1, 1).squeeze(1)


# Local validation
if __name__ == "__main__":
    _seq = torch.randn(2, 77, 512) # batch=2, seq=77 (CLIP tokens), dim=512
    _noise = torch.randn(2, 100, 1, 1)
    
    G = ConditionalGenerator()
    D = ConditionalDiscriminator()
    
    _gen = G(_noise, _seq)
    _prob = D(_gen, _seq)
    
    print("Cross-Attention CGAN verified")
    print(f"Generator output shape: {_gen.shape}")
    print(f"Discriminator prob mapping: {_prob.shape}")
