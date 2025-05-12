import torch 
import torch.nn as nn 
import torch.nn.functional as F

class Conv1dModule(nn.Module):
    def __init__(self, n_mels, conv_dim, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=n_mels, 
            out_channels=conv_dim, 
            kernel_size=kernel_size, 
            padding=kernel_size//2
            )
        
        self.conv2 = nn.Conv1d(
            in_channels=conv_dim, 
            out_channels=conv_dim, 
            kernel_size=kernel_size, 
            stride=2, #reduce time dimension by factor of 2
            padding=kernel_size//2
            )
            
        self.bn1 = nn.BatchNorm1d(conv_dim)
        self.bn2 = nn.BatchNorm1d(conv_dim)
        
    def forward(self, x):
        #input shape: [batch_size, n_mels, time_frames]
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.attn_ln = nn.LayerNorm(dim)
        
        #multi-head attention
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        #feed-forward network
        self.mlp_ln = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*dim, dim),
            nn.Dropout(dropout)
            )
    def forward(self, x):
        x_ln = self.attn_ln(x)
        attn_output, _ = self.attn(x_ln, x_ln, x_ln)
        x = x + attn_output
        
        x_ln = self.mlp_ln(x)
        mlp_output = self.mlp(x_ln)
        x = x + mlp_output
        
        return x
    
class CNNModule(nn.Module):
    def __init__(self, input_dim, cnn_dims=[256, 512]):
        super().__init__()
        self.cnn_layers = nn.ModuleList()
        in_dim = input_dim 
        
        for dim in cnn_dims: 
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=in_dim, 
                    out_channels=dim, 
                    kernel_size=3, 
                    padding=1
                    ),
                nn.BatchNorm1d(dim),
                nn.GELU(),
            ))
            in_dim = dim     
            
        self.global_pool = nn.AdaptiveAvgPool1d(1) #[batchsize, 512, varying_time_frames] -> [batchsize, 512, 1]
        self.output_dim = cnn_dims[-1]
        
    def forward(self, x):
        #input shape: [batch_size, input_dim, time_frames]
        for cnn_layer in self.cnn_layers: 
            x = cnn_layer(x)
        
        #global pooling 
        x = self.global_pool(x) #[batchsize, 512, 1]
        x = x.view(x.size(0), -1) #flatten
        
        
class UrbanSoundModel(nn.Module):
    def __init__(
            self, 
            n_mels=64, 
            num_classes=10, 
            
            #1d conv params 
            conv_dim=128, 
            conv_kernel=3, 
            
            #encoder params 
            encoder_dim=256, 
            num_encoder_blocks=4, 
            num_heads=8,
            dropout=0.1, 
            
            #cnn params 
            cnn_dim=[256, 512]
    ):
        super().__init__()
        
        # 1d conv module 
        self.conv_module = Conv1dModule(
            n_mels=n_mels, 
            conv_dim=conv_dim, 
            kernel_size=conv_kernel
            )
        
        #project to encoder dim if needed 
        self.projection = nn.Linear(conv_dim, encoder_dim) if conv_dim != encoder_dim else nn.Identity()
        
        #position embeddings 
        self.register_buffer('position_embeddings', self._get_sinusoidal_embeddings(1000, encoder_dim))
        
        #encoder blocks 
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(dim=encoder_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_encoder_blocks)]
        )
        
        #encoder output normalisation 
        self.ln_post = nn.LayerNorm(encoder_dim)
        
        #cnn classification module 
        self.cnn_module = CNNModule(
            input_dim=encoder_dim, 
            cnn_dims=cnn_dim
            )
        
        #classification layer: 
        self.classifier = nn.Linear(self.cnn_module.output_dim, num_classes)
        
        #initialise weights 
        self._init_weights()
    
    #go back and review this 
    def _get_sinusoidal_embeddings(self, max_len, dim):
        half_dim = dim // 2 
        positions = torch.arange(max_len, dtype=torch.float32)
        frequencies = torch.exp(-torch.arrang(half_dim, dtype=torch.float32) * (np.log(10000) / half_dim - 1))
        
        #create position embeddings 
        args = positions[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        
        #pad if needed for odd dimensions 
        if dim % 2 == 1: 
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding 
    
    #review this too 
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='gelu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x): 
        #input [batch_size, 1, n_mels, time_frames] ->> double check this, check the output from dataset.py
        batch_size = x.size(0)
        
        #remove channel dimension: 
        x = x.squeeze(1) #[batch_size, n_mels, time_frames]
        
        #apply 1d conv module 
        x = self.conv_module(x) #[batch_size, conv_dim, reduced_time_frames]
        
        #transposte to [batch_size, reduced_time_frames, conv_dim]
        x = x.transpose(1, 2) #[batch_size, reduced_time_frames, conv_dim]
        
        #project to encoder dim if needed 
        x = self.projection(x)
        
        #add position embeddings 
        seq_len = x.size(1) 
        pos_emb = self.positional_embeddings[:seq_len]
        x = x + pos_emb
        
        #apply encoder blocks 
        for block in self.encoder_blocks: 
            x = block(x)
        
        #apply layer norm
        x = self.ln_post(x)
        
        #transpose back to [batch_size, encoder_dim, reduced_time_frames]
        x = x.transpose(1, 2) 
        
        #apply cnn module 
        x = self.cnn_module_module(x)
        
        #classification 
        x = self.classifier(x)
        
        return x 
    
if __name__ == "__main__":
    from dataset import get_dataloaders
    import numpy as np 
    
    #create data loaders 
    train_loader, test_loader = get_dataloaders(
        batch_size=32, 
        sr=22050,
        duration=4.0,
        n_mels=64
    )
        
    #create model   
    model = UrbanSoundModel(
        n_mels=64, 
        num_classes=10, 
        conv_dim=128, 
        encoder_dim=256, 
        num_encoder_blocks=4, 
        num_heads=8, 
    )
    
    #print model summary 
    print(model)
    
    #calculate total parameters 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    #example of forward pass 
    batch = next(iter(train_loader))[0]
    output = model(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {output.shape}")
    