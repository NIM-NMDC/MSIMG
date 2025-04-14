import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):

    def __init__(self, patch_size=224, emb_dim=768, in_channels=1):
        super().__init__()
        self.emb_dim = emb_dim
        self.project = nn.Conv2d(
            in_channels=in_channels,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.view(B * C, 1, H, W)
        x = self.project(x)  # (B * C, emb_dim, 1, 1) (kernel_size=h=W)
        x = x.flatten(2).squeeze(-1)  # (B * C, emb_dim)
        x = x.view(B, C, self.emb_dim)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, x, positions):
        """
        :param positions: (B, C, 2) each patch's (y, x) coordinates
        :return: positional encoding of shape (B, C, emb_dim)
        """
        B, C, _ = positions.shape
        device = x.device

        positions = positions.float().to(device)

        div_term = torch.exp(torch.arange(0, self.emb_dim // 2, 2).float() * -(torch.log(torch.tensor(10000.0)) / (self.emb_dim // 2))).to(device)

        pos_y = positions[:, :, 0].unsqueeze(-1)  # (B, C, 1)
        pos_x = positions[:, :, 1].unsqueeze(-1)  # (B, C, 1)

        pe_y = torch.zeros(B, C, self.emb_dim // 2).to(device)
        pe_y[:, :, 0::2] = torch.sin(pos_y * div_term)
        pe_y[:, :, 1::2] = torch.cos(pos_y * div_term)

        pe_x = torch.zeros(B, C, self.emb_dim // 2).to(device)
        pe_x[:, :, 0::2] = torch.sin(pos_x * div_term)
        pe_x[:, :, 1::2] = torch.cos(pos_x * div_term)

        pe = torch.cat([pe_y, pe_x], dim=-1)  # (B, C, emb_dim)

        return pe


class ViT(nn.Module):

    def __init__(self, patch_size, emb_dim, in_channels, num_heads, num_layers, num_classes, dropout=0.4):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, emb_dim=emb_dim, in_channels=in_channels)
        self.positional_encoding = PositionalEncoding(emb_dim=emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x, positions, padding_mask=None):
        """
        :param x: (B, C, H, W)
        :param positions: (B, C, 2) each patch's (y, x) coordinates
        :param padding_mask: (B, C) mask for padding tokens
        """
        B, C, H, W = x.shape

        patch_embed = self.patch_embedding(x)
        positional_encoding = self.positional_encoding(x, positions)

        tokens = patch_embed + positional_encoding

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, emb_dim)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, C + 1, emb_dim)

        if padding_mask:
            cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_pad, padding_mask], dim=1)

        out = self.encoder(tokens, src_key_padding_mask=padding_mask)
        cls_output = out[:, 0]  # (B, emb_dim)

        return self.classifier(cls_output)