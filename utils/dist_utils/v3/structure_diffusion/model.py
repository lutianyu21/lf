import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Tuple
from functools import partial
from .triangle import TriangleMultiplication, TriangleAttention, Transition, PreLayerNorm
from torch.utils.checkpoint import checkpoint


REFERENCE_FRAME = torch.tensor(
    [
        [-0.522, 1.362, 0.0],  # N
        [0.0, 0.0, 0.0],      # CA
        [1.525, 0.0, 0.0],    # C
    ],
    dtype=torch.float32,
)


class MLP(nn.Module):
    """MLP module for transformer block."""
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x, key_padding_mask=None):
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class PairBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        pre_ln = partial(PreLayerNorm, dim = embed_dim)
        self.tri_mult_outgoing = pre_ln(TriangleMultiplication(mix='outgoing', dim=embed_dim, dropout=dropout, dropout_type='row'))
        self.tri_mult_incoming = pre_ln(TriangleMultiplication(mix='incoming', dim=embed_dim, dropout=dropout, dropout_type='row'))
        # self.tri_attn_starting = pre_ln(TriangleAttention(node_type='starting', dim=embed_dim, heads=4, dropout=dropout, dropout_type='row'))
        # self.tri_attn_ending = pre_ln(TriangleAttention(node_type='ending', dim=embed_dim, heads=4, dropout=dropout, dropout_type='col'))
        self.pairwise_transition = Transition(dim=embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, N, C] feature map
            mask: Optional boolean mask of shape [B, N]
        Returns:
            Updated feature map of the same shape.
        """

        x = self.tri_mult_outgoing(x, mask=mask) + x
        x = self.tri_mult_incoming(x, mask=mask) + x
        # attn_start_out = self.tri_attn_starting(x, mask=mask)
        # x = attn_start_out + x
        # attn_end_out = self.tri_attn_ending(x, mask=mask)
        # x = attn_end_out + x
        x = self.pairwise_transition(x) + x

        return x


class DistanceMatrixEmbedding(nn.Module):
    """Embedding layer for distance matrices with stacked criss-cross blocks."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 256, num_blocks: int = 6, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_channels, embed_dim)
        self.blocks = nn.ModuleList([
            PairBlock(embed_dim, num_heads=num_heads, dropout=0.25)
            for _ in range(num_blocks)
        ])
        self.merge = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, distance_matrix: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            distance_matrix: [B, N, N, D] distance matrix
            mask: Optional boolean mask of shape [B, N]
        Returns:
            embedded: [B, N, embed_dim] token embeddings
        """
        B, N, _, _ = distance_matrix.shape
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=distance_matrix.device)
        else:
            mask = mask.to(device=distance_matrix.device)

        x = self.linear(distance_matrix)
        for block in self.blocks:
            x = checkpoint(block, x, mask)

        row_valid = mask.unsqueeze(1).expand(-1, N, -1)
        row_weights = row_valid.to(dtype=x.dtype)
        row_denom = row_weights.sum(dim=2, keepdim=True).clamp(min=1.0)
        row_ctx = (x * row_weights.unsqueeze(-1)).sum(dim=2) / row_denom

        col_valid = mask.unsqueeze(2).expand(-1, -1, N).transpose(1, 2)
        col_weights = col_valid.to(dtype=x.dtype)
        col_denom = col_weights.sum(dim=2, keepdim=True).clamp(min=1.0)
        col_ctx = (x.transpose(1, 2) * col_weights.unsqueeze(-1)).sum(dim=2) / col_denom

        token_embed = torch.cat([row_ctx, col_ctx], dim=-1)
        token_embed = self.merge(token_embed)
        token_embed = self.norm(token_embed)

        return token_embed


class AttentionPairBias(nn.Module):
    def __init__(self, H, Hz, n_head):
        """
        H: hidden dim of a  (c_a)
        Hz: pair feature dim
        n_head: number of heads
        """
        super().__init__()
        assert H % n_head == 0
        self.n_head = n_head
        self.c = H // n_head  # per-head dim

        # Input projections
        self.ln_a = nn.LayerNorm(H)
        self.ln_z = nn.LayerNorm(Hz)

        self.q_proj = nn.Linear(H, H)
        self.kv_proj = nn.Linear(H, 2 * H, bias=False)
        self.g_proj = nn.Linear(H, H, bias=False)
        self.b_proj = nn.Linear(Hz, n_head, bias=False)

        # Output projection
        self.out_proj = nn.Linear(H, H, bias=False)

    def forward(self, a, z, mask=None):
        """
        a: [B, L, H]
        z: [B, L, L, Hz]
        """
        B, L, H = a.shape
        n_head = self.n_head
        c = self.c

        a_ln = self.ln_a(a)  # [B, L, H]

        q = self.q_proj(a_ln).reshape(B, L, n_head, c)
        kv = self.kv_proj(a_ln).reshape(B, L, 2, n_head, c)
        k, v = kv[:, :, 0], kv[:, :, 1]  # [B, L, n_head, c] each

        g = torch.sigmoid(self.g_proj(a_ln)).reshape(B, L, n_head, c)  # [B,L,H]->[B,L,h,c]
        z_ln = self.ln_z(z)  # [B, L, L, Hz]
        if mask is not None:
            pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # [B, L, L]
            z_ln = torch.where(pair_mask.unsqueeze(-1), z_ln, torch.zeros_like(z_ln))
        b = self.b_proj(z_ln).reshape(B, L, L, n_head)  # [B, L, L, h]

        qh = q.permute(0, 2, 1, 3)  # [B,h,L,c]
        kh = k.permute(0, 2, 1, 3)  # [B,h,L,c]

        att = torch.einsum("bhic,bhjc->bhij", qh, kh)  # [B,h,L,L]
        att = att / (c ** 0.5)

        bh = b.permute(0, 3, 1, 2)  # [B,h,L,L]
        att = att + bh

        A = F.softmax(att, dim=-1)  # [B,h,L,L]

        vh = v.permute(0, 2, 1, 3)  # [B,h,L,c]
        ctx = torch.einsum("bhij,bhjc->bhic", A, vh)  # [B,h,L,c]

        gh = g.permute(0, 2, 1, 3)  # [B,h,L,c]
        ctx = gh * ctx  # [B,h,L,c]

        if mask is not None:
            ctx = ctx * mask[:, None, :, None]

        ctx = ctx.permute(0, 2, 1, 3).reshape(B, L, H)  # [B,L,H]

        out = self.out_proj(ctx)
        return out


class StructureViT(nn.Module):
    """
    Vision Transformer-based model for predicting residue coordinates from distance matrices.
    Input: Variable-length distance matrix [N, N, 3]
    Output: Frame coordinates [N, atoms_per_residue, 3]
    """

    def __init__(
        self, 
        in_channels: int = 3,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_pairformer_blocks: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_sinusoidal_pos_embed: bool = False,
        output_dim: Optional[int] = None,
        frame_parameterization: str = "direct",
        reference_frame: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.use_sinusoidal_pos_embed = use_sinusoidal_pos_embed
        self.frame_parameterization = frame_parameterization
        self.reference_frame = reference_frame
        
        # Distance matrix embedding
        self.distance_embedding = DistanceMatrixEmbedding(in_channels, embed_dim, num_blocks=num_pairformer_blocks, num_heads=num_heads, dropout=dropout)
        self.linear_z = nn.Linear(in_channels, embed_dim)
        
        # Positional encoding (learnable or sinusoidal)
        self.pos_embed = None
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            # TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            AttentionPairBias(embed_dim, embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output head for coordinate prediction / rigid parameters
        self.coord_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, output_dim)
        )
        
    def forward(self, distance_matrix: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Forward pass through the structure prediction model.
        
        Args:
            distance_matrix: [B, N, N, 3] distance matrix (or [N, N, 3] for single sample)
            mask: Optional boolean tensor [B, N] (or [N]) marking valid tokens
            
        Returns:
            coordinates: [B, N, 3, 3] predicted residue frames
        """
        B, N, _, _ = distance_matrix.shape

        x = self.distance_embedding(distance_matrix, mask)  # [B, N, embed_dim]
        z = self.linear_z(distance_matrix)  # [B, N, embed_dim]

        pos_embed = self._get_pos_embed(N, dtype=x.dtype, device=x.device)
        x = x + pos_embed
        for block in self.blocks:
            x = block(x, z, mask)

        x = self.norm(x)

        raw_output = self.coord_head(x)
        coordinates = self._decode_frames(raw_output)

        return coordinates

    def _decode_frames(self, raw_output: torch.Tensor) -> torch.Tensor:
        if self.frame_parameterization == "direct":
            return raw_output.view(raw_output.shape[0], raw_output.shape[1], 3, 3)
        elif self.frame_parameterization == "rigid":
            rot6d = raw_output[..., :6]
            translation = raw_output[..., 6:9]

            a1 = F.normalize(rot6d[..., 0:3], dim=-1, eps=1e-6)
            a2 = rot6d[..., 3:6]
            proj = (a1 * a2).sum(dim=-1, keepdim=True)
            a2 = F.normalize(a2 - proj * a1, dim=-1, eps=1e-6)
            a3 = torch.cross(a1, a2, dim=-1)
            rotation = torch.stack([a1, a2, a3], dim=-1)
            reference = self.reference_frame.view(1, 1, 3, 3)
            rotated = torch.matmul(reference.to(raw_output.device, raw_output.dtype), rotation.transpose(-1, -2))
            coords = rotated + translation.unsqueeze(-2)
            return coords

    def _get_pos_embed(self, seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.use_sinusoidal_pos_embed:
            return self._build_sinusoidal_table(seq_len, self.embed_dim, device, dtype).unsqueeze(0)

        if seq_len <= self.max_seq_len:
            pos_embed = self.pos_embed[:, :seq_len, :]
        else:
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        return pos_embed.to(device=device, dtype=dtype)

    def _build_sinusoidal_table(
        self,
        seq_len: int,
        embed_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base_dtype = torch.float32
        position = torch.arange(seq_len, device=device, dtype=base_dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, device=device, dtype=base_dtype) * (-(math.log(10000.0) / embed_dim))
        )

        table = torch.zeros(seq_len, embed_dim, device=device, dtype=base_dtype)
        angles = position * div_term
        table[:, 0::2] = torch.sin(angles)
        table[:, 1::2] = torch.cos(angles)

        if dtype != base_dtype:
            table = table.to(dtype=dtype)

        return table


class StructurePredictionModel(nn.Module):
    """
    Complete structure prediction model that maps distance matrices to residue coordinates.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_pairformer_blocks: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_sinusoidal_pos_embed: bool = False,
        output_dim: Optional[int] = None,
        frame_parameterization: str = "direct",
        reference_frame: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.vit = StructureViT(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_pairformer_blocks=num_pairformer_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_sinusoidal_pos_embed=use_sinusoidal_pos_embed,
            output_dim=output_dim,
            frame_parameterization=frame_parameterization,
            reference_frame=reference_frame,
        )
        
    def forward(self, distance_matrix, mask: Optional[torch.Tensor] = None):
        """
        Forward pass.
        
        Args:
            distance_matrix: [B, N, N, 3] distance matrix (or [N, N, 3] for single sample)
            mask: Optional boolean tensor [B, N] (or [N])
            
        Returns:
            coordinates: [B, N, 3, 3] predicted residue frames
        """
        if mask is None:
            B, N, _, _ = distance_matrix.shape
            mask = torch.ones(B, N, dtype=torch.bool, device=distance_matrix.device)
        return self.vit(distance_matrix, mask=mask)