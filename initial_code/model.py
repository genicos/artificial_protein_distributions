import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLanguageModel(nn.Module):
    """Enhanced Masked Language Model with positional embeddings and transformer components"""
    def __init__(self, alphabet_size, max_length=512, embedding_dim=128, hidden_dim=256, 
                 num_attention_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length

        # Token embeddings (+1 for mask token)
        self.token_embedding = nn.Embedding(alphabet_size + 1, embedding_dim)
        
        # Positional embeddings
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, alphabet_size)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights like BERT"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len) containing token indices
               where alphabet_size is used as the [MASK] token
        Returns:
            logits: Tensor of shape (batch_size, seq_len, alphabet_size)
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Get token embeddings
        token_embeds = self.token_embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Get positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        embeddings = token_embeds + pos_embeds
        
        # Transformer encoder expects (batch_size, seq_len, embedding_dim)
        transformer_out = self.transformer_encoder(embeddings)
        
        # Layer normalization
        norm_out = self.layer_norm(transformer_out)
        
        # Project to output vocabulary
        logits = self.output_proj(norm_out)
        
        return logits

