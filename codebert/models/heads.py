"""Classification heads for CodeBERT vulnerability classifier."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    """MLP classification head.
    
    Architecture: CLS token → Dropout → Linear → ReLU → Dropout → Linear → 2 classes
    
    Args:
        hidden_size: Size of the input hidden state (from encoder).
        num_labels: Number of output classes. Default: 2.
        dropout_prob: Dropout probability. Default: 0.1.
        intermediate_size: Size of intermediate layer. Default: None (uses hidden_size).
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 2,
        dropout_prob: float = 0.1,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        intermediate_size = intermediate_size or hidden_size
        
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(intermediate_size, num_labels)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            hidden_states: CLS token representation of shape (batch_size, hidden_size).
            
        Returns:
            Logits of shape (batch_size, num_labels).
        """
        x = self.dropout1(hidden_states)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout2(x)
        logits = self.classifier(x)
        return logits


class CNNHead(nn.Module):
    """CNN classification head with multi-scale convolutions.
    
    Uses multiple kernel sizes (2, 3, 5) to capture different n-gram patterns.
    Designed for ablation studies comparing with MLP head.
    
    Args:
        hidden_size: Size of the input hidden state (from encoder).
        num_labels: Number of output classes. Default: 2.
        dropout_prob: Dropout probability. Default: 0.1.
        num_filters: Number of filters per kernel size. Default: 128.
        kernel_sizes: Tuple of kernel sizes. Default: (2, 3, 5).
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 2,
        dropout_prob: float = 0.1,
        num_filters: int = 128,
        kernel_sizes: tuple = (2, 3, 5),
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2,
            )
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_labels)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            hidden_states: Sequence representation of shape (batch_size, seq_len, hidden_size).
            
        Returns:
            Logits of shape (batch_size, num_labels).
        """
        # (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size, seq_len)
        x = hidden_states.transpose(1, 2)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch_size, num_filters, seq_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(pooled)
        
        # Concatenate all filter outputs
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(kernel_sizes))
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
