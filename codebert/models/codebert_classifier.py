"""CodeBERT-based vulnerability classifier."""

from typing import Optional, Tuple, Union, Literal

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from .heads import MLPHead, CNNHead


class CodeBERTClassifier(nn.Module):
    """CodeBERT-based binary vulnerability classifier.
    
    Uses microsoft/codebert-base as the encoder with a configurable classification head.
    
    Args:
        model_name: Pretrained model name or path. Default: "microsoft/codebert-base".
        num_labels: Number of output classes. Default: 2.
        head_type: Type of classification head ("mlp" or "cnn"). Default: "mlp".
        dropout_prob: Dropout probability for classification head. Default: 0.1.
        freeze_encoder: Whether to freeze encoder layers. Default: False.
        freeze_n_layers: Number of encoder layers to freeze (from bottom). Default: 0.
        pos_weight: Positive class weight for BCEWithLogitsLoss. Default: None.
    
    Example:
        >>> model = CodeBERTClassifier(pos_weight=torch.tensor([1.0, 5.0]))
        >>> outputs = model(input_ids, attention_mask, labels=labels)
        >>> loss, logits = outputs
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        num_labels: int = 2,
        head_type: Literal["mlp", "cnn"] = "mlp",
        dropout_prob: float = 0.1,
        freeze_encoder: bool = False,
        freeze_n_layers: int = 0,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.head_type = head_type
        
        # Load pretrained CodeBERT encoder
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size
        
        # Initialize classification head
        if head_type == "mlp":
            self.head = MLPHead(
                hidden_size=hidden_size,
                num_labels=num_labels,
                dropout_prob=dropout_prob,
            )
        elif head_type == "cnn":
            self.head = CNNHead(
                hidden_size=hidden_size,
                num_labels=num_labels,
                dropout_prob=dropout_prob,
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}. Must be 'mlp' or 'cnn'.")
        
        # Handle encoder freezing
        if freeze_encoder:
            self._freeze_encoder()
        elif freeze_n_layers > 0:
            self._freeze_n_layers(freeze_n_layers)
        
        # Loss function with class imbalance handling
        self.pos_weight = pos_weight
        self.loss_fn = nn.CrossEntropyLoss(weight=pos_weight) if pos_weight is not None else nn.CrossEntropyLoss()
    
    def _freeze_encoder(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def _freeze_n_layers(self, n: int) -> None:
        """Freeze the first n encoder layers (embeddings + n transformer layers).
        
        Args:
            n: Number of layers to freeze from the bottom.
        """
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze first n transformer layers
        for i in range(min(n, len(self.encoder.encoder.layer))):
            for param in self.encoder.encoder.layer[i].parameters():
                param.requires_grad = False
    
    def set_head(self, head_type: Literal["mlp", "cnn"], dropout_prob: float = 0.1) -> None:
        """Switch classification head type.
        
        Args:
            head_type: Type of classification head ("mlp" or "cnn").
            dropout_prob: Dropout probability for new head.
        """
        hidden_size = self.config.hidden_size
        
        if head_type == "mlp":
            self.head = MLPHead(
                hidden_size=hidden_size,
                num_labels=self.num_labels,
                dropout_prob=dropout_prob,
            )
        elif head_type == "cnn":
            self.head = CNNHead(
                hidden_size=hidden_size,
                num_labels=self.num_labels,
                dropout_prob=dropout_prob,
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
        
        self.head_type = head_type
        
        # Move to same device as encoder
        device = next(self.encoder.parameters()).device
        self.head.to(device)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            token_type_ids: Token type IDs of shape (batch_size, seq_len).
            labels: Ground truth labels of shape (batch_size,). Optional.
            
        Returns:
            If labels provided: Tuple of (loss, logits).
            Otherwise: logits of shape (batch_size, num_labels).
        """
        # Encode input
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # Get representations for classification head
        if self.head_type == "mlp":
            # Use CLS token representation
            cls_output = encoder_outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
            logits = self.head(cls_output)
        else:
            # CNN head uses full sequence
            sequence_output = encoder_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
            logits = self.head(sequence_output)
        
        # Compute loss if labels provided
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        
        return logits
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
