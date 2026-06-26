import logging
import re
from typing import Dict, List, Optional, Set, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: float = 8, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.A = nn.Parameter(torch.randn(base_layer.in_features, r) / r)
        self.B = nn.Parameter(torch.zeros(r, base_layer.out_features))
        
        for p in self.base_layer.parameters():
            p.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + ((self.dropout(x) @ self.A) @ self.B) * self.scale
    
    def merge(self):
        with torch.no_grad():
            self.base_layer.weight.data += (self.B.T @ self.A.T) * self.scale
            del self.A, self.B
            self.forward = self.base_layer.forward


class LoraWrapper(nn.Module):
    """Wrapper that adds LoRA to a base model based on configuration."""
    
    def __init__(self, base_model: nn.Module, config_lora: Dict):
        super().__init__()
        
        # base model
        self.base_model = base_model
        
        # LoRA configuration
        self.r = config_lora.get('r', 8)
        self.alpha = config_lora.get('alpha', None)
        if self.alpha is None:
            self.alpha = self.r  # Default: scale = 1.0
        self.dropout = config_lora.get('dropout', 0.0)
        self.target_modules = config_lora.get('target_modules', [])
        
        # Add LoRA Layers to target modules
        self._add_lora_layers()
        
        # Log initialization
        logger.info(f"LoraWrapper initialized with r={self.r}, alpha={self.alpha}, dropout={self.dropout}")
    
    def _add_lora_layers(self):
        patterns = [re.compile(t) if any(c in t for c in ".*+?[]") else t for t in self.target_modules]
        modified = []
        
        for name, module in list(self.base_model.named_modules()):
            if len(list(module.children())) > 0 or not isinstance(module, nn.Linear):
                continue
            
            if any(p.match(name) if hasattr(p, 'match') else name.endswith(p) for p in patterns):
                
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = self.base_model
                if parent_name:
                    for part in parent_name.split("."):
                        parent = getattr(parent, part)
                
                # Replace with LoRA layer
                setattr(parent, child_name, LoRALayer(module, self.r, self.alpha, self.dropout))
                modified.append(name)
        
        logger.info(f"Added LoRA to {len(modified)} layers")
        logger.info(f"Modified layers: {modified}")
    
    def forward(self, *args, **kwargs):
        """Forward pass through the base model."""
        return self.base_model(*args, **kwargs)
    
    def get_lora_params(self) -> List[nn.Parameter]:
        """Get all LoRA parameters for optimization."""

        lora_params = []
        for name, module in self.named_modules():
            if isinstance(module, LoRALayer):
                if hasattr(module, 'A'):
                    lora_params.append(module.A)
                if hasattr(module, 'B'):
                    lora_params.append(module.B)
        return lora_params
    
    def save_lora_weights(self, path: str):
        """Save only the LoRA weights."""
        state = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRALayer):
                # Remove 'base_model.' prefix if present
                clean_name = name.replace('base_model.', '')
                state[f"{clean_name}.A"] = module.A.data
                state[f"{clean_name}.B"] = module.B.data
        
        torch.save(state, path)
        logger.info(f"Saved {len(state)//2} LoRA adapters to {path}")
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights from file."""
        state = torch.load(path, map_location='cpu')
        
        # Map state dict keys to current model structure
        current_state = self.state_dict()
        mapped_state = {}
        
        for key, value in state.items():
            # Try with and without 'base_model.' prefix
            if key in current_state:
                mapped_state[key] = value
            elif f"base_model.{key}" in current_state:
                mapped_state[f"base_model.{key}"] = value
        
        missing, unexpected = self.load_state_dict(mapped_state, strict=False)
        
        logger.info(f"Loaded {len(state)//2} LoRA adapters from {path}")
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
    
    def merge_and_unload(self):
        """Merge LoRA weights into base model and remove LoRA layers. Used for inference."""
        for module in self.modules():
            if isinstance(module, LoRALayer):
                module.merge()
        logger.info("Merged all LoRA weights into base model")
    
    def __getattr__(self, name):
        """Forward attribute access to base model if not found in wrapper."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
    
    @property
    def device(self):
        """Get device of the model."""
        return next(self.parameters()).device
    
    def state_dict(self, *args, **kwargs):
        """Override to handle state dict properly."""
        # Get base model state dict
        state = super().state_dict(*args, **kwargs)
        
        # Ensure LoRA parameters are included
        for name, module in self.named_modules():
            if isinstance(module, LoRALayer):
                if hasattr(module, 'A'):
                    state[f"{name}.A"] = module.A
                if hasattr(module, 'B'):
                    state[f"{name}.B"] = module.B
        
        return state