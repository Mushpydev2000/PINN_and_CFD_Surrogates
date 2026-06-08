"""Unit tests for PINN module."""

import pytest
import torch
from core.pinn import PINNNetwork, ImprovedPINNNetwork, FourierFeatures


class TestFourierFeatures:
    """Test Fourier feature encoding."""
    
    def test_initialization(self):
        """Test Fourier features initialization."""
        ff = FourierFeatures(n_freqs=8)
        assert ff.n_freqs == 8
    
    def test_forward_pass(self):
        """Test Fourier encoding forward pass."""
        ff = FourierFeatures(n_freqs=8)
        x = torch.tensor([[0.5, 0.3]], dtype=torch.float32)
        
        encoded = ff(x)
        
        # Output shape should be [batch, n_dims * n_freqs * 2]
        expected_shape = (1, 2 * 8 * 2)
        assert encoded.shape == expected_shape


class TestPINNNetwork:
    """Test PINN network architecture."""
    
    def test_initialization(self):
        """Test PINN network initialization."""
        model = PINNNetwork(
            n_inputs=2,
            n_outputs=3,
            n_hidden_layers=4,
            n_neurons=128
        )
        
        # Test parameter count
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
    
    def test_forward_pass(self):
        """Test PINN forward pass."""
        model = PINNNetwork(n_inputs=2, n_outputs=3)
        x = torch.randn(10, 2)  # Batch of 10 samples
        
        output = model(x)
        
        # Output shape should be [batch_size, n_outputs]
        assert output.shape == (10, 3)
    
    def test_output_range(self):
        """Test output values are reasonable."""
        model = PINNNetwork(n_inputs=2, n_outputs=3)
        x = torch.randn(100, 2)
        
        output = model(x)
        
        # For tanh activation, outputs should be bounded
        assert torch.all(output >= -1.5) and torch.all(output <= 1.5)


class TestImprovedPINNNetwork:
    """Test improved PINN with batch norm."""
    
    def test_batch_norm_network(self):
        """Test improved PINN with batch normalization."""
        model = ImprovedPINNNetwork(
            n_inputs=2,
            n_outputs=3,
            use_batch_norm=True,
            dropout_rate=0.1
        )
        
        x = torch.randn(32, 2)
        output = model(x, training=True)
        
        assert output.shape == (32, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
