"""Unit tests for surrogate model module."""

import pytest
import torch
import pandas as pd
from core.surrogate import (
    CFDDatasetGenerator, CFDDataset, create_dataloaders,
    create_surrogate_model, MLPSurrogate, DeepSurrogate
)
from solver.services import SurrogateService


class TestCFDDatasetGenerator:
    """Test CFD dataset generation."""
    
    def test_initialization(self):
        """Test generator initialization."""
        gen = CFDDatasetGenerator(n_samples=100, seed=42)
        assert gen.n_samples == 100
    
    def test_dataset_generation(self):
        """Test dataset generation."""
        gen = CFDDatasetGenerator(n_samples=50)
        df = gen.generate_dataset()
        
        assert len(df) == 50
        assert 'drag_coefficient' in df.columns
        assert 'velocity' in df.columns
    
    def test_reynolds_number_computation(self):
        """Test Reynolds number computation."""
        gen = CFDDatasetGenerator()
        re_array = torch.linspace(100, 10000, 10).numpy()
        
        # Should not raise error
        cd = gen.cylinder_drag_coefficient(re_array)
        assert len(cd) == len(re_array)

    def test_scalar_aerodynamic_coefficients(self):
        """Test one-off web form scalar prediction inputs."""
        gen = CFDDatasetGenerator(seed=42)
        coeffs = gen.generate_aerodynamic_coefficients(
            {
                'velocity': 10.0,
                'angle_of_attack': 5.0,
                'reynolds_number': 100000.0,
                'mach_number': 0.3,
                'turbulence_intensity': 0.05,
                'surface_roughness': 0.001,
            }
        )

        assert coeffs['drag_coefficient'].shape == (1,)
        assert coeffs['lift_coefficient'].shape == (1,)

    def test_surrogate_service_returns_template_safe_scalars(self):
        """Test the Django service returns Python floats for template filters."""
        results = SurrogateService.run_surrogate_prediction(
            {
                'velocity': 10.0,
                'angle_of_attack': 5.0,
                'reynolds_number': 100000.0,
                'mach_number': 0.3,
                'turbulence_intensity': 0.05,
                'surface_roughness': 0.001,
                'model_type': 'deep',
            }
        )

        assert isinstance(results['drag'], float)
        assert isinstance(results['lift'], float)
        assert results['plot']


class TestCFDDataset:
    """Test CFD dataset loader."""
    
    def test_dataset_creation(self):
        """Test CFD dataset creation."""
        gen = CFDDatasetGenerator(n_samples=100)
        df = gen.generate_dataset()
        
        dataset = CFDDataset(
            df,
            input_features=['velocity', 'reynolds_number'],
            output_features=['drag_coefficient'],
            normalize=True
        )
        
        assert len(dataset) == 100
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        gen = CFDDatasetGenerator(n_samples=50)
        df = gen.generate_dataset()
        
        dataset = CFDDataset(
            df,
            input_features=['velocity', 'reynolds_number'],
            output_features=['drag_coefficient']
        )
        
        x, y = dataset[0]
        
        assert x.shape == (2,)
        assert y.shape == (1,)


class TestSurrogateModels:
    """Test surrogate model architectures."""
    
    def test_mlp_surrogate(self):
        """Test MLP surrogate model."""
        model = MLPSurrogate(n_inputs=2, n_outputs=1, n_neurons=64)
        
        x = torch.randn(32, 2)
        output = model(x)
        
        assert output.shape == (32, 1)
    
    def test_deep_surrogate(self):
        """Test deep surrogate model."""
        model = DeepSurrogate(
            n_inputs=2,
            n_outputs=1,
            n_hidden_layers=6,
            n_neurons=128,
            use_batch_norm=True
        )
        
        x = torch.randn(32, 2)
        output = model(x)
        
        assert output.shape == (32, 1)
    
    def test_model_factory(self):
        """Test model creation factory."""
        model_mlp = create_surrogate_model('mlp', n_inputs=2, n_outputs=1)
        model_deep = create_surrogate_model('deep', n_inputs=2, n_outputs=1)
        
        x = torch.randn(10, 2)
        out_mlp = model_mlp(x)
        out_deep = model_deep(x)
        
        assert out_mlp.shape == (10, 1)
        assert out_deep.shape == (10, 1)


class TestDataLoaders:
    """Test data loading functionality."""
    
    def test_dataloader_creation(self):
        """Test dataloader creation."""
        gen = CFDDatasetGenerator(n_samples=100)
        df = gen.generate_dataset()
        
        train_loader, val_loader, test_loader, full_dataset = create_dataloaders(
            df,
            input_features=['velocity', 'reynolds_number'],
            output_features=['drag_coefficient'],
            batch_size=32
        )
        
        # Check that sizes sum to original
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)
        
        assert train_size + val_size + test_size == 100
        assert train_size > val_size
        assert train_size > test_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
