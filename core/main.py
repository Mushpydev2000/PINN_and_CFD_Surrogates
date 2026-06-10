"""
Main Entry Point
================

Example script demonstrating PINN and surrogate model training.

Author: Scientific ML Research Group
Version: 1.0.0


ENCAPSULATED CLASSES & ABSTRACTIONS:
- None (Utility/Functional module)

DATA STRUCTURES & RATIONALE:
- PyTorch Tensors: Used extensively for automatic differentiation (autograd) and GPU acceleration.
- NumPy Arrays/Pandas DataFrames: Used for data processing and interfacing with visualization libraries.
- Dictionaries: Used for flexible configuration and mapping (e.g., loss histories, parameters).

DESIGN RATIONALE:
This file is part of the modular architecture separating physics definitions, neural networks,
and training pipelines. This separation of concerns allows for easy substitution of PDE
equations, network architectures, and training strategies without tight coupling.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path so `python core/main.py` works without installation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.utilities import logger, set_seed, load_config, resolve_device
from core.physics import NavierStokesEquations2D, DirichletBC, BoundaryConditionManager
from core.pinn import PINNNetwork, PINNTrainer
from core.surrogate import CFDDatasetGenerator, create_dataloaders, create_surrogate_model, SurrogateTrainer
from core.webapp import run_local_site


def train_pinn(config_path: str = "configs/pinn_config.yaml"):
    """Train Physics-Informed Neural Network."""
    logger.info("=" * 70)
    logger.info("PHYSICS-INFORMED NEURAL NETWORK (PINN) TRAINING")
    logger.info("=" * 70)
    
    # Load configuration
    config = load_config(config_path)
    set_seed(config['collocation']['seed'])
    
    logger.info(f"Loading config from {config_path}")
    
    # Initialize physics
    pde = NavierStokesEquations2D(
        rho=config['physics']['density'],
        mu=config['physics']['dynamic_viscosity'],
        reynolds_number=config['domain']['reynolds_number']
    )
    
    # Initialize boundary conditions
    bc_manager = BoundaryConditionManager()
    domain_bounds = {
        'x': tuple(config['domain']['x_bounds']),
        'y': tuple(config['domain']['y_bounds'])
    }
    
    # Add boundary conditions
    bc_manager.add_bc(DirichletBC('left', u_value=1.0, domain_bounds=domain_bounds))
    bc_manager.add_bc(DirichletBC('right', domain_bounds=domain_bounds))
    bc_manager.add_bc(DirichletBC('top', domain_bounds=domain_bounds))
    bc_manager.add_bc(DirichletBC('bottom', domain_bounds=domain_bounds))
    
    # Create PINN network
    model = PINNNetwork(
        n_inputs=config['network']['n_inputs'],
        n_outputs=config['network']['n_outputs'],
        n_hidden_layers=config['network']['n_hidden_layers'],
        n_neurons=config['network']['n_neurons'],
        activation=config['network']['activation'],
        use_fourier=True,
        weight_init=config['network']['weight_init']
    )
    
    logger.info(f"Created PINN with {sum(p.numel() for p in model.parameters())} parameters")
    
    requested_device = config['training']['device']
    device = resolve_device(requested_device)
    if requested_device == 'cuda' and device != 'cuda':
        logger.warning(f"CUDA unavailable; falling back to CPU")

    # Create trainer
    trainer = PINNTrainer(
        model=model,
        pde=pde,
        bc_manager=bc_manager,
        domain_bounds=domain_bounds,
        lr=config['training']['learning_rate'],
        device=device,
        checkpoint_dir='checkpoints',
        log_dir='logs'
    )
    
    logger.info("Starting PINN training...")
    
    # Train
    history = trainer.train(
        n_epochs=config['training']['epochs'],
        n_interior=config['collocation']['n_interior'],
        n_boundary=config['collocation']['n_boundary'],
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    
    logger.info("PINN training completed!")
    
    return trainer, history


def train_surrogate(config_path: str = "configs/surrogate_config.yaml"):
    """Train CFD surrogate model."""
    logger.info("=" * 70)
    logger.info("CFD SURROGATE MODEL TRAINING")
    logger.info("=" * 70)
    
    # Load configuration
    config = load_config(config_path)
    set_seed(config['dataset']['seed'])
    
    logger.info(f"Loading config from {config_path}")
    
    # Generate dataset
    logger.info("Generating CFD dataset...")
    generator = CFDDatasetGenerator(
        n_samples=config['dataset']['n_samples'],
        seed=config['dataset']['seed']
    )
    
    dataset = generator.generate_dataset()
    logger.info(f"Generated dataset with {len(dataset)} samples")
    
    # Create dataloaders
    input_features = config['dataset']['input_features']
    output_features = config['dataset']['output_features']
    
    train_loader, val_loader, test_loader, full_dataset = create_dataloaders(
        dataset,
        input_features=input_features,
        output_features=output_features,
        batch_size=config['training']['batch_size'],
        train_split=config['dataset']['train_split'],
        val_split=config['dataset']['val_split'],
        test_split=config['dataset']['test_split']
    )
    
    # Create surrogate model
    model = create_surrogate_model(
        model_type=config['models']['model_type'],
        n_inputs=len(input_features),
        n_outputs=len(output_features),
        n_hidden_layers=config['models']['deep_nn']['n_hidden_layers'],
        n_neurons=config['models']['deep_nn']['n_neurons'],
        activation=config['models']['deep_nn']['activation'],
        dropout_rate=config['models']['deep_nn']['dropout_rate'],
        use_batch_norm=config['models']['deep_nn']['use_batch_norm']
    )
    
    logger.info(f"Created surrogate model with {sum(p.numel() for p in model.parameters())} parameters")
    
    requested_device = config['training']['device']
    device = resolve_device(requested_device)
    if requested_device == 'cuda' and device != 'cuda':
        logger.warning("CUDA unavailable; falling back to CPU")

    # Create trainer
    trainer = SurrogateTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config['training']['learning_rate'],
        device=device,
        checkpoint_dir='checkpoints',
        log_dir='logs'
    )
    
    logger.info("Starting surrogate model training...")
    
    # Train
    history = trainer.train(
        n_epochs=config['training']['epochs'],
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    
    logger.info("Surrogate model training completed!")
    
    return trainer, history, test_loader


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train PINN or CFD surrogate model")
    parser.add_argument(
        '--mode', 
        choices=['pinn', 'surrogate', 'both', 'web'],
        default='both',
        help='Mode: pinn, surrogate, both, or web dashboard'
    )
    parser.add_argument(
        '--pinn-config',
        default='configs/pinn_config.yaml',
        help='Path to PINN config file'
    )
    parser.add_argument(
        '--surrogate-config',
        default='configs/surrogate_config.yaml',
        help='Path to surrogate config file'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to run the web dashboard on when using --mode web'
    )
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(42)
    
    try:
        if args.mode == 'web':
            logger.info("Starting local web dashboard...")
            run_local_site(port=args.port)
            return

        if args.mode in ['pinn', 'both']:
            logger.info("Training PINN model...")
            pinn_trainer, pinn_history = train_pinn(args.pinn_config)
        
        if args.mode in ['surrogate', 'both']:
            logger.info("Training surrogate model...")
            surr_trainer, surr_history, test_loader = train_surrogate(args.surrogate_config)
        
        logger.info("All training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
