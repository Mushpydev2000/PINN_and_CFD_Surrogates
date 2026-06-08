import io
import base64
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.physics import NavierStokesEquations2D, BoundaryConditionManager, DirichletBC
from core.pinn import PINNNetwork, PINNTrainer, PINNInference
from core.surrogate import create_surrogate_model, CFDDatasetGenerator
from core.visualization import plot_contour, plot_vector_field

def get_base64_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image_base64

class PINNService:
    @staticmethod
    def run_pinn_solver(data):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. Physics Setup
        pde = NavierStokesEquations2D(
            rho=data['density'],
            mu=data['viscosity'],
            reynolds_number=data['reynolds_number']
        )
        
        # 2. Domain & Boundary Conditions
        domain_bounds = {
            'x': (data['domain_x_min'], data['domain_x_max']),
            'y': (data['domain_y_min'], data['domain_y_max'])
        }
        
        bc_manager = BoundaryConditionManager()
        bc_manager.add_bc(DirichletBC('left', u_value=data['left_wall_velocity'], domain_bounds=domain_bounds))
        bc_manager.add_bc(DirichletBC('right', domain_bounds=domain_bounds))
        bc_manager.add_bc(DirichletBC('top', domain_bounds=domain_bounds))
        bc_manager.add_bc(DirichletBC('bottom', domain_bounds=domain_bounds))
        
        # 3. Model
        model = PINNNetwork(
            n_inputs=2,
            n_outputs=3,
            n_hidden_layers=4,
            n_neurons=64,
            activation='tanh',
            use_fourier=True
        )
        
        # 4. Trainer (Quick run)
        trainer = PINNTrainer(
            model=model,
            pde=pde,
            bc_manager=bc_manager,
            domain_bounds=domain_bounds,
            lr=0.005,
            device=device
        )
        
        # Train for requested epochs
        trainer.train(n_epochs=data['n_epochs'], n_interior=1000, n_boundary=400, log_interval=data['n_epochs']+1)
        
        # 5. Inference & Plotting
        inference = PINNInference(model, device=device)
        X, Y, u, v, p = inference.predict_on_grid(
            (data['domain_x_min'], data['domain_x_max']),
            (data['domain_y_min'], data['domain_y_max']),
            nx=data['grid_resolution'],
            ny=data['grid_resolution']
        )
        
        # Generate Plots
        fig_vel = plot_vector_field(X, Y, u, v, title="Velocity Field")
        fig_p = plot_contour(X, Y, p, title="Pressure Distribution")
        
        return {
            'velocity_plot': get_base64_plot(fig_vel),
            'pressure_plot': get_base64_plot(fig_p),
            'final_loss': trainer.loss_history['total'][-1] if trainer.loss_history['total'] else 0.0
        }

class SurrogateService:
    @staticmethod
    def run_surrogate_prediction(data):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Generate CFD dataset locally to train a small model for demonstration
        # In a real scenario, this would load a pre-trained model.
        generator = CFDDatasetGenerator(n_samples=500, seed=42)
        dataset_df = generator.generate_dataset()
        
        input_features = ['velocity', 'angle_of_attack', 'reynolds_number', 'mach_number', 'turbulence_intensity', 'surface_roughness']
        output_features = ['drag_coefficient', 'lift_coefficient']
        
        # A quick fake prediction based on the synthetic generation logic for demonstration
        input_dict = {
            'velocity': data['velocity'],
            'angle_of_attack': data['angle_of_attack'],
            'reynolds_number': data['reynolds_number'],
            'mach_number': data['mach_number'],
            'turbulence_intensity': data['turbulence_intensity'],
            'surface_roughness': data['surface_roughness']
        }
        
        # Using the actual generator logic to give physically plausible answers
        # in absence of a fully pre-trained model loading mechanism.
        coeffs = generator.generate_aerodynamic_coefficients(input_dict)
        
        # Create a simple bar plot to show the coefficients
        fig, ax = plt.subplots(figsize=(6, 4))
        categories = ['Drag Coefficient', 'Lift Coefficient']
        values = [coeffs['drag_coefficient'], coeffs['lift_coefficient']]
        ax.bar(categories, values, color=['#ff9999','#66b3ff'])
        ax.set_ylabel('Coefficient Value')
        ax.set_title('Predicted Aerodynamic Coefficients')
        
        return {
            'drag': coeffs['drag_coefficient'],
            'lift': coeffs['lift_coefficient'],
            'plot': get_base64_plot(fig)
        }
