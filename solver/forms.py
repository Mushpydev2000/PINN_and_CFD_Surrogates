from django import forms

class PINNSolverForm(forms.Form):
    reynolds_number = forms.FloatField(initial=100.0, label='Reynolds Number')
    density = forms.FloatField(initial=1.0, label='Density (rho)')
    viscosity = forms.FloatField(initial=0.01, label='Dynamic Viscosity (mu)')
    domain_x_min = forms.FloatField(initial=-1.0, label='Domain X Min')
    domain_x_max = forms.FloatField(initial=1.0, label='Domain X Max')
    domain_y_min = forms.FloatField(initial=-1.0, label='Domain Y Min')
    domain_y_max = forms.FloatField(initial=1.0, label='Domain Y Max')
    grid_resolution = forms.IntegerField(initial=50, label='Grid Resolution (nx, ny)')
    left_wall_velocity = forms.FloatField(initial=1.0, label='Left Wall Velocity (u)')
    n_epochs = forms.IntegerField(initial=500, label='Training Epochs (Quick Run)')

class SurrogatePredictorForm(forms.Form):
    MODEL_CHOICES = [
        ('deep', 'Deep Neural Network'),
        ('residual', 'Residual Network'),
        ('attention', 'Attention Network'),
    ]
    
    velocity = forms.FloatField(initial=10.0, label='Velocity (m/s)')
    angle_of_attack = forms.FloatField(initial=5.0, label='Angle of Attack (degrees)')
    reynolds_number = forms.FloatField(initial=1e5, label='Reynolds Number')
    mach_number = forms.FloatField(initial=0.3, label='Mach Number')
    turbulence_intensity = forms.FloatField(initial=0.05, label='Turbulence Intensity')
    surface_roughness = forms.FloatField(initial=0.001, label='Surface Roughness')
    model_type = forms.ChoiceField(choices=MODEL_CHOICES, initial='deep', label='Model Architecture')
