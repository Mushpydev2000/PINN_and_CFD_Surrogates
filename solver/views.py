from django.shortcuts import render
from .forms import PINNSolverForm, SurrogatePredictorForm
from .services import PINNService, SurrogateService

def index_view(request):
    return render(request, 'solver/index.html')

def pinn_solver_view(request):
    context = {'form': PINNSolverForm()}
    if request.method == 'POST':
        form = PINNSolverForm(request.POST)
        if form.is_valid():
            results = PINNService.run_pinn_solver(form.cleaned_data)
            context['results'] = results
            context['form'] = form
    return render(request, 'solver/pinn_solver.html', context)

def surrogate_view(request):
    context = {'form': SurrogatePredictorForm()}
    if request.method == 'POST':
        form = SurrogatePredictorForm(request.POST)
        if form.is_valid():
            results = SurrogateService.run_surrogate_prediction(form.cleaned_data)
            context['results'] = results
            context['form'] = form
    return render(request, 'solver/surrogate.html', context)
