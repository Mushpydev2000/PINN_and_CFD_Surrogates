from django.urls import path
from . import views

app_name = 'solver'
urlpatterns = [
    path('', views.index_view, name='index'),
    path('pinn/', views.pinn_solver_view, name='pinn_solver'),
    path('surrogate/', views.surrogate_view, name='surrogate'),
]
