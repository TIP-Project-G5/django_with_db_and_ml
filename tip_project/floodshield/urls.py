from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_page_view, name='home'),
    path('ml_test/', views.ml_test, name='ml_test'),
]
