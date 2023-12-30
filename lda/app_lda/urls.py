from django.urls import path
from . import views


urlpatterns = [
    path('', views.input),
    path('adminhome/', views.adminhome),
    path('view/', views.viewdata)
]
