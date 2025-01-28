from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('chat/', views.chat_interface, name='chat'),
    #path('response/', views.chat_response, name='chat'),
]