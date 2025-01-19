from django.urls import path

from . import views

app_name = "polls"
urlpatterns = [
    path("", views.IndexView, name="index"),
    path('process-images/', views.process_images, name='process_images'),
]
