from django.urls import path

from . import views

app_name = "cwapp"
urlpatterns = [
    path('process-images/', views.process_images, name='process_images'),
    path("version=<str:version>/", views.IndexView, name="index"),
]
