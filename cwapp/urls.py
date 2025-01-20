from django.urls import path

from . import views

app_name = "cwapp"
urlpatterns = [
    path("<str:version>/", views.IndexView, name="index"),
    path("<str:version>", views.IndexView, name="index"),
    path('process-images', views.process_images, name='process_images'),
]
