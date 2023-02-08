from django.urls import path
from . import views

urlpatterns = [
    path('', views.home,name="home_page"),
    path('result/', views.result,name="result_page"),
    path('predict',views.predict,name="predict"),
]