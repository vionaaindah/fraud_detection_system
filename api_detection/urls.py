from django.urls import path
from .views import trainingModel, digiloginDyamicFraud, mainTRXDyamicFraud

urlpatterns = [
    path('training_model/', trainingModel.as_view(), name = 'training_model'),
    path('digiloginDyamicFraud/', digiloginDyamicFraud.as_view(), name = 'digiloginDyamicFraud'),
    path('mainTRXDyamicFraud/', mainTRXDyamicFraud.as_view(), name = 'mainTRXDyamicFraud'),

]
