from django.urls import path
from .views import digiloginFraud, mainTRXFraud, trainingModel, digiloginDyamicFraud

urlpatterns = [
    path('loginfraud/', digiloginFraud.as_view(), name = 'digilogin_fraud'),
    path('trxfraud/', mainTRXFraud.as_view(), name = 'maintrx_fraud'),
    path('training_model/', trainingModel.as_view(), name = 'training_model'),
    path('digiloginDyamicFraud/', digiloginDyamicFraud.as_view(), name = 'digiloginDyamicFraud'),

]
