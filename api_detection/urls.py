from django.urls import path
from .views import digiloginFraud, mainTRXFraud

urlpatterns = [
    path('loginfraud/', digiloginFraud.as_view(), name = 'digilogin_fraud'),
    path('trxfraud/', mainTRXFraud.as_view(), name = 'maintrx_fraud'),
]