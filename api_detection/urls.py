from django.urls import path
from .views import loginFraud, digiloginFraud

urlpatterns = [
    path('fraud2/', loginFraud.as_view(), name = 'login_fraud'),
    path('loginfraud/', digiloginFraud.as_view(), name = 'digilogin_fraud'),
]