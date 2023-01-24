from django.urls import path
from .views import FraudDetection

urlpatterns = [
    path('fraud/', FraudDetection.as_view(), name = 'fraud_detection'),
]