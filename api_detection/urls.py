from django.urls import path
from .views import FraudDetection, fraudDetection

urlpatterns = [
    path('fraud/', FraudDetection.as_view(), name = 'fraud_detection'),
    path('fraud2/', fraudDetection.as_view(), name = 'fraud2_detection'),
]