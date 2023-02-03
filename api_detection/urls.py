from django.urls import path
from .views import fraudDetection

urlpatterns = [
    path('fraud/', fraudDetection.as_view(), name = 'fraud_detection'),
]