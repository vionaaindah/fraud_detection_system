from django.urls import path
from .views import FraudDetection, fraudDetection, fraudDetectionUI

urlpatterns = [
    path('fraud/', FraudDetection.as_view(), name = 'fraud_detection'),
    path('fraud2/', fraudDetection.as_view(), name = 'fraud2_detection'),
    path('ui/', fraudDetectionUI.as_view(), name = 'fraud_detection_ui'),
]