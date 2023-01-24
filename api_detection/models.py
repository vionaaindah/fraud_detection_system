from django.db import models

# Create your models here.

class digi_login_activity(models.Model):
    customer_id = models.CharField(max_length=37)
    activity_date = models.DateTimeField()
    app_id = models.CharField(max_length=64)
    app_version = models.CharField(max_length=16)
    device_name = models.CharField(max_length=32)
    device_type = models.CharField(max_length=32)
    device_os = models.CharField(max_length=32)
    latitude = models.CharField(max_length=20)
    longitude = models.CharField(max_length=20)
    location_desc = models.TextField()
