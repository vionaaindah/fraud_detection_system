from django.db import models

# Create your models here.
class digi_login(models.Model):
    customer_id = models.CharField(max_length=37)
    activity_date = models.DateTimeField(blank=True, null=True)
    app_id = models.CharField(max_length=64, blank=True, null=True)
    app_version = models.CharField(max_length=16, blank=True, null=True)
    device_name = models.CharField(max_length=32, blank=True, null=True)
    device_type = models.CharField(max_length=32, blank=True, null=True)
    device_os = models.CharField(max_length=32, blank=True, null=True)
    latitude = models.CharField(max_length=25, blank=True, null=True)
    longitude = models.CharField(max_length=25, blank=True, null=True)
    location_desc = models.TextField(blank=True, null=True)