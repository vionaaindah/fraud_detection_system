import pandas as pd
from rest_framework.views import APIView
import fraud_detection_system.settings as sett
from api_detection.models import digi_login_activity
from django.http import JsonResponse
import json
from django.shortcuts import render

class FraudDetection(APIView):
    def get(self, request):
        model = sett.model
        scaler = sett.scale

        database = digi_login_activity.objects.all().values()
        data = pd.DataFrame.from_records(database)
        data = data.assign(result=0, keterangan={})

        for i, row in data.iterrows():
            data_pred = [[int(row.customer_id), row.activity_date,  row.device_name,  row.device_type,  row.device_os, float(row.latitude), float(row.longitude)]]
            data_pred = pd.DataFrame(data_pred, columns =['customer_id', 'activity_date', 'device_name', 'device_type', 'device_os', 'latitude', 'longitude'])
            data_pred['activity_date'] = pd.to_datetime(data_pred['activity_date'], format='%Y-%m-%d %H:%M:%S.%f').apply(lambda x: x.timestamp())
            data_pred['device_name'] = data_pred['device_name'].apply(lambda x: sum(ord(c) for c in x))
            data_pred['device_type'] = data_pred['device_type'].apply(lambda x: sum(ord(c) for c in x))
            data_pred['device_os'] = data_pred['device_os'].apply(lambda x: sum(ord(c) for c in x))
            data_pred_scaled = scaler.transform(data_pred)

            result = model.predict(data_pred_scaled)

            if(result == 2):
                result = 0
            elif(result == 3):
                result = 1
            data.at[i,'result'] = result
            if (result==1):
                data['keterangan'].at[i] = pd.Series({1: 'Aktivitas login dilakukan di lokasi yang jauh berbeda dari aktivitas login lain'})
            
        data_result = data.to_json(orient='records')
        response = json.loads(data_result)
        return JsonResponse(response, safe=False)

class fraudDetection(APIView):
    def get(self, request):
        model = sett.model
        scaler = sett.scale

        database = digi_login_activity.objects.all().values()
        data = pd.DataFrame.from_records(database)
        data = data.assign(result=0, keterangan={})

        for i, row in data.iterrows():
            data_pred = [[int(row.customer_id), row.activity_date,  row.device_name,  row.device_type,  row.device_os, float(row.latitude), float(row.longitude)]]
            data_pred = pd.DataFrame(data_pred, columns =['customer_id', 'activity_date', 'device_name', 'device_type', 'device_os', 'latitude', 'longitude'])
            data_pred['activity_date'] = pd.to_datetime(data_pred['activity_date'], format='%Y-%m-%d %H:%M:%S.%f').apply(lambda x: x.timestamp())
            data_pred['device_name'] = data_pred['device_name'].apply(lambda x: sum(ord(c) for c in x))
            data_pred['device_type'] = data_pred['device_type'].apply(lambda x: sum(ord(c) for c in x))
            data_pred['device_os'] = data_pred['device_os'].apply(lambda x: sum(ord(c) for c in x))
            data_pred_scaled = scaler.transform(data_pred)

            result = model.predict(data_pred_scaled)

            data.at[i,'result'] = result
            if (result==1):
                data['keterangan'].at[i] = pd.Series({1: 'Aktivitas login dilakukan di lokasi yang jauh berbeda dari aktivitas login lain'})
            elif(result==2):
                data['keterangan'].at[i] = pd.Series({1: 'Aktivitas login dilakukan pada device yang berbeda'})
            elif(result==3):
                data['keterangan'].at[i] = pd.Series({1: 'Aktivitas login dilakukan pada device yang berbeda', 2: 'Aktivitas login dilakukan di lokasi yang jauh berbeda dari aktivitas login lain'})
       
        data_result = data.to_json(orient='records')
        response = json.loads(data_result)
        return JsonResponse(response, safe=False)

class fraudDetectionUI(APIView):
    def get(self, request):
        model = sett.model
        scaler = sett.scale
        database = digi_login_activity.objects.all().values()
        data = pd.DataFrame.from_records(database)
        data = data.drop(columns='id')

        data = data.assign(result=0, keterangan='')

        for i, row in data.iterrows():
            data_pred = [[int(row.customer_id), row.activity_date,  row.device_name,  row.device_type,  row.device_os, float(row.latitude), float(row.longitude)]]
            data_pred = pd.DataFrame(data_pred, columns =['customer_id', 'activity_date', 'device_name', 'device_type', 'device_os', 'latitude', 'longitude'])
            data_pred['activity_date'] = pd.to_datetime(data_pred['activity_date'], format='%Y-%m-%d %H:%M:%S.%f').apply(lambda x: x.timestamp())
            data_pred['device_name'] = data_pred['device_name'].apply(lambda x: sum(ord(c) for c in x))
            data_pred['device_type'] = data_pred['device_type'].apply(lambda x: sum(ord(c) for c in x))
            data_pred['device_os'] = data_pred['device_os'].apply(lambda x: sum(ord(c) for c in x))
            data_pred_scaled = scaler.transform(data_pred)
            result = model.predict(data_pred_scaled)
            data.at[i,'result'] = result

            if (result==1):
                data['keterangan'].at[i] = pd.Series({1: 'Aktivitas login dilakukan di lokasi yang jauh berbeda dari aktivitas login lain'})
            elif(result==2):
                data['keterangan'].at[i] = pd.Series({1: 'Aktivitas login dilakukan pada device yang berbeda'})
            elif(result==3):
                data['keterangan'].at[i] = pd.Series({1: 'Aktivitas login dilakukan pada device yang berbeda', 2: 'Aktivitas login dilakukan di lokasi yang jauh berbeda dari aktivitas login lain'})
       
        return render(request,'index.html', {'data': data.to_html(), 'num_columns': range(data.shape[1])})
