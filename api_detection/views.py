import pandas as pd
from rest_framework.views import APIView
import fraud_detection_system.settings as sett
from api_detection.models import digi_login, test_data
from django.http import JsonResponse
import json
from django.db import connections
import ast
from django.utils import timezone 
from pymongo import MongoClient

class loginFraud(APIView):
    def get(self, request):
        database = digi_login.objects.all().order_by('-activity_date').values()
        database = pd.DataFrame.from_records(database)
        database['keterangan'] = database['keterangan'].replace('nan', pd.np.nan)
        database['keterangan'] = database['keterangan'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        database['result'] = database['result'].astype(int)

        database['activity_date'] = database['activity_date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

        data_result = database.to_json(orient='records')
        response = json.loads(data_result)
        return JsonResponse(response, safe=False)
    
    def post(self, request):
        conn = connections['default']

        data_request = request.data
        
        model = sett.login_model
        scaler = sett.login_scaler

        customer_id = int(data_request['customer_id'])
        app_id = data_request['app_id']
        app_version = data_request['app_version']
        activity_date = timezone.now()
        activity_date = activity_date.strftime("%Y-%m-%d %H:%M:%S.%f")
        device_name = data_request['device_name']
        device_type = data_request['device_type']
        device_os = data_request['device_os']
        latitude = float(data_request['latitude'])
        longitude = float(data_request['longitude'])
        location_desc = data_request['location_desc']

        data = [[customer_id, app_id, app_version, activity_date,  device_name,  device_type,  device_os, latitude, longitude, location_desc]]
        data = pd.DataFrame(data, columns =['customer_id', 'app_id', 'app_version', 'activity_date', 'device_name', 'device_type', 'device_os', 'latitude', 'longitude', 'location_desc'])
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
        
        for i, row in data.iterrows():
            obj = digi_login(
                customer_id=row['customer_id'],
                app_id=row['app_id'],
                app_version=row['app_version'],
                activity_date=row['activity_date'],
                device_name=row['device_name'],
                device_type=row['device_type'],
                device_os=row['device_os'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                location_desc=row['location_desc'],
                result=row['result'],
                keterangan=row['keterangan']              
                )
            obj.save()
        conn.commit()

        database = digi_login.objects.all().order_by('-activity_date').values()
        database = pd.DataFrame.from_records(database)
        database['keterangan'] = database['keterangan'].replace('nan', pd.np.nan)
        database['keterangan'] = database['keterangan'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        database['result'] = database['result'].astype(int)

        database['activity_date'] = database['activity_date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

        data_result = database.to_json(orient='records')
        response = json.loads(data_result)
        return JsonResponse(response, safe=False)
    

client = MongoClient('mongodb://[IP-ADDRESS]:27017/')
db = client['[NAMA-DB]']
collection = db['[NAMA-COLLECTION]']


class digiloginFraud(APIView):
    def get(self, request):
        
        model = sett.login_model
        scaler = sett.login_scaler

        data = test_data.objects.all().order_by('-activity_date').values()
        data = pd.DataFrame.from_records(data)
        data.drop(labels=['id'], axis=1, inplace=True)
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
                data['keterangan'].at[i] = ['Aktivitas login dilakukan di lokasi yang jauh berbeda dari aktivitas login lain']
            elif(result==2):
                data['keterangan'].at[i] = ['Aktivitas login dilakukan pada device yang berbeda']
            elif(result==3):
                data['keterangan'].at[i] = ['Aktivitas login dilakukan pada device yang berbeda', 'Aktivitas login dilakukan di lokasi yang jauh berbeda dari aktivitas login lain']

        data['keterangan'] = data['keterangan'].replace('nan', pd.np.nan)
        data['result'] = data['result'].astype(int)

        data['activity_date'] = data['activity_date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            
        data_dict = data.to_dict("records")
        collection.insert_many(data_dict)


        response = {
            'status': 'ok'
        }

        return JsonResponse(response, safe=False)
