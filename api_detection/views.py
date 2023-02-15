import pandas as pd
from rest_framework.views import APIView
import fraud_detection_system.settings as sett
from api_detection.models import digi_login
from django.http import JsonResponse
import json
import ast
from django.db import connections
from pymongo import MongoClient
from django.utils import timezone
from datetime import datetime
from bson import Decimal128
from decimal import Decimal

client = MongoClient('mongodb://<ip-address>:27017/')
db = client['<nama-database>']
digi_login = db['<nama-collection>']
main_trx = db['<nama-collection>']

def loginFraudDetection(data):
    model = sett.model
    scaler = sett.scale

    for i, row in data.iterrows():
        data_pred = [[int(row.customer_id), row.activity_date,  row.device_name,  row.device_type,  row.device_os, float(row.latitude), float(row.longitude)]]
        data_pred = pd.DataFrame(data_pred, columns =['customer_id', 'activity_date', 'device_name', 'device_type', 'device_os', 'latitude', 'longitude'])
        data_pred['activity_date'] = pd.to_datetime(data_pred['activity_date'], format='%Y-%m-%d %H:%M:%S.%f').apply(lambda x: x.timestamp())
        data_pred['device_name'] = data_pred['device_name'].apply(lambda x: sum(ord(c) for c in x))
        data_pred['device_type'] = data_pred['device_type'].apply(lambda x: sum(ord(c) for c in x))
        data_pred['device_os'] = data_pred['device_os'].apply(lambda x: sum(ord(c) for c in x))

        data_pred_scaled = scaler.transform(data_pred)
        result = model.predict(data_pred_scaled)

        data.at[i,'_result'] = result
        if (result==0):
            data['_isFraud'].at[i] = 0
        elif (result==1):
            data['_description'].at[i] = "['Aktivitas login dilakukan di lokasi yang jauh berbeda dari aktivitas login lain']"
            data['_isFraud'].at[i] = 1
        elif(result==2):
            data['_description'].at[i] = "['Aktivitas login dilakukan pada device yang berbeda']"
            data['_isFraud'].at[i] = 1
        elif(result==3):
            data['_description'].at[i] = "['Aktivitas login dilakukan pada device yang berbeda', 'Aktivitas login dilakukan di lokasi yang jauh berbeda dari aktivitas login lain']"
            data['_isFraud'].at[i] = 1

    data['_description'] = data['_description'].replace('nan', pd.np.nan)
    data['_result'] = data['_result'].astype(int)
    data['_isFraud'] = data['_isFraud'].astype(int)

class digiloginFraud(APIView):
    def get(self, request):
        customer_id = request.GET.get('customer_id')
        status = request.GET.get('status')
        start_date = request.GET.get('start_date')
        end_date = request.GET.get('end_date')

        if customer_id is not None:
            customer_id = str(customer_id)
            data_mongo = list(digi_login.find({"customer_id": customer_id}).sort("_id", -1).limit(10))
        elif start_date is not None and end_date is not None:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            if status is not None:
                status = int(status)
                data_mongo = list(digi_login.find({"activity_date": {"$gte": start_date, "$lte": end_date}, "_isFraud": status}).sort("_id", -1).limit(10))
            else:
                data_mongo = list(digi_login.find({"activity_date": {"$gte": start_date, "$lte": end_date}}).sort("_id", -1).limit(10))
        elif status is not None:
            status = int(status)
            data_mongo = list(digi_login.find({"_isFraud": status}).sort("_id", -1).limit(10))
        else:
            data_mongo = list(digi_login.find().sort("_id", -1).limit(10000))

        database = pd.DataFrame(data_mongo)
        database.drop(labels=['_id'], axis=1, inplace=True)

        try:
            database['_description'] = database['_description'].replace('', pd.np.nan)
            database['_description'] = database['_description'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        except KeyError:
            database['_description'] = pd.np.nan

        database['_result'] = database['_result'].astype(int)
        database['activity_date'] = database['activity_date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        database['_isFraud'] = database['_isFraud'].astype(int)

        data_result = database.to_json(orient='records')
        response = json.loads(data_result)
        return JsonResponse(response, safe=False)

    def post(self, request):
        data_request = request.data

        customer_id = data_request['customer_id']
        app_id = data_request['app_id']
        app_version = data_request['app_version']
        activity_date = timezone.now() + timezone.timedelta(hours=7)
        activity_date = activity_date.strftime("%Y-%m-%d %H:%M:%S.%f")
        device_name = data_request['device_name']
        device_type = data_request['device_type']
        device_os = data_request['device_os']
        latitude = data_request['latitude']
        longitude = data_request['longitude']
        location_desc = data_request['location_desc']
        data = [[customer_id, app_id, app_version, activity_date,  device_name,  device_type,  device_os, latitude, longitude, location_desc]]
        data = pd.DataFrame(data, columns =['customer_id', 'app_id', 'app_version', 'activity_date', 'device_name', 'device_type', 'device_os', 'latitude', 'longitude', 'location_desc'])
        
        data = data.assign(_result=0, _description={}, _isFraud=0)

        loginFraudDetection(data)

        data_dict = data.to_dict("records")
        digi_login.insert_many(data_dict)

        response = {
            'status': 'Data Baru Berhasil Ditambahkan'
        }
        return JsonResponse(response, safe=False)

def trxFraudDetection(data):
    model = sett.trx_model
    scaler = sett.trx_scale

    for i, row in data.iterrows():
        narrative = 0
        if not pd.isna(row['NARRATIVE1']) and 'tarik tunai' in row['NARRATIVE1']:
            narrative = 1
        data_pred = [[row.TIME_STAMP,  narrative]]
        data_pred = pd.DataFrame(data_pred, columns =['TIME_STAMP', 'NARRATIVE'])
        data_pred['TIME_STAMP'] = pd.to_datetime(data_pred['TIME_STAMP'], format='%Y-%m-%d %H:%M:%S.%f').apply(lambda x: x.timestamp())

        data_pred_scaled = scaler.transform(data_pred)
        result = model.predict(data_pred_scaled)

        data.at[i,'_result'] = result
        if (result==0):
            data['_isFraud'].at[i] = 0
        elif (result==14):
            data['_description'].at[i] = "['Transaksi Penarikan Tunai dengan kartu Debit di mesin ATM dengan Waktu diantara Pkl 00.01 WIB s.d Pkl 06.00 WIB']"
            data['_isFraud'].at[i] = 1

    data['_description'] = data['_description'].replace('nan', pd.np.nan)
    data['_result'] = data['_result'].astype(int)
    data['_isFraud'] = data['_isFraud'].astype(int)

class mainTRXFraud(APIView):
    def get(self, request):
        account = request.GET.get('account')
        cif = request.GET.get('cif')
        status = request.GET.get('status')
        start_date = request.GET.get('start_date')
        end_date = request.GET.get('end_date')
        channel = request.GET.get('channel')

        if account is not None:
            account = str(account)
            data_mongo = list(main_trx.find({"ACCOUNT": account}).sort("_id", -1).limit(10))
        elif cif is not None:
            cif = str(cif)
            data_mongo = list(main_trx.find({"CIF": cif}).sort("_id", -1).limit(10))

        elif start_date is not None and end_date is not None:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            if status is not None:
                status = int(status)
                if channel is not None:
                    channel = str(channel)
                    data_mongo = list(main_trx.find({"$and": [{"BUSS_DATE": {"$gte": start_date, "$lte": end_date}}, {"_isFraud": status}, {"USER_REF": {"$regex": "^" + channel}}]}).sort("_id", -1).limit(10))
                else:
                    data_mongo = list(main_trx.find({"BUSS_DATE": {"$gte": start_date, "$lte": end_date}, "_isFraud": status}).sort("_id", -1).limit(10))
            elif channel is not None:
                channel = str(channel)
                data_mongo = list(main_trx.find({"BUSS_DATE": {"$gte": start_date, "$lte": end_date}, "USER_REF": {"$regex": "^" + channel}}).sort("_id", -1).limit(10))
            else:
                data_mongo = list(main_trx.find({"BUSS_DATE": {"$gte": start_date, "$lte": end_date}}).sort("_id", -1).limit(10))

        elif status is not None:
            status = int(status)
            if channel is not None:
                channel = str(channel)
                data_mongo = list(main_trx.find({"$and": [{"USER_REF": {"$regex": "^" + channel}}, {"_isFraud": status}]}).sort("_id", -1).limit(10))
            else:
                data_mongo = list(main_trx.find({"_isFraud": status}).sort("_id", -1).limit(10))

        elif channel is not None:
            channel = str(channel)
            data_mongo = list(main_trx.find({"USER_REF": {"$regex": "^" + channel}}).sort("_id", -1).limit(10))

        else:
            data_mongo = list(main_trx.find().sort("_id", -1).limit(10000))

        database = pd.DataFrame(data_mongo)
        database.drop(labels=['_id'], axis=1, inplace=True)

        try:
            database['_description'] = database['_description'].replace('', pd.np.nan)
            database['_description'] = database['_description'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        except KeyError:
            database['_description'] = pd.np.nan

        database['BUSS_DATE'] = pd.to_datetime(database['BUSS_DATE'], unit='ms', utc=True)
        database['BUSS_DATE'] = database['BUSS_DATE'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        database['AMOUNT'] = database['AMOUNT'].astype(str)
        database['AMOUNT'] = database['AMOUNT'].astype(int)
        database['SEQ'] = database['SEQ'].astype(str)
        database['SEQ'] = database['SEQ'].astype(int)
        database['_result'] = database['_result'].astype(int)
        database['_isFraud'] = database['_isFraud'].astype(int)

        data_result = database.to_json(orient='records')
        response = json.loads(data_result)
        return JsonResponse(response, safe=False)
    
    def post(self, request):
        data_request = request.data

        NOW = timezone.now() + timezone.timedelta(hours=7)

        BUSS_DATE = data_request['BUSS_DATE']
        BRANCH = data_request['BRANCH']
        CIF = data_request['CIF']
        SUFIX = data_request['SUFIX']
        ACCOUNT = data_request['ACCOUNT']
        TRX_CODE = data_request['TRX_CODE']
        USER_REF = data_request['USER_REF']
        SIGN = data_request['SIGN']
        AMOUNT = data_request['AMOUNT']
        NARRATIVE1 = data_request['NARRATIVE1']
        GROUP_OR_USER_ID = data_request['GROUP_OR_USER_ID']
        TIME_STAMP = data_request['TIME_STAMP']
        SEQ = data_request['SEQ']
        BRANCH_INPUT = data_request['BRANCH_INPUT']
        CUSTOMER_TYPE = data_request['CUSTOMER_TYPE']
        JKartu = data_request['JKartu']

        AMOUNT = Decimal128(Decimal(AMOUNT))
        SEQ = Decimal128(Decimal(SEQ))

        data = [[BUSS_DATE, BRANCH, CIF,  SUFIX, ACCOUNT, TRX_CODE, USER_REF, SIGN, AMOUNT, NARRATIVE1, GROUP_OR_USER_ID,  TIME_STAMP, SEQ, BRANCH_INPUT, CUSTOMER_TYPE, JKartu]]
        data = pd.DataFrame(data, columns =["BUSS_DATE", "BRANCH", "CIF", "SUFIX", "ACCOUNT", "TRX_CODE", "USER_REF", "SIGN", "AMOUNT", "NARRATIVE1", "GROUP_OR_USER_ID", "TIME_STAMP", "SEQ", "BRANCH_INPUT", "CUSTOMER_TYPE", "JKartu"])
        data = data.assign(_result=0, _description={}, _isFraud=0)
        
        trxFraudDetection(data)

        data['BUSS_DATE'] = pd.to_datetime(data['BUSS_DATE'])
        data_dict = data.to_dict("records")
        main_trx.insert_many(data_dict)

        response = {
            'status': 'Data Baru Berhasil Ditambahkan'
        }
        return JsonResponse(response, safe=False)
    