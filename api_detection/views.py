import os
import ast
import json
import operator
import pickle
import psycopg2
import pandas as pd
import fraud_detection_system.settings as sett

from bson import Decimal128
from decimal import Decimal
from datetime import datetime
from pymongo import MongoClient
from rest_framework.views import APIView
# from api_detection.models import transaksi, test_data, digi_login
from django.http import JsonResponse
from django.db import connections
from django.utils import timezone

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

client = MongoClient('mongodb://<ip-address>:27017/')
db = client['<nama-database>']
digi_login = db['<nama-collection>']
main_trx = db['<nama-collection>']

# Koneksi ke database
conn_public = psycopg2.connect(
    host="<ip-address>",
    database="<nama-database>",
    user="<nama_user>",
    password="<password>",
    options="-c search_path=<nama_skema>"
)

# Koneksi ke database
conn_collection = psycopg2.connect(
    host="<ip-address>",
    database="<nama-database>",
    user="<nama_user>",
    password="<password>",
    options="-c search_path=<nama_skema>"
)

def label_decs(data, names):
    for idx, name in enumerate(names):
        if data == 0:
            decs =  ""
        elif data == idx+1:
            decs =  "['{}']".format(name)
        else:
            str_data = f'{data}'
            label = str_data[3:]
            result = []
            for i in range(0, len(label), 2):
                for idx, name in enumerate(names):
                    if int(label[i:i+2]) == idx+1:
                        result.append(name)
            if result:
                decs =  "{}".format(result)
    return decs

def label_stat(data):
    if data == 0:
        return 0
    else:
        return 1
            
class trainingModel(APIView):
    def post(self, request):
        cur_public = conn_public.cursor()
        cur_collection = conn_collection.cursor()
        
        id_rules = request.query_params.get('id_rules', None)
        id_rules = str(id_rules)
        cur_public.execute("SELECT table_id FROM master_rules WHERE id = '{}'".format(id_rules))
        result = cur_public.fetchone()
        table = result[0]

        cur_public.execute("SELECT name, configurations FROM master_rules WHERE table_id = '{}'".format(table))
        results = cur_public.fetchall()

        rules = []
        names = []
        for result in results:
            names.append(result[0])
            rules.append(result[1])

        cur_public.execute("SELECT name FROM master_tables WHERE id = '{}'".format(table))
        nama_tabel = cur_public.fetchone()
        nama_tabel = nama_tabel[0]

        pickle.dump(names, open("{}_rulename.pkl".format(nama_tabel), "wb"))

        collection = db['{}'.format(nama_tabel)]
        data_mongo = list(collection.find())

        if len(data_mongo) > 0:
            dataframe = pd.DataFrame(data_mongo)
            dataframe.drop(labels=['_id', '_result', '_description', '_isFraud'], axis=1, inplace=True)
            collection.delete_many({})
        else:
            cur_collection.execute("SELECT * FROM {}".format(nama_tabel))
            result = cur_collection.fetchall()
            dataframe = pd.DataFrame.from_records(result, columns=[desc[0] for desc in cur_collection.description])
            dataframe.drop(labels=['id'], axis=1, inplace=True)

        try:
            dataframe['activity_date'] = pd.to_datetime(dataframe['activity_date'], utc=True)
        except KeyError:
            dataframe['TIME_STAMP'] = pd.to_datetime(dataframe['TIME_STAMP'], utc=True)
        
        def not_contains(a, b):
            return not a.__contains__(b)

        op_map = {
            'LESS_THAN': 'lt',
            'LESS_EQUALS': 'le',
            'GREATER_THAN': 'gt',
            'GREATER_EQUALS': 'ge',
            'EQUALS': 'eq',
            'NOT_EQUALS': 'ne',
            'IN': 'contains'
        }

        val_list = []
        field_list = []
        operator_list = []

        for rule in rules:
            sub_val_list = []
            sub_field_list = []
            sub_op_list = []
            for r in rule:
                field = r['field']
                sub_field_list.append(field)
                try:
                    val = datetime.strptime(r["value"], '%H:%M').time()
                except ValueError:
                    val = r["value"]
                sub_val_list.append(val)
                if r['operator'] == 'NOT_IN':
                    op_func = not_contains
                else:
                    op_func = getattr(operator, op_map[r['operator']])
                sub_op_list.append(op_func)
            field_list.append(sub_field_list)
            val_list.append(sub_val_list)
            operator_list.append(sub_op_list)

        def labeling(row):
            fields = [[row[field[i]] for i in range(len(field))] for field in field_list]
            label = 0
            for i in range(len(rules)):
                label_i = True
                for j in range(len(rules[i])):
                    op_func = operator_list[i][j]
                    val = val_list[i][j]
                    field = fields[i][j]
                    try:                
                        field = field.time()
                    except AttributeError:
                        pass
                    if not op_func(field, val):
                        label_i = False
                        break
                if label_i:
                    if label != 0:
                        if label < 10:
                            l_old = '0' + f'{label}'
                            l_new = '0' + f'{label+1}'
                        else:
                            l_old = f'{label}'
                            l_new = f'{label+1}'
                        lbl = '100' + l_old + l_new
                        label = int(lbl)
                    else:
                        label = i+1
            return label
        dataframe['_result'] = dataframe.apply(labeling, axis=1)

        dataframe['_description'] = dataframe['_result'].apply(label_decs, names=names)

        dataframe['_isFraud'] = dataframe['_result'].apply(label_stat)

        data_dict = dataframe.to_dict("records")
        collection.insert_many(data_dict)

        unique_fields = []
        for sublist in field_list:
            for item in sublist:
                unique_fields.append(item)

        # unique_fields = list(set(unique_fields))
        unique_fields = list(set([item for sublist in field_list for item in sublist]))
        pickle.dump(unique_fields, open("{}_fields.pkl".format(nama_tabel), "wb"))

        data = dataframe.loc[:, unique_fields + ['_result']]

        for col in data.columns:
            if data[col].dtype == 'datetime64[ns, UTC]':
                data[col] = data[col].apply(lambda x: x.timestamp())
            elif data[col].dtype == 'object':
                data[col] = data[col].apply(lambda x: sum(ord(c) for c in x))

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        date_now = timezone.now() + timezone.timedelta(hours=7)
        date_now = date_now.strftime("%Y%m%d_%H%M%S")

        new_scaler = f'{nama_tabel}_scaler_{date_now}.pkl'
        try:
            os.rename("{}_scaler.pkl".format(nama_tabel), new_scaler)
            pickle.dump(scaler, open("{}_scaler.pkl".format(nama_tabel), "wb"))
        except FileNotFoundError:
            pickle.dump(scaler, open("{}_scaler.pkl".format(nama_tabel), "wb"))

        try:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 0)
            sm = SMOTE(random_state = 2)
            X_train, y_train = sm.fit_resample(X_train, y_train.ravel())
        except ValueError :
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 0)

        rfc = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
        rfc_pred = rfc.predict(X_test)

        new_model = f'{nama_tabel}_model_{date_now}.pkl'
        try:
            os.rename("{}_model.pkl".format(nama_tabel), new_model)
            pickle.dump(rfc, open("{}_model.pkl".format(nama_tabel), "wb"))
        except FileNotFoundError:
            pickle.dump(rfc, open("{}_model.pkl".format(nama_tabel), "wb"))
        response = {
            'status': 'Model Baru Tersimpan'
        }
        return JsonResponse(response, safe=False)

def loginFraudDynamic(data):
    model = sett.login_model
    scaler = sett.login_scaler

    unique_fields = pickle.load(open("digi_login_fields.pkl", "rb"))
    names = pickle.load(open("digi_login_rulename.pkl", "rb"))

    for i, row in data.iterrows():
        data_pred = [[row[field] for field in unique_fields]]
        data_pred = pd.DataFrame(data_pred, columns = unique_fields)
        try:
            data_pred['activity_date'] = data_pred['activity_date'].apply(lambda x: x.timestamp())
        except KeyError:
            pass
        try:
            data_pred['longitude'] = data_pred['longitude'].astype(float)
        except KeyError:
            pass
        try:
            data_pred['latitude'] = data_pred['latitude'].astype(float)
        except KeyError:
            pass

        for col in data_pred.columns:
            if data_pred[col].dtype == 'object':
                data_pred[col] = data_pred[col].apply(lambda x: sum(ord(c) for c in x))

        data_pred_scaled = scaler.transform(data_pred)
        result = model.predict(data_pred_scaled)

        data.at[i,'_result'] = result

    data['_description'] = data['_result'].apply(label_decs, names=names)
    data['_isFraud'] = data['_result'].apply(label_stat)

    data['_description'] = data['_description'].replace('nan', pd.np.nan)
    data['_result'] = data['_result'].astype(int)
    data['_isFraud'] = data['_isFraud'].astype(int)

class digiloginDyamicFraud(APIView):
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
                status_txt = str(status)
                if status_txt == 'SUSPECT':
                    status = 1
                elif status_txt == 'FRAUD':
                    status = 2
                else:
                    status = 0
                data_mongo = list(digi_login.find({"activity_date": {"$gte": start_date, "$lte": end_date}, "_isFraud": status}).sort("_id", -1).limit(10))
            else:
                data_mongo = list(digi_login.find({"activity_date": {"$gte": start_date, "$lte": end_date}}).sort("_id", -1).limit(10))
        elif status is not None:
            status_txt = str(status)
            if status_txt == 'SUSPECT':
                status = 1
            elif status_txt == 'FRAUD':
                status = 2
            else:
                status = 0
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
        # activity_date = data_request['activity_date']
        activity_date = timezone.now() + timezone.timedelta(hours=7)
        activity_date = activity_date.strftime("%Y-%m-%d %H:%M:%S.%f %Z")
        device_name = data_request['device_name']
        device_type = data_request['device_type']
        device_os = data_request['device_os']
        latitude = data_request['latitude']
        longitude = data_request['longitude']
        location_desc = data_request['location_desc']
        data = [[customer_id, app_id, app_version, activity_date,  device_name,  device_type,  device_os, latitude, longitude, location_desc]]
        data = pd.DataFrame(data, columns =['customer_id', 'app_id', 'app_version', 'activity_date', 'device_name', 'device_type', 'device_os', 'latitude', 'longitude', 'location_desc'])
        data['activity_date'] = pd.to_datetime(data['activity_date'], utc=True)
        data = data.assign(_result=0, _description={}, _isFraud=0)

        loginFraudDynamic(data)
        
        data_dict = data.to_dict("records")
        digi_login.insert_many(data_dict)

        response = {
            'status': 'Data Baru Berhasil Ditambahkan'
        }
        return JsonResponse(response, safe=False)
    
def loginFraudDetection(data):
    model = sett.login_model
    scaler = sett.login_scaler

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
                status_txt = str(status)
                if status_txt == 'SUSPECT':
                    status = 1
                elif status_txt == 'FRAUD':
                    status = 2
                else:
                    status = 0
                data_mongo = list(digi_login.find({"activity_date": {"$gte": start_date, "$lte": end_date}, "_isFraud": status}).sort("_id", -1).limit(10))
            else:
                data_mongo = list(digi_login.find({"activity_date": {"$gte": start_date, "$lte": end_date}}).sort("_id", -1).limit(10))
        elif status is not None:
            status_txt = str(status)
            if status_txt == 'SUSPECT':
                status = 1
            elif status_txt == 'FRAUD':
                status = 2
            else:
                status = 0
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
    scaler = sett.trx_scaler

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
                status_txt = str(status)
                if status_txt == 'SUSPECT':
                    status = 1
                elif status_txt == 'FRAUD':
                    status = 2
                else:
                    status = 0
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
            status_txt = str(status)
            if status_txt == 'SUSPECT':
                status = 1
            elif status_txt == 'FRAUD':
                status = 2
            else:
                status = 0
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