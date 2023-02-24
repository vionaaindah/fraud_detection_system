import os
import re
import ast
import json
import pytz
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

time_zone = pytz.timezone('Asia/Jakarta')

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

def is_number(num):
    return bool(re.match(r'^[-]?\d+(?:\.\d+)?$', num))

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

        cur_public.execute("SELECT name, configurations FROM master_rules WHERE table_id = '{}' AND deleted_at IS NULL".format(table))
        results = cur_public.fetchall()

        rules = []
        names = []
        for result in results:
            names.append(result[0])
            rules.append(result[1])

        cur_public.execute("SELECT name FROM master_tables WHERE id = '{}'".format(table))
        nama_tabel = cur_public.fetchone()
        nama_tabel = nama_tabel[0]

        pickle.dump(names, open("machine_learning/{}_rulename.pkl".format(nama_tabel), "wb"))

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
            try:
                dataframe.drop(labels=['id'], axis=1, inplace=True)
            except:
                pass

        def not_contains(a, b):
            return not a.__contains__(b)
        
        def function_in(a, b):
            return a in b

        def function_not_in(a, b):
            return a not in b

        op_map = {
            'LESS_THAN': 'lt',
            'LESS_EQUALS': 'le',
            'GREATER_THAN': 'gt',
            'GREATER_EQUALS': 'ge',
            'EQUALS': 'eq',
            'NOT_EQUALS': 'ne',
            'CONTAINS': 'contains'
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
                
                val = r["value"]
                try:
                    val = datetime.strptime(val, '%H:%M').time()
                except ValueError:
                    if '[' in val or is_number(val):
                        val = ast.literal_eval(val) if isinstance(val, str) else val
                    else:
                        pass
                sub_val_list.append(val)
                
                if r['operator'] == 'NOT_CONTAINS':
                    op_func = not_contains
                elif r['operator'] == 'IN':
                    op_func = function_in
                elif r['operator'] == 'NOT_IN':
                    op_func = function_not_in
                else:
                    op_func = getattr(operator, op_map[r['operator']])
                sub_op_list.append(op_func)

            field_list.append(sub_field_list)
            val_list.append(sub_val_list)
            operator_list.append(sub_op_list)
        
        unique_fields = []
        for sublist in field_list:
            for item in sublist:
                unique_fields.append(item)

        unique_fields = list(set([item for sublist in field_list for item in sublist]))
        pickle.dump(unique_fields, open("machine_learning/{}_fields.pkl".format(nama_tabel), "wb"))

        for col in unique_fields:
            if col == 'USER_REF' or col == 'NARRATIVE1':
                dataframe[col] = dataframe[col].fillna(' ')
                dataframe = dataframe[~dataframe[col].str.isdigit()]
            else:
                dataframe = dataframe.dropna(subset=[col])

        if nama_tabel == 'digi_login':
            dataframe['activity_date'] = pd.to_datetime(dataframe['activity_date'], utc=True)
        elif nama_tabel == 'main_trx':
            dataframe['BUSS_DATE'] = pd.to_datetime(dataframe['BUSS_DATE'], utc=True)
            dataframe['TIME_STAMP'] = pd.to_datetime(dataframe['TIME_STAMP'], format="%Y-%m-%d-%H.%M.%S.%f", utc=True)

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
                        if '[' in field or is_number(field):
                            field = ast.literal_eval(field) if isinstance(field, str) else field
                        else:
                            pass
                    if not op_func(field, val):
                        label_i = False
                        break

                if label_i:
                    if label != 0:
                        if label < 100:
                            l_old = str(label).zfill(2)
                            l_new = str(i+1).zfill(2)
                            label = int('100' + l_old + l_new)
                        else:
                            l_new = str(i+1).zfill(2)
                            label = int(str(label) + l_new)
                    else:
                        label = i+1
            return label
        
        dataframe['_result'] = dataframe.apply(labeling, axis=1)
        dataframe['_description'] = dataframe['_result'].apply(label_decs, names=names)
        dataframe['_isFraud'] = dataframe['_result'].apply(label_stat)

        data_dict = dataframe.to_dict("records")
        collection.insert_many(data_dict)

        data = dataframe.loc[:, unique_fields + ['_result']]

        for col in data.columns:
            if data[col].dtype == 'datetime64[ns, UTC]':
                data[col] = data[col].apply(lambda x: x.timestamp())
            elif data[col].dtype == 'object':
                if all(is_number(x) for x in data[col]):
                    data[col] = data[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                else:
                    data[col] = data[col].apply(lambda x: sum(ord(c) for c in x))

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        date_now = timezone.now() + timezone.timedelta(hours=7)
        date_now = date_now.strftime("%Y%m%d_%H%M%S")

        new_scaler = f'machine_learning/history/{nama_tabel}_scaler_{date_now}.pkl'
        try:
            os.rename("machine_learning/{}_scaler.pkl".format(nama_tabel), new_scaler)
            pickle.dump(scaler, open("machine_learning/{}_scaler.pkl".format(nama_tabel), "wb"))
        except FileNotFoundError:
            pickle.dump(scaler, open("machine_learning/{}_scaler.pkl".format(nama_tabel), "wb"))

        try:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 0)
            sm = SMOTE(random_state = 2)
            X_train, y_train = sm.fit_resample(X_train, y_train.ravel())
        except ValueError :
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 0)

        rfc = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
        rfc_pred = rfc.predict(X_test)

        new_model = f'machine_learning/history/{nama_tabel}_model_{date_now}.pkl'
        try:
            os.rename("machine_learning/{}_model.pkl".format(nama_tabel), new_model)
            pickle.dump(rfc, open("machine_learning/{}_model.pkl".format(nama_tabel), "wb"))
        except FileNotFoundError:
            pickle.dump(rfc, open("machine_learning/{}_model.pkl".format(nama_tabel), "wb"))
        response = {
            'status': 'Model Baru Tersimpan'
        }
        return JsonResponse(response, safe=False)

def loginFraudDynamic(data):
    model = sett.digi_login_model
    scaler = sett.digi_login_scaler

    unique_fields = pickle.load(open("machine_learning/digi_login_fields.pkl", "rb"))
    names = pickle.load(open("machine_learning/digi_login_rulename.pkl", "rb"))

    for i, row in data.iterrows():
        data_pred = [[row[field] for field in unique_fields]]
        data_pred = pd.DataFrame(data_pred, columns = unique_fields)

        for col in data_pred.columns:
            if data_pred[col].dtype == 'datetime64[ns, UTC]':
                data_pred[col] = data_pred[col].apply(lambda x: x.timestamp())
            elif data_pred[col].dtype == 'object':
                if all(is_number(x) for x in data[col]):
                    data_pred[col] = data_pred[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                else:
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
        limit = request.GET.get('limit')
        skip = request.GET.get('skip')

        filter_mongo = {}

        if customer_id is not None:
            customer_id = str(customer_id)
            filter_mongo["customer_id"] = customer_id
            
        if status is not None:
            status_txt = str(status)
            if status_txt == 'SUSPECT':
                status = 1
            elif status_txt == 'FRAUD':
                status = 2
            else:
                status = 0
            filter_mongo["_isFraud"] = status

        if start_date is not None:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            filter_mongo["activity_date"] = {"$gte": start_date}

        if end_date is not None:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            filter_mongo["activity_date"] = {"$lte": end_date}

        limit = int(limit) if limit is not None else 10
        skip = int(skip) if skip is not None else 0

        data_mongo = list(digi_login.find(filter_mongo).limit(limit).skip(skip))

        database = pd.DataFrame(data_mongo)

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
    
def mainTRXFraudDynamic(data):
    model = sett.main_trx_model
    scaler = sett.main_trx_scaler

    unique_fields = pickle.load(open("machine_learning/main_trx_fields.pkl", "rb"))
    names = pickle.load(open("machine_learning/main_trx_rulename.pkl", "rb"))

    for i, row in data.iterrows():
        data_pred = [[row[field] for field in unique_fields]]
        data_pred = pd.DataFrame(data_pred, columns = unique_fields)

        print(data_pred['AMOUNT'])
        print(type(data_pred['AMOUNT'][0]))

        for col in data_pred.columns:
            if data_pred[col].dtype == 'datetime64[ns, UTC]':
                data_pred[col] = data_pred[col].apply(lambda x: x.timestamp())
            elif data_pred[col].dtype == 'object':
                if all(is_number(x) for x in data_pred[col]):
                    data_pred[col] = data_pred[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                else:
                    data_pred[col] = data_pred[col].apply(lambda x: sum(ord(c) for c in x))


        data_pred_scaled = scaler.transform(data_pred)
        result = model.predict(data_pred_scaled)

        data.at[i,'_result'] = result

    data['_description'] = data['_result'].apply(label_decs, names=names)
    data['_isFraud'] = data['_result'].apply(label_stat)

    data['_description'] = data['_description'].replace('nan', pd.np.nan)
    data['_result'] = data['_result'].astype(int)
    data['_isFraud'] = data['_isFraud'].astype(int)

class mainTRXDyamicFraud(APIView):
    def get(self, request):
        account = request.GET.get('account')
        cif = request.GET.get('cif')
        status = request.GET.get('status')
        start_date = request.GET.get('start_date')
        end_date = request.GET.get('end_date')
        channel = request.GET.get('channel')
        limit = request.GET.get('limit')
        skip = request.GET.get('skip')

        filter_mongo = {}

        if cif is not None:
            cif = str(cif)
            filter_mongo["CIF"] = cif

        if account is not None:
            account = str(account)
            filter_mongo["ACCOUNT"] = account

        if channel is not None:
            channel = str(channel)
            filter_mongo["USER_REF"] = {"$regex": "^" + channel}

        if status is not None:
            status_txt = str(status)
            if status_txt == 'SUSPECT':
                status = 1
            elif status_txt == 'FRAUD':
                status = 2
            else:
                status = 0
            filter_mongo["_isFraud"] = status

        if start_date is not None:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            filter_mongo["BUSS_DATE"] = {"$gte": start_date}

        if end_date is not None:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            filter_mongo["BUSS_DATE"] = {"$lte": end_date}

        limit = int(limit) if limit is not None else 10
        skip = int(skip) if skip is not None else 0

        data_mongo = list(main_trx.find(filter_mongo).limit(limit).skip(skip))

        database = pd.DataFrame(data_mongo)

        try:
            database['_description'] = database['_description'].replace('', pd.np.nan)
            database['_description'] = database['_description'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        except KeyError:
            database['_description'] = pd.np.nan

        database['BUSS_DATE'] = database['BUSS_DATE'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        database['TIME_STAMP'] = database['TIME_STAMP'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        database['_result'] = database['_result'].astype(int)
        database['_isFraud'] = database['_isFraud'].astype(int)

        data_result = database.to_json(orient='records')
        response = json.loads(data_result)
        return JsonResponse(response, safe=False)
    
    def post(self, request):
        data_request = request.data

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

        data = [[BUSS_DATE, BRANCH, CIF,  SUFIX, ACCOUNT, TRX_CODE, USER_REF, SIGN, AMOUNT, NARRATIVE1, GROUP_OR_USER_ID,  TIME_STAMP, SEQ, BRANCH_INPUT, CUSTOMER_TYPE, JKartu]]
        data = pd.DataFrame(data, columns =["BUSS_DATE", "BRANCH", "CIF", "SUFIX", "ACCOUNT", "TRX_CODE", "USER_REF", "SIGN", "AMOUNT", "NARRATIVE1", "GROUP_OR_USER_ID", "TIME_STAMP", "SEQ", "BRANCH_INPUT", "CUSTOMER_TYPE", "JKartu"])
        data['BUSS_DATE'] = pd.to_datetime(data['BUSS_DATE'], utc=True)
        data['TIME_STAMP'] = pd.to_datetime(data['TIME_STAMP'], utc=True)
        data = data.assign(_result=0, _description={}, _isFraud=0)

        mainTRXFraudDynamic(data)
        
        data_dict = data.to_dict("records")
        main_trx.insert_many(data_dict)

        response = {
            'status': 'Data Baru Berhasil Ditambahkan'
        }
        return JsonResponse(response, safe=False)