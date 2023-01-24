import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
import fraud_detection_system.settings as sett
from api_detection.models import digi_login_activity
# from django.db.models import QuerySet
from django.http import JsonResponse

class FraudDetection(APIView):
    def get(self, request):
        model = sett.model
        database = digi_login_activity.objects.all().values()
        data = pd.DataFrame.from_records(database)
        # data = QuerySet.to_dataframe(database)

        data = data.assign(result=0, keterangan='')

        for i, row in data.iterrows():
            data_pred = [[int(row.customer_id), row.activity_date, float(row.latitude), float(row.longitude)]]
            data_pred = pd.DataFrame(data_pred, columns =['customer_id', 'activity_date', 'latitude', 'longitude'])
            data_pred['activity_date'] = pd.to_datetime(data_pred['activity_date'], format='%Y-%m-%d %H:%M:%S.%f').apply(lambda x: x.timestamp())
            result = model.predict(data_pred)
            data.at[i,'result'] = result
            if (result==1):
                data.at[i,'keterangan'] = 'Aktivitas login dilakukan di lokasi yang jauh berbeda dari aktivitas login lain'

        response = data.to_json(orient='records')
        return JsonResponse(response, safe=False)


    def post(self, request):
        data = request.data
        model = sett.model

        customer_id = int(data['custumer_id'])
        activity_date = data['activity_date']
        latitude = int(data['latitude'])
        longitude = int(data['longitude'])

        data = [[customer_id, activity_date, latitude, longitude]]
        
        data = pd.DataFrame(data, columns =['customer_id', 'activity_date', 'latitude', 'longitude'])

        data['activity_date'] = pd.to_datetime(data['activity_date'], format='%Y-%m-%d %H:%M:%S.%f').apply(lambda x: x.timestamp())

        result = model.predict(data)

        response_dict = {"cust_id": customer_id,
        "Hasil": result}
        return Response(response_dict, status=200)