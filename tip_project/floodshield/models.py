from django.db import models

class Area(models.Model):
    area_id = models.AutoField(primary_key=True)
    location = models.TextField()
    office_address = models.TextField()
    phone_no = models.TextField()
    latitude = models.FloatField()
    longitude = models.FloatField()

    class Meta:
        db_table = 'AREA'

class Depot(models.Model):
    depot_no = models.AutoField(primary_key=True)
    dp_area_id = models.ForeignKey('Area', on_delete=models.CASCADE, db_column='dp_area_id')
    dp_office_address = models.TextField()
    dp_phone_no = models.TextField(null=True)
    dp_mobile_no = models.TextField(null=True)

    class Meta:
        db_table = 'DEPOT'

class Groundwater(models.Model):
    gw_timestamp = models.DateField()
    gw_depot_no = models.ForeignKey('Depot', on_delete=models.CASCADE, db_column='gw_depot_no')
    groundwater_value = models.FloatField()
    groundwater_id = models.AutoField(primary_key=True)

    class Meta:
        db_table = 'GROUNDWATER'
        
class Rainfall(models.Model):
    rainfall_id = models.AutoField(primary_key=True)
    timestamp = models.DateField()
    rf_area_id = models.ForeignKey('Area', on_delete=models.CASCADE, db_column='rf_area_id')
    rainfall_value = models.DecimalField(max_digits=10, decimal_places=2)

    class Meta:
        db_table = 'RAINFALL'

class Prediction(models.Model):
    prediction_id = models.AutoField(primary_key=True)
    p_area_id = models.ForeignKey('Area', on_delete=models.CASCADE, db_column='area_id')
    groundwater_forecast1 = models.FloatField()
    groundwater_forecast2 = models.FloatField()
    groundwater_forecast3 = models.FloatField()
    groundwater_forecast4 = models.FloatField()
    groundwater_forecast5 = models.FloatField()
    groundwater_forecast6 = models.FloatField()
    groundwater_forecast7 = models.FloatField()
    rainfall_forecast1 = models.FloatField()
    rainfall_forecast2 = models.FloatField()
    rainfall_forecast3 = models.FloatField()
    rainfall_forecast4 = models.FloatField()
    rainfall_forecast5 = models.FloatField()
    rainfall_forecast6 = models.FloatField()
    rainfall_forecast7 = models.FloatField()
    min_temp_forecast1 = models.FloatField()
    min_temp_forecast2 = models.FloatField()
    min_temp_forecast3 = models.FloatField()
    min_temp_forecast4 = models.FloatField()
    min_temp_forecast5 = models.FloatField()
    min_temp_forecast6 = models.FloatField()
    min_temp_forecast7 = models.FloatField()
    max_temp_forecast1 = models.FloatField()
    max_temp_forecast2 = models.FloatField()
    max_temp_forecast3 = models.FloatField()
    max_temp_forecast4 = models.FloatField()
    max_temp_forecast5 = models.FloatField()
    max_temp_forecast6 = models.FloatField()
    max_temp_forecast7 = models.FloatField()
    flood_forecast1 = models.FloatField()
    flood_forecast2 = models.FloatField()
    flood_forecast3 = models.FloatField()
    flood_forecast4 = models.FloatField()
    flood_forecast5 = models.FloatField()
    flood_forecast6 = models.FloatField()
    flood_forecast7 = models.FloatField()

    class Meta:
        db_table = 'PREDICTION'
