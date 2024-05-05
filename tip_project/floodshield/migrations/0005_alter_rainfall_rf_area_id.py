# Generated by Django 4.0.2 on 2024-05-04 08:39

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('floodshield', '0004_alter_depot_dp_phone_no'),
    ]

    operations = [
        migrations.AlterField(
            model_name='rainfall',
            name='rf_area_id',
            field=models.ForeignKey(db_column='rf_area_id', on_delete=django.db.models.deletion.CASCADE, to='floodshield.area'),
        ),
    ]