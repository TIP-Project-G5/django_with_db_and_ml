# Generated by Django 4.0.2 on 2024-05-04 08:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('floodshield', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='depot',
            name='dp_phone_no',
            field=models.TextField(null=True),
        ),
    ]