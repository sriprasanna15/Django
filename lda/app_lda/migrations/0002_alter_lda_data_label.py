# Generated by Django 5.0 on 2023-12-05 17:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("app_lda", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="lda_data",
            name="label",
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
