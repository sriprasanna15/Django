# Generated by Django 5.0 on 2023-12-14 18:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("app_lda", "0003_rename_lda_data_ldadata"),
    ]

    operations = [
        migrations.CreateModel(
            name="LdaDataInput",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("s_length", models.FloatField()),
                ("s_width", models.FloatField()),
                ("p_length", models.FloatField()),
                ("p_width", models.FloatField()),
                ("target", models.FloatField()),
                ("species", models.CharField(blank=True, max_length=255, null=True)),
            ],
        ),
        migrations.DeleteModel(name="LdaData",),
    ]