from django.db import models

# Create your models here.


class LdaInput(models.Model):
    s_length = models.FloatField()
    s_width = models.FloatField()
    p_length = models.FloatField()
    p_width = models.FloatField()
    target = models.FloatField()
    species = models.CharField(max_length=255, null=True, blank=True)
