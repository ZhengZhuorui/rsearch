from django.db import models

# Create your models here.
class Dataset(models.Model):
    name = models.CharField(max_length=255)
    database_path = models.CharField(max_length=255)
    path1 = models.CharField(max_length=255)
    path2 = models.CharField(max_length=255)

# 不使用，仅仅表示数据集格式
class RemoteSensing(models.Model):
    time = models.CharField(max_length=255)
    lat = models.CharField(max_length=12)
    lng = models.CharField(max_length=12)
    feature = models.TextField()
    filename = models.TextField()