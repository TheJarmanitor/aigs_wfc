from django.db import models

from django.utils import timezone
import datetime

class Layout(models.Model):
    data = models.TextField()
    pub_date = models.DateTimeField("date published")
    img_path = models.CharField(max_length=400, default="")

    def __str__(self):
        return "Layout" + str(self.id)
    
class CPPNState(models.Model):
    data = models.TextField()
    pub_date = models.DateTimeField("date published")

    def __str__(self):
        return "CPPNState" + str(self.id)