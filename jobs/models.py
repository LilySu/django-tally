from django.db import models
import uuid


# Create your models here.

class JobConfig(models.Model):
    job_id = models.CharField(max_length=100, blank=False, null=False, primary_key=True)
    job_desc = models.CharField(max_length=200, blank=True, null=True)
    job_rate = models.FloatField(blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        """Return a human readable representation of the model instance."""
        return f"【Job ID】{self.job_id},【Description】{self.job_desc},【Job Rate】every {self.job_rate} day(s)"

    class Meta:
        managed = False
        db_table = 'job_config'


class JobLog(models.Model):
    uuid = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    business_id = models.CharField(max_length=100, blank=True, null=True)
    job_type = models.SmallIntegerField(blank=True, null=True)
    job_status = models.SmallIntegerField(blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    job_message = models.CharField(max_length=500, blank=True, null=True)
    
    def __str__(self):
        """Return a human readable representation of the model instance."""
        return f"【Business ID】{self.business_id},\
【Job Type】{self.job_type},\
【Status】{self.job_status}, \
{self.timestamp}, {self.job_message[:100]}"

    class Meta:
        managed = False
        db_table = 'job_log'
