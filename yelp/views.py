import requests
import json
import time
from django.shortcuts import render
from django.http import HttpResponse
# Local imports
from tallylib.textrank import yelpTrendyPhrases
from tallylib.scattertxt import getDataViztype0
from tallylib.statistics import yelpReviewCountMonthly
<<<<<<< HEAD
from tallylib.sentiment import yelpReviewSentiment
from tallylib.sql import deleteVizdata
from tallylib.sql import getLatestVizdata
from tallylib.sql import updateVizdata
from tallylib.sql import insertVizdataLog
from tallylib.sql import isTallyBusiness
from tallylib.sql import insertTallyBusiness
from tallylib.locks import lock_yelpscraper
from tasks.tasks import task_yelpScraper


# Create your views here.

# Nothing here but say hello
def hello(request):
    result = "Hello, you are at the Tally Yelp Analytics home page."
    return HttpResponse(result)
=======
from .models import YelpReview                # for data maintenance
from .serializers import YelpReviewSerializer # for data maintenance
>>>>>>> 64b43fc29e334cc1394c9ccd6af8ba0312fe7064


# Query strings -> Main analytics
def home(request, business_id):
<<<<<<< HEAD
    '''get data for views (APIs)'''
    returncode, result = 0, ""
    try: 
        # check whether the business ID has never been scraped before
        # check whether some other session(s) is scraping the same business ID
        if not isTallyBusiness(business_id) and not lock_yelpscraper.isLocked(business_id):
            deleteVizdata(business_id)
            task_yelpScraper([business_id], job_type=1) # triggered by end user
            insertTallyBusiness([business_id])

        for i in range(1200):
            if lock_yelpscraper.isLocked(business_id):
                time.sleep(1)
                if i % 30 == 0:
                    print("Waiting for some other session(s) finishing web scraping...")
            else:
                break

        viztype = request.GET.get('viztype')
        viztype = int(viztype)
        data = getLatestVizdata(business_id, viztype=viztype) # a list of tuples
        if len(data) > 0:
                result = data[0][0]
                returncode = 0 # success
        else:
            if viztype == 0: # viztype0 and viztype3
                result = json.dumps(getDataViztype0(business_id),
                                    sort_keys=False)
                returncode = 0 # success
            elif viztype == 1:    
                result = json.dumps(yelpTrendyPhrases(business_id), 
                                    sort_keys=False)
                returncode = 0 # success
            elif viztype == 2:     
                result = json.dumps(yelpReviewCountMonthly(business_id), 
                                    sort_keys=False)
                returncode = 0 # success
            elif viztype == 4:
                result = json.dumps(yelpReviewSentiment(business_id), 
                                    sort_keys=False)
                returncode = 0 # success
            else: 
                print(f"Error: There is no viztype {str(viztype)}.")
                returncode = 1 # error

            # update table ds_vizdata and ds_vizdata_log
            if returncode == 0:
                updateVizdata(business_id, viztype, result)
                insertVizdataLog(business_id, viztype, triggeredby=1) # triggered by end user
    except Exception as e:
        print(e)
        returncode = 1 # error

    return HttpResponse(result)

=======
    viztype = request.GET.get('viztype')
    if viztype == '1':
        result = json.dumps(yelpTrendyPhrases(business_id), 
                            sort_keys=False)
    elif viztype == '2':
        result = json.dumps(yelpReviewCountMonthly(business_id), 
                            sort_keys=False)
    else: # viztype0 and viztype3
        result = json.dumps(getDataViztype0(business_id),
                            sort_keys=False)
    return HttpResponse(result)


# Nothing here
def hello(request):
    result = "Hello, you are at the Tally Yelp Analytics home page."
    return HttpResponse(result)


# example
class YelpReviewCreateView(generics.ListCreateAPIView):
    """This class defines the create behavior of our rest api."""
    queryset = YelpReview.objects.all()
    serializer_class = YelpReviewSerializer

    def perform_create(self, serializer):
        """Save the post data when creating a new bucketlist."""
        serializer.save()


# example
class YelpReviewDetailsView(generics.RetrieveUpdateDestroyAPIView):
    """This class handles the http GET, PUT and DELETE requests."""
    queryset = YelpReview.objects.all()
    serializer_class = YelpReviewSerializer

>>>>>>> 64b43fc29e334cc1394c9ccd6af8ba0312fe7064
