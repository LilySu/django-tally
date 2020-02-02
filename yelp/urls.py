# yelp/urls.py
from django.urls import path
from django.conf.urls import url
from rest_framework.urlpatterns import format_suffix_patterns
# view functions
from .views import hello
<<<<<<< HEAD
from .views import home
=======
from .views import YelpReviewCreateView # data maintenance
from .views import YelpReviewDetailsView # data maintenance
>>>>>>> 64b43fc29e334cc1394c9ccd6af8ba0312fe7064


urlpatterns = {
    path('', hello, name='hello'),
<<<<<<< HEAD
    path('<slug:business_id>', home, name='home'),
=======
    # example
    # create/get/put/delete yelp scraping data via APIs
    # URL - /yelp/yelpscraping/ (create)
    # URL - /yelp/yelpscraping/9759c0c0-b28a-44ff-b770-4cf303367a60/
    url(r'^yelpscraping/$', YelpReviewCreateView.as_view(), name="create"),
    url(r'^yelpscraping/(?P<pk>[0-9a-f-]+)/$',
        YelpReviewDetailsView.as_view(), name="details"),
>>>>>>> 64b43fc29e334cc1394c9ccd6af8ba0312fe7064
}

urlpatterns = format_suffix_patterns(urlpatterns)