﻿# django-tally
2019-01-06 This is a Django app, locally running on Windows 10, deployed on AWS Elastic Beanstalk.     
2019-01-09 [AWS EB deployment logs](https://github.com/Nov05/yelp-dataset-challenge/blob/master/aws/2020-01-09%20aws%20eb.md)  


### Testing URLs 
[Testing data document](https://drive.google.com/open?id=1EKPZh1e88_jnXafk7OJ_euSB54ilmFdJ)    
[Testing script on Colab notebook](https://colab.research.google.com/drive/1Gzo2lFj1cEj72mkVoko5rAo-OXaeLwot)     
http://127.0.0.1:8000/admin   
Below links are for demonstration.    
https://www.yelp.com/biz/aunt-jakes-new-york   
http://127.0.0.1:8000/yelp/index    
http://127.0.0.1:8000/yelp/aunt-jakes-new-york (by business alias)      
http://127.0.0.1:8000/yelp/I2lgw_7DUnwD92ND4PN-Ow?viztype=0 (by business ID)   
http://127.0.0.1:8000/yelp/DR22QPe3A52diajwPuooVA?viztype=0    
http://127.0.0.1:8000/yelp/Iq7NqQD-sESu3vr9iEGuTA?viztype=1    
Below links are examples.     
http://127.0.0.1:8000/yelp/review/ (create review)      
http://127.0.0.1:8000/yelp/review/9759c0c0-b28a-44ff-b770-4cf303367a60 (get, put, delete, by review UUID)           
http://127.0.0.1:8000/bucketlists (create)    
http://127.0.0.1:8000/bucketlists/1 (get, put, delete)   


### Frequently used commands
```
$ python manage.py runserver
$ python manage.py makemigrations  
$ python manage.py migrate  
$ python manage.py test
$ python manage.py inspectdb > models.py
$ pip freeze | Out-File -Encoding UTF8 requirements.txt
$ python -m django --version
```  


### Deploy to AWS Elastic Beanstalk   
During the deployment, you may need to use the following AWS CLI commands.
```  
eb init -p python-3.6 django-tally
eb create django-tally
eb status
eb deploy
eb open
eb logs
eb config
eb terminate django-tally
```

### Activate virtual enviroment  
(base) PS D:\github\django-tally>     
```
$ pipenv shell
```
Install dependencies: 
```
$ pipenv install django psycopg2-binary djangorestframework spacy lxml scattertext pytextrank awscli pylint pyyaml https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz
```

### Create project 
If you have downloaded this repo, skip this step.      
PS D:\github\django-tally>     
```
$ cd C:\Users\guido\.virtualenvs\django-tally-QTYVOJb0\Scripts\
$ python django-admin.py startproject tally D:\github\django-tally
```
project name: tally  
project created in directory: D:\github\django-tally   


### Run Django app    
PS D:\github\django-tally>     
```
$ python manage.py runserver
```   
Logs:     
```
Watching for file changes with StatReloader
Performing system checks...

System check identified no issues (0 silenced).

You have 17 unapplied migration(s). Your project may not work properly until you apply the migrations for app(s): admin, auth, contenttypes, sessions.
Run 'python manage.py migrate' to apply them.
January 07, 2020 - 01:05:29
Django version 3.0.2, using settings 'tally.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.
[07/Jan/2020 01:05:55] "GET / HTTP/1.1" 200 16351
[07/Jan/2020 01:05:55] "GET /static/admin/css/fonts.css HTTP/1.1" 200 423
[07/Jan/2020 01:05:55] "GET /static/admin/fonts/Roboto-Light-webfont.woff HTTP/1.1" 200 85692
[07/Jan/2020 01:05:55] "GET /static/admin/fonts/Roboto-Bold-webfont.woff HTTP/1.1" 200 86184
[07/Jan/2020 01:05:55] "GET /static/admin/fonts/Roboto-Regular-webfont.woff HTTP/1.1" 200 85876
```

### Configurate settings.py  
```
# Internationalization
# https://docs.djangoproject.com/en/3.0/topics/i18n/
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'US/Central' # 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True
```
```
# Database
# https://docs.djangoproject.com/en/3.0/ref/settings/#databases
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'postgres',
        'USER': '*',
        'PASSWORD': '*',
        'HOST': '*.*.us-east-2.rds.amazonaws.com',
        'PORT': '5432',
        'OPTIONS': {
                        'options': '-c search_path=django'
                    },        
        'TEST': {
            'ENGINE': 'django.db.backends.sqlite3',
        },
    },
}
```
Migrate Django admin tables to database `django` schema.   
[Grant permissions](https://github.com/Nov05/yelp-dataset-challenge/blob/master/tallysql/grant_permissions.sql) to the user in database accordingly.   

### Migration   
```
$ cd d:/github/django-tally
```
PS D:\github\django-tally> 
```
$ python manage.py migrate
```
Logs:    
```
Operations to perform:
  Apply all migrations: admin, auth, contenttypes, sessions
Running migrations:
  Applying contenttypes.0001_initial... OK
  Applying auth.0001_initial... OK
  Applying admin.0001_initial... OK
  Applying admin.0002_logentry_remove_auto_add... OK
  Applying admin.0003_logentry_add_action_flag_choices... OK
  Applying contenttypes.0002_remove_content_type_name... OK
  Applying auth.0002_alter_permission_name_max_length... OK
  Applying auth.0003_alter_user_email_max_length... OK
  Applying auth.0004_alter_user_username_opts... OK
  Applying auth.0005_alter_user_last_login_null... OK
  Applying auth.0006_require_contenttypes_0002... OK
  Applying auth.0007_alter_validators_add_error_messages... OK
  Applying auth.0008_alter_user_username_max_length... OK
  Applying auth.0009_alter_user_last_name_max_length... OK
  Applying auth.0010_alter_group_name_max_length... OK
  Applying auth.0011_update_proxy_permissions... OK
  Applying sessions.0001_initial... OK
```

### Create admin user  
PS D:\github\django-tally> 
```
$ python manage.py createsuperuser
```
```
Username (leave blank to use 'guido'): ***
Email address: admin@example.com
Password:
Password (again):
This password is too short. It must contain at least 8 characters.
This password is too common.
This password is entirely numeric.
Bypass password validation and create user anyway? [y/N]: n
Password:
Password (again):
Superuser created successfully.
```
a**** / T****_******  

### Using Django REST Framework for APIs
PS D:\github\django-tally>    
```
# D:\github\django-tally\tally\settings.py
...
# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',             # Add this line; other app names are not allowed
    'apis',                       # Add this line; you can use app names other than "apis" 
    'yelp',                       # Add this app as well
]
```
```
$ python manage.py startapp apis
```  
E.g. regular expression match UUID as primary key `(?P<pk>[0-9a-f-]+)`:  
```
urlpatterns = {
    url(r'^yelp/$', 
        YelpYelpScrapingCreateView.as_view(), name="create"),
    url(r'^yelp/(?P<pk>[0-9a-f-]+)/$',
        YelpYelpScrapingDetailsView.as_view(), name="details"),
}
```
E.g. query strings
```
urlpatterns = {path('<slug:business_id>', home, name='home')}
def home(request, business_id):
    viztype = request.GET.get('viztype')
    if viztype == '1':
        result = json.dumps(yelpTrendyPhrases(business_id))
    elif viztype == '2':
        result = json.dumps(yelpReviewCountMonthly(business_id))
    else:
        result = json.dumps(getDataViztype0(business_id))
    return HttpResponse(result)
```
Follow this [tutorial](https://scotch.io/tutorials/build-a-rest-api-with-django-a-test-driven-approach-part-1).    


### Auto-generate data models from database tables
```
$ python manage.py inspectdb > models.py
```
After running this command, modify class names in the `models.py` file.     
Add <AppName> to every class name. E.g.   
For app "apis", change `class Bucketlist` -> `class ApisBucketlist`   
For app "yelp", change `class Business` -> `class YelpBusiness`    
Follow the instructions in the `models.py` file, make sure model definitions are correct.   
Then move the `models.py` file to the corresponding app folder.    
So every app would have their own models without conflicting with other apps.   


### Debug
Issue: [Django “ValueError: source code string cannot contain null bytes”](https://stackoverflow.com/questions/52273840/django-valueerror-source-code-string-cannot-contain-null-bytes)  
Solution: You can simply create a new .py file, copy and paste the `models.py` content to it, then replace the `models.py` file with it.         


### spaCy
spaCy models   
https://spacy.io/usage/models  
How to install models     
https://pypi.org/project/spacy/  
Download spaCy model manually (Not in use)       
https://github.com/explosion/spacy-models/releases  
You can install spaCy models just like installing a Python package.   
`pipenv install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz`       
Then import the models in your code.    
```
import en_core_web_sm
nlp = en_core_web_sm.load()   
```
or
```
import spacy
nlp = spacy.load("en_core_web_sm") 
```
【Manually】
Put the following folder in the repo (same level with `manage.py`).       
**spacy.load("en_core_web_sm/en_core_web_sm-2.2.5")** with `__init__.py`  
CAUTION: You can do it this way, but deployment from Windows 10 to AWS Elastica Beanstalk might have UnicodeDecodingError when loading a model, while Windows 10 locally or deployment from MacOS seem fine.  


### Reference  
[Django Documentation](https://docs.djangoproject.com/en/3.0/)   
[Django Message Framework](https://docs.djangoproject.com/en/3.0/ref/contrib/messages/)    
[AWS: Deploying a Django Application to Elastic Beanstalk](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-django.html#w510aac13c37c15c13b7b2b3b3)    
[AWS: Configure the EB CLI](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-configuration.html)       
[AWS: My security credentials](https://console.aws.amazon.com/iam/home?#/security_credentials)   
[Build a REST API with Django – A Test Driven Approach: Part 1](https://scotch.io/tutorials/build-a-rest-api-with-django-a-test-driven-approach-part-1)  
[List of Useful URL Patterns](https://simpleisbetterthancomplex.com/references/2016/10/10/url-patterns.html)   
[Understanding file encoding in VSCode and PowerShell](https://docs.microsoft.com/en-us/powershell/scripting/components/vscode/understanding-file-encoding?view=powershell-7) 02/27/2019  
【Repo】[flask-yelp-reviews](https://github.com/Nov05/flask-yelp-reviews)    
【Repo】[Lily's Django](https://github.com/Lambda-School-Labs/tally-ai-ds/tree/b95c67d7f0989b49a5ab8b89d9e6884233622da3/ElasticBeanstalkDjango_v.0.2/ebdjango)         