# tallylib/sql.py
from django.db import connection
# Local
from yelp.models import YelpReview
'''
2020-01-10 Database table "tallyds.review" has the following index created.
    "CREATE INDEX idx_review ON tallyds.review (business_id, datetime DESC);"
'''


# Query with Django data models
def getReviews(business_id, 
               starting_date, 
               ending_date):
    sql = f'''
    SELECT uuid, date, text FROM tallyds.review
    WHERE business_id = '{business_id}'
    AND datetime >= '{starting_date}'
    AND datetime <= '{ending_date}';
    '''
    return [[record.date, record.text] for record in YelpReview.objects.raw(sql)]


# Query without Django data models
def getReviewCountMonthly(business_id,
                          number_of_months=12):
    sql = f'''
    SELECT extract(year from date)::INTEGER AS year,
           extract(month from date)::INTEGER AS month,
           count(*) AS count
    FROM tallyds.review AS r
    WHERE business_id = '{business_id}'
    GROUP BY 1, 2
    ORDER BY 1 DESC, 2 DESC
    LIMIT {number_of_months};
    '''
    with connection.cursor() as cursor:
        cursor.execute(sql)
        # return a tuple
        return cursor.fetchall()

    
def getLatestReviewDate(business_id):
    sql = f'''
    SELECT datetime
    FROM tallyds.review
    WHERE business_id = '{business_id}'
    ORDER BY datetime DESC
    LIMIT 1;
    '''
    with connection.cursor() as cursor:
        cursor.execute(sql)
        # return a datetime.datatime object
        return cursor.fetchone()[0]