# Establish connection to google cloud sql
import os
import pandas as pd
from credentials import (
    DB_USER, DB_PASS ,DB_NAME ,INSTANCE_CONNECTION_NAME
)
from google.cloud.sql.connector import Connector, IPTypes
import sqlalchemy

# initialize Connector object
connector = Connector()
ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC

# function to return the database connection object
def getconn():
    conn = connector.connect(
        INSTANCE_CONNECTION_NAME,
        "pg8000",
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME,
        ip_type=ip_type,
    )
    return conn

# create connection pool with 'creator' argument to our connection object function
pool = sqlalchemy.create_engine(
    "postgresql+pg8000://",
    creator=getconn,
)


# Query data from database
with pool.connect() as db_conn:
  # query and fetch ratings table
  results = db_conn.execute(sqlalchemy.text("SELECT * FROM retail_sales ")).fetchall()
  sales_df = pd.DataFrame(results, columns=['id', 'sales_date', 'sales_amount'])

# Convert the sales_date column to a datetime type with a monthly frequency
# Remove the id column from the DataFrame
test= sales_df.drop('id', axis=1)
test['sales_date'] = pd.to_datetime(test['sales_date'])
test.set_index('sales_date', inplace=True)
test.index = pd.date_range(start=test.index.min(), end=test.index.max(), freq='MS')
test = test['2021-05-01':]
# save the latest data to a CSV file
test.to_csv('test.csv')

connector.close()