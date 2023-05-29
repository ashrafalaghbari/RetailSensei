import pandas as pd
import os
from fredapi import Fred
from credentials import FRED_API_KEY
import sqlalchemy
from google.cloud.sql.connector import Connector, IPTypes

# # get FRED API key
FRED_API_KEY = os.environ['FRED_API_KEY']

# get data from FRED
fred = Fred(api_key = FRED_API_KEY)
series = fred.search('RSXFSN')

# get data from FRED
retail_sales = fred.get_series('RSXFSN')
# Convert the data to a Pandas DataFrame
df = pd.DataFrame(retail_sales, columns=['sales_amount'])

# add freq MS to index date
df.index.freq = 'MS'

# # Obtain consumer price index data from FRED
# cpi = fred.get_series('CPIAUCSL', observation_start='1992-01-01')

# # Convert the retail sales to real terms
# base_value = cpi[-1]
# adj_retail_sales = df['sales_amount'] * (base_value / cpi)
# # # convert the retail sales to billions of dollars
# adj_retail_sales = adj_retail_sales / 1000
# # # Convert the data to a Pandas DataFrame
# adj_retail_sales = pd.DataFrame(adj_retail_sales, columns=['sales_amount'])
# # add freq MS to index date
# adj_retail_sales.index.freq = 'MS'

# Convert the index and sales values of the DataFrame into a list of tuples
sales_tuples = [(index.date(), float(sales)) for index, sales in df.itertuples()]

# add the last observation date to database
# last_date = sales_tuples[-1]
import datetime
last_date= (datetime.date(2023, 5, 1), 690.857)


# Get environment variables for database connection
DB_USER = os.environ['DB_USER']
DB_PASS = os.environ['DB_PASS']
DB_NAME = os.environ['DB_NAME']
INSTANCE_CONNECTION_NAME = os.environ['INSTANCE_CONNECTION_NAME']

# Establish connection to google cloud sql
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

# # connect to connection pool
# with pool.connect() as db_conn:
#   # create ratings table in our sandwiches database
#   db_conn.execute(
#     sqlalchemy.text(
#       "CREATE TABLE IF NOT EXISTS retail_sales " 
#         "(id serial PRIMARY KEY, "
#         "sales_date DATE, sales_amount NUMERIC);"
#     )
#   )

  # commit transaction (SQLAlchemy v2.X.X is commit as you go)
#   db_conn.commit()

# insert entries into table
# connect to connection pool
# with pool.connect() as db_conn:
#     # Insert the data into the table
#      # Insert the data into the table if it doesn't already exist
#     insert_stmt = sqlalchemy.text(
#         "INSERT INTO retail_sales (sales_date, sales_amount) "
#         "SELECT :sales_date, :sales_amount "
#         "WHERE NOT EXISTS (SELECT 1 FROM retail_sales "
#         "WHERE sales_date = :sales_date)"
#     )
#     for sale in sales_tuples:
#         # insert entries into table
#         db_conn.execute(insert_stmt, parameters={"sales_date": sale[0], "sales_amount": sale[1]})

#     # # commit transactions
#     db_conn.commit()

# # drop table
# with pool.connect() as db_conn:
#   db_conn.execute(
#     sqlalchemy.text(
#       "DROP TABLE IF EXISTS retail_sales;"
#     )
#   )
#   db_conn.commit()


# insert last date
# connect to connection pool
with pool.connect() as db_conn:
    # Insert the data into the table
     # Insert the data into the table if it doesn't already exist
    insert_stmt = sqlalchemy.text(
        "INSERT INTO retail_sales (sales_date, sales_amount) "
        "SELECT :sales_date, :sales_amount "
        "WHERE NOT EXISTS (SELECT 1 FROM retail_sales "
        "WHERE sales_date = :sales_date)"
    )

    # insert entries into table
    db_conn.execute(insert_stmt, parameters={"sales_date": last_date[0], "sales_amount": last_date[1]})

    # # commit transactions
    db_conn.commit()

connector.close()