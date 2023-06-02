import os
import pandas as pd
from google.cloud.sql.connector import Connector, IPTypes
import sqlalchemy

class SalesData:
    def __init__(self):
        # Get environment variables for database connection
        self.DB_USER = os.environ['DB_USER']
        self.DB_PASS = os.environ['DB_PASS']
        self.DB_NAME = os.environ['DB_NAME']
        self.INSTANCE_CONNECTION_NAME = os.environ['INSTANCE_CONNECTION_NAME']

        # initialize Connector object
        self.connector = Connector()
        self.ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC

        # create connection pool with 'creator' argument to our connection object function
        self.pool = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=self.getconn,
        )

    # function to return the database connection object
    def getconn(self):
        conn = self.connector.connect(
            self.INSTANCE_CONNECTION_NAME,
            "pg8000",
            user=self.DB_USER,
            password=self.DB_PASS,
            db=self.DB_NAME,
            ip_type=self.ip_type,
        )
        return conn

    # function to query data from database and return a DataFrame
    def save_sales_data(self, filepath):
        with self.pool.connect() as db_conn:
            # select data from the retail_sales table since '2023-03-01'
            query = "SELECT * FROM retail_sales WHERE sales_date >= '2023-05-01'"
            results = db_conn.execute(sqlalchemy.text(query)).fetchall()
            sales_df = pd.DataFrame(results, columns=['id', 'sales_date', 'sales_amount'])

        # check if there is new onvservations in the sales_df
        if len(sales_df) > 0:
            # Convert the sales_date column to a datetime type with a monthly frequency
            # Remove the id column from the DataFrame
            sales_df = sales_df.drop('id', axis=1)
            sales_df['sales_date'] = pd.to_datetime(sales_df['sales_date'])
            sales_df.set_index('sales_date', inplace=True)
            sales_df.index = pd.date_range(start=sales_df.index.min(), end=sales_df.index.max(), freq='MS')
              # add new data points to the test set as this set will be used to monitor the model performance
            # read test.csv file
            test = pd.read_csv(filepath, index_col=0, parse_dates=True)
            # append new data points to the test set
            test = pd.concat([test, sales_df])

            # Save the test.csv file to the specified filepath
            test.to_csv(filepath, index=True)
            print('New sales data has been added to the test set')
        else:
            print('No new sales data has been added to the test set')

    # function to close the database connection
    def close(self):
        self.connector.close()

# Specify the path where the test.csv file should be saved
filepath = 'artifacts/test.csv'

# Create an instance of the SalesData class
sales_data = SalesData()

# Retrieve the new sales data records and save it as a CSV file
sales_data.save_sales_data(filepath)

# Close the database connection
sales_data.close()
