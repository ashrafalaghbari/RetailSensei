import os
import pandas as pd
from google.cloud.sql.connector import Connector, IPTypes
import sqlalchemy

class SalesData:
    def __init__(self):
        # Get environment variables for database connection
        # self.DB_USER = os.environ['DB_USER']
        # self.DB_PASS = os.environ['DB_PASS']
        # self.DB_NAME = os.environ['DB_NAME']
        # self.INSTANCE_CONNECTION_NAME = os.environ['INSTANCE_CONNECTION_NAME']
        self.DB_USER = "postgres"
        self.DB_PASS = "626228282@As"
        self.DB_NAME = "fred"
        self.INSTANCE_CONNECTION_NAME = "starlit-brand-297618:asia-southeast1:postgresqlfred"

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
            # query and fetch sales table
            results = db_conn.execute(sqlalchemy.text("SELECT * FROM retail_sales")).fetchall()
            sales_df = pd.DataFrame(results, columns=['id', 'sales_date', 'sales_amount'])

        # Convert the sales_date column to a datetime type with a monthly frequency
        # Remove the id column from the DataFrame
        sales_df = sales_df.drop('id', axis=1)
        sales_df['sales_date'] = pd.to_datetime(sales_df['sales_date'])
        sales_df.set_index('sales_date', inplace=True)
        sales_df.index = pd.date_range(start=sales_df.index.min(), end=sales_df.index.max(), freq='MS')
        # Select only from the beginning of test set because we want to
        # monitor the model performance after adding the new data points to the test set
        sales_df = sales_df['2021-05-01':]

        # Save the test.csv file to the specified filepath
        sales_df.to_csv(filepath, index=True)

        return sales_df

    # function to close the database connection
    def close(self):
        self.connector.close()

# Specify the path where the test.csv file should be saved
filepath = 'artifacts/test.csv'

# Create an instance of the SalesData class
sales_data = SalesData()

# Retrieve the sales data and save it as a CSV file
sales_df = sales_data.save_sales_data(filepath)

# Close the database connection
sales_data.close()
