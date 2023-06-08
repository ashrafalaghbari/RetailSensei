import pandas as pd
import os
from fredapi import Fred
import sqlalchemy
from google.cloud.sql.connector import Connector, IPTypes

class FredToDB:
    def __init__(self, fred_api_key, db_user, db_pass, db_name, instance_connection_name):

        self.fred_api_key = fred_api_key
        self.db_user = db_user
        self.db_pass = db_pass
        self.db_name = db_name
        self.instance_connection_name = instance_connection_name

    def get_sales(self):
        # get data from FRED
        fred = Fred(api_key=self.fred_api_key)
        # Divide the data by 1000 to convert from millions of dollars to billions of dollars
        retail_sales = fred.get_series('RSXFSN', observation_start='2023-05-01')/1000
        # test the update functionality for the dashboard
        # retail_sales = pd.Series(data=[800000, 900000], index=pd.date_range(start='2023-05-01', periods=2, freq='MS'))/1000

        # Convert the data to a Pandas DataFrame
        df = pd.DataFrame(retail_sales, columns=['sales_amount'])

        # add a monthly frequencey to index date
        df.index.freq = 'MS'

        # Convert the index and sales values of the DataFrame into a list of tuples
        sales_tuples = [(index.date(), float(sales)) for index, sales in df.itertuples()]

        return sales_tuples

    def store_sales(self, sales_tuples):
        # Get environment variables for database connection
        DB_USER = self.db_user
        DB_PASS = self.db_pass
        DB_NAME = self.db_name
        INSTANCE_CONNECTION_NAME = self.instance_connection_name

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

        # insert entries into table
        with pool.connect() as db_conn:
            # Insert the data into the table
            # Insert the data into the table if it doesn't already exist
            insert_stmt = sqlalchemy.text(
                "INSERT INTO retail_sales (sales_date, sales_amount) "
                "SELECT :sales_date, :sales_amount "
                "WHERE NOT EXISTS (SELECT 1 FROM retail_sales "
                "WHERE sales_date = :sales_date)"
            )
            for sale in sales_tuples:
                # insert entries into table
                db_conn.execute(insert_stmt, parameters={"sales_date": sale[0], "sales_amount": sale[1]})

            # # commit transactions
            db_conn.commit()

        connector.close()

fred_to_db = FredToDB(
    fred_api_key=os.environ['FRED_API_KEY'],
    db_user=os.environ['DB_USER'],
    db_pass=os.environ['DB_PASS'],
    db_name=os.environ['DB_NAME'],
    instance_connection_name=os.environ['INSTANCE_CONNECTION_NAME']
)

# get data from FRED and and convert it to a list of tuples
sales_tuples = fred_to_db.get_sales()
# store these tuples in the database
fred_to_db.store_sales(sales_tuples)
