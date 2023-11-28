import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql
from tqdm import tqdm

from .access_base import DatasetManager
from .config import *


class PriceDatasetManager(DatasetManager):
    @staticmethod
    def fetch_data():
        """
        Fetches the price data from the HM Land Registry website
        :return: A list of filepaths to the downloaded files
        """
        year_range = range(2014, 2023)

        paths = []

        for year in year_range:
            for part in [1, 2]:
                filepath = Path(f"../data/{year}_{part}.csv")
                if not filepath.exists() or not config["use_cache"]:
                    DatasetManager.fetch_url(
                        f"{config['price_data_stem']}-{year}-part{part}.csv",
                        filepath,
                    )
                paths.append(filepath)

        return paths


class PostcodeDatasetManager(DatasetManager):
    @staticmethod
    def fetch_data():
        """
        Fetches the postcode data from the HM Land Registry website
        :return: A list of filepaths to the downloaded files
        """
        file_dir = Path(f"../data/postcode_data")
        if not file_dir.exists() or not config["use_cache"]:
            zip_path = Path(f"{file_dir}.zip")
            DatasetManager.fetch_url(config["postode_data_url"], zip_path)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(file_dir)

            zip_file.unlink()

        return [
            filepath for filepath in file_dir.iterdir() if filepath.suffix == ".csv"
        ]


class MiscellaneousDataManager(DatasetManager):
    """
    This class takes care of all of the things that are not
    part of the main assessment, e.g. the UK town boundary data
    """

    def fetch_uk_town_boundaries():
        """
        Fetches the UK town boundaries from the HM Land Registry website
        :return: A filepath to the downloaded file
        """
        filepath = Path("../data/uk_town_boundaries.geojson")
        if not filepath.exists() or not config["use_cache"]:
            DatasetManager.fetch_url(config["uk_town_boundaries"], filepath)
        return filepath

    fetcher_list = [fetch_uk_town_boundaries]

    @staticmethod
    def fetch_data():
        paths = []
        for fetcher in MiscellaneousDataManager.fetcher_list:
            filepath = fetcher()
            paths.append(filepath)
        return paths


class GlobalDatabaseManager:
    def __init__(self, username, password):
        self.url = config["url"]
        self.port = config["port"]
        self.username = username
        self.password = password
        self.connect()

    def execute(self, sql, *args, return_results=False):
        """
        Executes a SQL command on the database
        :param sql: The SQL command to execute
        :param args: The arguments to pass to the SQL command
        :param return_results: Whether to return the results of the query
        :return: The results of the query, if return_results is True
        """
        with self.db.cursor() as cursor:
            cursor.execute(sql, *args)
            if return_results:
                return cursor.fetchall()
            return None

    def connect(self):
        """
        Connects to the database
        :return: None
        """
        self.db = pymysql.connect(
            host=self.url,
            port=self.port,
            user=self.username,
            password=self.password,
            local_infile=True,
            cursorclass=pymysql.cursors.DictCursor,
            client_flag=pymysql.constants.CLIENT.MULTI_STATEMENTS,
        )

    def query(
        self,
        where=None,
        cols="*",
        groupby=None,
        table="prices_coordinates_data",
        orderby=None,
        having=None,
        limit=None,
        as_pandas=True,
    ):
        """
        Queries the database
        :param where: The WHERE clause of the query
        :param cols: The columns to select
        :param groupby: The GROUP BY clause of the query
        :param table: The table to query
        :param orderby: The ORDER BY clause of the query
        :param having: The HAVING clause of the query
        :param limit: The LIMIT clause of the query
        :param as_pandas: Whether to return the results as a pandas dataframe
        :return: The results of the query
        """
        query = self.generate_query(
            where=where,
            cols=cols,
            groupby=groupby,
            table=table,
            orderby=orderby,
            having=having,
            limit=limit,
        )
        res = self.execute(query, return_results=True)
        if as_pandas:
            res = pd.DataFrame(res).replace(r"^\s*$", np.nan, regex=True)
        return res

    def generate_query(
        self,
        where=None,
        cols="*",
        groupby=None,
        table="prices_coordinates_data",
        orderby=None,
        having=None,
        limit=None,
    ):
        """
        Generates a SQL query
        :param where: The WHERE clause of the query
        :param cols: The columns to select
        :param groupby: The GROUP BY clause of the query
        :param table: The table to query
        :param orderby: The ORDER BY clause of the query
        :param having: The HAVING clause of the query
        :param limit: The LIMIT clause of the query
        :return: The generated query
        """
        where = f"WHERE {where}" if where is not None else ""
        groupby = f"GROUP BY {groupby}" if groupby is not None else ""
        limit = f"LIMIT {limit}" if limit is not None else ""
        orderby = f"ORDER BY {orderby}" if orderby is not None else ""
        having = f"HAVING {having}" if having is not None else ""
        query = (
            f"(SELECT {cols} FROM {table} {where} {groupby} {orderby} {having} {limit})"
        )
        return query


class DatasetLoader:
    def __init__(self, db: GlobalDatabaseManager):
        self.db = db

    def create_main_table(self):
        """
        Creates the main table in the database
        An explanation for the steps the command below performs:
        NO_AUTO_VALUE_ON_ZERO: Don't automatically assign a value of 0 to an auto-incremented column
        time_zone: Set the timezone to UTC
        Create the database if it doesn't exist
        Use the database
        """
        setup_database_cmd = """
SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";
CREATE DATABASE IF NOT EXISTS `property_prices` DEFAULT CHARACTER SET utf8 COLLATE utf8_bin;
USE `property_prices`;
"""
        self.db.execute(setup_database_cmd)

    def _upload_csv(self, filepath, table):
        """
        Uploads a csv file to the database
        :param filepath: The filepath of the csv file
        :param table: The table to upload the csv file to
        :return: None

        An explanation for the steps the command below performs:
        LOAD DATA LOCAL INFILE: Load the data from a local file
        INTO TABLE: Upload the data to the specified table
        FIELDS TERMINATED BY: Specify the field separator
        OPTIONALLY ENCLOSED BY: Specify the field enclosure character
        LINES STARTING BY: Specify the line prefix
        LINES TERMINATED BY: Specify the line suffix
        """
        upload_command = f"""
        LOAD DATA LOCAL INFILE %s INTO TABLE `{table}`
        FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '"'
        LINES STARTING BY '' TERMINATED BY '\n';
        """
        self.db.execute(upload_command, (filepath,))

    def _upload_csv_files_to_table(self, files, table):
        """
        Uploads a list of csv files to the database
        :param files: The list of csv files to upload
        :param table: The table to upload the csv files to
        :return: None
        """
        for filepath in tqdm(list(files)):
            self._upload_csv(filepath, table)

    def add_auto_inrements(self, table):
        """
        Adds auto-increments to a table
        :param table: The table to add auto-increments to
        :return: None

        An explanation for the steps the command below performs:
        ADD PRIMARY KEY: Add a primary key to the table
        MODIFY: Modify the primary key column
        AUTO_INCREMENT: Set the auto-increment value to 1 for the primary key column.
        """
        self.db.execute(
            f"""
        --
        -- Indexes for table
        --
        ALTER TABLE {table}
         ADD PRIMARY KEY (`db_id`);
        
        ALTER TABLE {table}
        MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;
        """
        )

    def process_prices(self):
        """
        Processes the price data
        :return: None

        An explanation for the steps the command below performs:
        DROP TABLE IF EXISTS: Drop the table if it already exists
        CREATE TABLE IF NOT EXISTS: Create the table if it doesn't exist

        The types are explained below:
        tinytext: A string with a maximum length of 255 characters
        varchar: A string with a maximum length of 255 characters
        int: An integer
        date: A date
        enum: An enumeration (a set of strings)
        decimal: A decimal number

        The NOT NULL constraint means that the column cannot be null.
        The AUTO_INCREMENT constraint means that the column will automatically increment by 1 for each row.

        The DEFAULT CHARSET=utf8 COLLATE=utf8_bin means that the table will use the utf8 character set and the utf8_bin collation.
        The AUTO_INCREMENT=1 means that the auto-increment value will start at 1."""
        files = PriceDatasetManager.fetch_data()
        self.db.execute(
            """
--
-- Table structure for table `pp_data`
--
DROP TABLE IF EXISTS `pp_data`;
CREATE TABLE IF NOT EXISTS `pp_data` (
  `transaction_unique_identifier` tinytext COLLATE utf8_bin NOT NULL,
  `price` int(10) unsigned NOT NULL,
  `date_of_transfer` date NOT NULL,
  `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
  `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
  `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
  `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
  `primary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
  `secondary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
  `street` tinytext COLLATE utf8_bin NOT NULL,
  `locality` tinytext COLLATE utf8_bin NOT NULL,
  `town_city` tinytext COLLATE utf8_bin NOT NULL,
  `district` tinytext COLLATE utf8_bin NOT NULL,
  `county` tinytext COLLATE utf8_bin NOT NULL,
  `ppd_category_type` varchar(2) COLLATE utf8_bin NOT NULL,
  `record_status` varchar(2) COLLATE utf8_bin NOT NULL,
  `db_id` bigint(20) unsigned NOT NULL
) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;
"""
        )
        self.add_auto_inrements("pp_data")
        self._upload_csv_files_to_table(files, "pp_data")
        self.db.execute("CREATE INDEX pp_data_postcode_index ON `pp_data` (postcode);")

    def process_postcodes(self):
        """
        Processes the postcode data
        :return: None

        An explanation for the steps the command below performs:
        DROP TABLE IF EXISTS: Drop the table if it already exists
        CREATE TABLE IF NOT EXISTS: Create the table if it doesn't exist

        The types are explained below:
        tinytext: A string with a maximum length of 255 characters
        varchar: A string with a maximum length of 255 characters
        int: An integer
        date: A date
        enum: An enumeration (a set of strings)
        decimal: A decimal number

        The NOT NULL constraint means that the column cannot be null.
        The AUTO_INCREMENT constraint means that the column will automatically increment by 1 for each row.

        The DEFAULT CHARSET=utf8 COLLATE=utf8_bin means that the table will use the utf8 character set and the utf8_bin collation.
        The AUTO_INCREMENT=1 means that the auto-increment value will start at 1.
        """

        files = PostcodeDatasetManager.fetch_data()
        self.db.execute(
            """
USE `property_prices`;
--
-- Table structure for table `postcode_data`
--
DROP TABLE IF EXISTS `postcode_data`;
CREATE TABLE IF NOT EXISTS `postcode_data` (
  `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
  `status` enum('live','terminated') NOT NULL,
  `usertype` enum('small', 'large') NOT NULL,
  `easting` int unsigned,
  `northing` int unsigned,
  `positional_quality_indicator` int NOT NULL,
  `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
  `latitude` decimal(11,8) NOT NULL,
  `longitude` decimal(10,8) NOT NULL,
  `postcode_no_space` tinytext COLLATE utf8_bin NOT NULL,
  `postcode_fixed_width_seven` varchar(7) COLLATE utf8_bin NOT NULL,
  `postcode_fixed_width_eight` varchar(8) COLLATE utf8_bin NOT NULL,
  `postcode_area` varchar(2) COLLATE utf8_bin NOT NULL,
  `postcode_district` varchar(4) COLLATE utf8_bin NOT NULL,
  `postcode_sector` varchar(6) COLLATE utf8_bin NOT NULL,
  `outcode` varchar(4) COLLATE utf8_bin NOT NULL,
  `incode` varchar(3)  COLLATE utf8_bin NOT NULL,
  `db_id` bigint(20) unsigned NOT NULL
) DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
"""
        )
        self.add_auto_inrements("postcode_data")
        self._upload_csv_files_to_table(files, "postcode_data")
        self.db.execute(
            "CREATE INDEX postcode_data_postcode_index ON `postcode_data` (postcode);"
        )

    def create_merged_table(self):
        """
        Creates a merged table from the price and postcode data
        :return: None
        
        An explanation for the steps the command below performs:
        DROP TABLE IF EXISTS: Drop the table if it already exists
        CREATE TABLE IF NOT EXISTS: Create the table if it doesn't exist
        
        The types are explained below:
        tinytext: A string with a maximum length of 255 characters
        varchar: A string with a maximum length of 255 characters
        int: An integer
        date: A date
        enum: An enumeration (a set of strings)
        decimal: A decimal number
        
        The NOT NULL constraint means that the column cannot be null.
        The AUTO_INCREMENT constraint means that the column will automatically increment by 1 for each row.
        
        The DEFAULT CHARSET=utf8 COLLATE=utf8_bin means that the table will use the utf8 character set and the utf8_bin collation.
        The AUTO_INCREMENT=1 means that the auto-increment value will start at 1.
        
        The SELECT statement is explained below:
        SELECT: Select the following columns
        FROM: From the following tables
        INNER JOIN: Join the following tables
        ON: On the following condition
        
        The condition is explained below:
        `pp_data`.postcode = `postcode_data`.postcode
        This means that the postcode column in the pp_data table must be equal to the postcode column in the postcode_data table.

        The CREATE INDEX statements create indexes on the specified columns.
        The columns are: price, date_of_transfer, town_city.
        This is done to speed up queries on those columns.
        """
        self.db.execute(
            """
USE `property_prices`;
--
-- Table structure for table `prices_coordinates_data`
--
DROP TABLE IF EXISTS `prices_coordinates_data`;
CREATE TABLE IF NOT EXISTS `prices_coordinates_data` (
  `price` int(10) unsigned NOT NULL,
  `date_of_transfer` date NOT NULL,
  `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
  `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
  `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
  `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
  `locality` tinytext COLLATE utf8_bin NOT NULL,
  `town_city` tinytext COLLATE utf8_bin NOT NULL,
  `district` tinytext COLLATE utf8_bin NOT NULL,
  `county` tinytext COLLATE utf8_bin NOT NULL,
  `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
  `latitude` decimal(11,8) NOT NULL,
  `longitude` decimal(10,8) NOT NULL,
  `db_id` bigint(20) unsigned NOT NULL
) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;
"""
        )
        self.add_auto_inrements("prices_coordinates_data")

        self.db.execute(
            """
INSERT INTO `prices_coordinates_data` (
price,
date_of_transfer,
postcode,
property_type,
new_build_flag,
tenure_type,
locality,
town_city,
district,
county,
country,
latitude,
longitude
)
SELECT
price,
date_of_transfer,
`pp_data`.postcode,
property_type,
new_build_flag,
tenure_type,
locality,
town_city,
district,
county,
country,
latitude,
longitude
FROM (
 `pp_data`
 INNER JOIN
     `postcode_data`
 ON 
     `pp_data`.postcode = `postcode_data`.postcode
)
"""
        )

        self.db.execute(
            "CREATE INDEX merged_price_index ON `prices_coordinates_data` (price);"
        )
        self.db.execute(
            "CREATE INDEX merged_date_of_transfer_index ON `prices_coordinates_data` (date_of_transfer);"
        )
        # self.db.execute("CREATE INDEX merged_town_city_index ON `prices_coordinates_data` (town_city);")

    def sample(self, n, table="prices_coordinates_data"):
        return self.db.execute(
            f"SELECT * FROM {table} LIMIT %s", n, return_results=True
        )


def legal():
    """
    Returns the legal statement for the data
    :return: The legal statement for the data
    """
    return f"""Contains HM Land Registry data © Crown copyright and database right 2021. 
This data is licensed under the Open Government Licence v3.0.

Contains OS data © Crown copyright and database right {datetime.now().year}
Contains Royal Mail data © Royal Mail copyright and database right {datetime.now().year}
Contains GeoPlace data © Local Government Information House Limited copyright and database right {datetime.now().year}
Source: Office for National Statistics licensed under the Open Government Licence v.3.0
"""
