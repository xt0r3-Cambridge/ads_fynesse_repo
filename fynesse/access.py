from .config import *
from .access_base import DatasetManager

from tqdm import tqdm
import zipfile
import pymysql
from datetime import datetime
from pathlib import Path

class PriceDatasetManager(DatasetManager):
    @staticmethod
    def fetch_data():
        year_range=range(2018, 2023)

        paths = []

        for year in year_range:
            for part in [1, 2]:
                filepath = Path(f"../data/{year}_{part}.csv")
                if not filepath.exists() or not config['use_cache']:
                    print(config['use_cache'])
                    DatasetManager.fetch_url(
                        f"{config['price_data_stem']}-{year}-part{part}.csv",
                        filepath,
                    )
                paths.append(filepath)

        return paths

class PostcodeDatasetManager(DatasetManager):
    @staticmethod
    def fetch_data():
        file_dir = Path(f'../data/postcode_data')
        if not file_dir.exists() or not config['use_cache']:
            zip_path = Path(f'{file_dir}.zip')
            DatasetManager.fetch_url(config['postode_data_url'], zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(file_dir)

            zip_file.unlink()

        return [filepath for filepath in file_dir.iterdir() if filepath.suffix == '.csv']

class GlobalDatabaseManager:
    def __init__(self, username, password):
        self.url = config['url']
        self.port = config['port']
        self.username = username
        self.password = password
        self.connect()
        
    def execute(self, sql, *args, return_results = False):
        with self.db.cursor() as cursor:
            cursor.execute(sql, *args)
            if return_results:
                return cursor.fetchall()
            return None
            
    def connect(self):
        self.db = pymysql.connect(
            host=self.url,
            port=self.port,
            user=self.username,
            password=self.password,
            local_infile=True,
            cursorclass=pymysql.cursors.DictCursor,
            client_flag=pymysql.constants.CLIENT.MULTI_STATEMENTS,
        )

    def query(self, sql=None, table='prices_coordinates_data',  n=None):
        return self.execute(f"SELECT * FROM {table} {f'WHERE {sql}' if sql is not None else ''} {f'LIMIT {n}' if n is not None else''}", return_results=True)


class DatasetLoader:
    def __init__(self, db: GlobalDatabaseManager):
        self.db = db

    def create_main_table(self):
        setup_database_cmd = """
SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";
CREATE DATABASE IF NOT EXISTS `property_prices` DEFAULT CHARACTER SET utf8 COLLATE utf8_bin;
USE `property_prices`;
"""
        self.db.execute(setup_database_cmd)

    def _upload_csv(self, filepath, table):
        # TODO: fix SQL injection
        upload_command = f"""
        LOAD DATA LOCAL INFILE %s INTO TABLE `{table}`
        FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '"'
        LINES STARTING BY '' TERMINATED BY '\n';
        """
        self.db.execute(upload_command, (filepath, ))

    def _upload_csv_files_to_table(self, files, table):
        for filepath in tqdm(list(files)):
            self._upload_csv(filepath, table)

    def add_auto_inrements(self, table):
        self.db.execute(f"""
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
        files = PriceDatasetManager.fetch_data()
        self.db.execute("""
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
        self.add_auto_inrements('pp_data')
        self._upload_csv_files_to_table(files, 'pp_data')
        self.db.execute("CREATE INDEX pp_data_postcode_index ON `pp_data` (postcode);")

    def process_postcodes(self):
        files = PostcodeDatasetManager.fetch_data()
        self.db.execute("""
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
        self.add_auto_inrements('postcode_data')
        self._upload_csv_files_to_table(files, 'postcode_data')
        self.db.execute("CREATE INDEX postcode_data_postcode_index ON `postcode_data` (postcode);")


    def create_merged_table(self):
        self.db.execute("""
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
        self.add_auto_inrements('prices_coordinates_data')

        self.db.execute("""
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

    def sample(self, n, table='prices_coordinates_data'):
        return self.db.execute(f"SELECT * FROM {table} LIMIT %s", n, return_results=True)

def legal():
    return f"""Contains HM Land Registry data © Crown copyright and database right 2021. 
This data is licensed under the Open Government Licence v3.0.

Contains OS data © Crown copyright and database right {datetime.now().year}
Contains OS data © Crown copyright and database right {datetime.now().year}
Contains Royal Mail data © Royal Mail copyright and database right {datetime.now().year}
Contains OS data © Crown copyright and database right {datetime.now().year}
Contains Royal Mail data © Royal Mail copyright and Database right {datetime.now().year}
Contains GeoPlace data © Local Government Information House Limited copyright and database right {datetime.now().year}
Source: Office for National Statistics licensed under the Open Government Licence v.3.0
"""