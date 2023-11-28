# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

import geopandas as gpd
import pandas as pd
from geopandas import sjoin
from sklearn.metrics import r2_score
from tqdm import tqdm

from .utils.io_utils import get_or_load
from .utils.osmnx_utils import features_from_bbox
from .utils.stats_utils import male
from .utils.type_utils import coerce_args


@coerce_args
def get_data_around(
    db,
    latitude,
    longitude,
    date,
    property_type,
    height=0.02,
    width=0.02,
    force_reload=False,
):
    """Gets data around a particular point.
    Note: This function is memorized, so it will only query the database once,
    unless force_reload is set to True.
    @param db: The database to query.
    @param latitude: The latitude of the point.
    @param longitude: The longitude of the point.
    @param date: The date of the transaction.
    @param property_type: The property type of the building that got sold.
    @param height: The height of the bounding box.
    @param width: The width of the bounding box.
    @param force_reload: Whether to force reload the data.
    @return: The data around the point.

    The data is composed of the following columns:
    - amenity: The amenity type of the building (e.g. pub).
    - geometry: The geometry of the building.
    - building: The building type (e.g. apartments).
    - nodes: The nodes of the building (e.g. [1, 2, 3]).
    - index_right: The index of the transaction in the database (e.g. 1).
    - price: The price of the transaction (e.g. 1000000.0).
    - date_of_transfer: The date of the transaction (e.g. 2021-01-01).
    - postcode: The postcode of the transaction (e.g. SW1A 1AA).
    - property_type: The property type of the transaction (e.g. D, meaning Detached).
    - new_build_flag: Whether the building is a new build (e.g. Y, meaning Yes).
    - tenure_type: The tenure type of the transaction (e.g. F, meaning Freehold).
    - locality: The locality of the transaction (e.g. BUCKINGHAM PALACE).
    - town_city: The town or city of the transaction (e.g. LONDON).
    - district: The district of the transaction (e.g. WESTMINSTER).
    - county: The county of the transaction (e.g. GREATER LONDON).
    - country: The country of the transaction (e.g. England).
    - latitude: The latitude of the transaction (e.g. 51.1234).
    - longitude: The longitude of the transaction (e.g. -0.1234).
    - db_id: The id of the transaction in the database (e.g. 1).
    - date: The date of the transaction. (e.g. 2021-01-01)
    - quarter: The quarter of the transaction (e.g. 2021Q1).
    - month: The month of the transaction (e.g. 2021-01).
    - year: The year of the transaction (e.g. 2021).
    - outcode: The outcode (first group of the postcode) (e.g. SW1A).
    - plot_area: The area of the plot in square meters (e.g. 1000.0)
    """
    def fetch_data_around(db, latitude, longitude, date, property_type, height, width):
        """
        Fetches data around a particular point.
        @param db: The database to query.
        @param latitude: The latitude of the point.
        @param longitude: The longitude of the point.
        @param date: The date of the transaction.
        @param property_type: The property type of the building that got sold.
        @param height: The height of the bounding box.
        @param width: The width of the bounding box.
        @return: The data around the point.
        """
        bbox_data = db.query(
            where=f"""(`latitude` BETWEEN {latitude - height} AND {latitude + height}) AND
    (`longitude` BETWEEN {longitude - height} AND {longitude + height}) AND
    (`property_type` = '{property_type}') AND
    (`date_of_transfer` < '{pd.Timestamp(date)}')
    """,
        )

        if len(bbox_data) == 0:
            return pd.DataFrame(
                columns=[
                    "amenity",
                    "geometry",
                    "building",
                    "nodes",
                    "index_right",
                    "price",
                    "date_of_transfer",
                    "postcode",
                    "property_type",
                    "new_build_flag",
                    "tenure_type",
                    "locality",
                    "town_city",
                    "district",
                    "county",
                    "country",
                    "latitude",
                    "longitude",
                    "db_id",
                    "date",
                    "quarter",
                    "month",
                    "year",
                    "outcode",
                    "plot_area",
                ]
            )

        bbox_df = bbox_data.assign(
            date=lambda bbox_df: bbox_df.date_of_transfer.apply(pd.Timestamp)
        )
        bbox_df["quarter"] = pd.PeriodIndex(bbox_df.date, freq="Q").astype(str)
        bbox_df["month"] = pd.PeriodIndex(bbox_df.date, freq="M").astype(str)
        bbox_df["year"] = pd.PeriodIndex(bbox_df.date, freq="Y").astype(str)
        bbox_df["outcode"] = bbox_df.postcode.str[:-3].str.strip()

        geometry = gpd.points_from_xy(bbox_df.longitude, bbox_df.latitude)
        bbox_gdf = gpd.GeoDataFrame(bbox_df, geometry=geometry, crs=4326)

        box_north = latitude + height / 2
        box_south = latitude - height / 2
        box_west = longitude - width / 2
        box_east = longitude + width / 2
        bbox_buildings = features_from_bbox(
            box_north,
            box_south,
            box_east,
            box_west,
            tags={"building": True},
        )

        bbox_with_buildings = (
            sjoin(bbox_buildings, bbox_gdf, how="inner", predicate="contains")
            .to_crs(crs=3857)
            .assign(plot_area=lambda df: df.area)
            .to_crs(crs=4326)
        )

        return bbox_with_buildings

    data = get_or_load(
        f"{latitude}_{longitude}_{date}_{property_type}_{height}_{width}",
        fetch_data_around,
        db=db,
        latitude=latitude,
        longitude=longitude,
        date=date,
        property_type=property_type,
        height=height,
        width=width,
        cache_dir="../cache/pred_data",
        force_reload=force_reload,
    )

    return data


def test_strategy(
    strategy,
    db,
    property_type=None,
    min_date="2021-01-01",
    num_samples=1000,
    random_seed=420,
):
    """
    Tests a strategy against the database.
    @param strategy: The strategy to test.
    @param db: The database to test against.
    @param property_type: The property type to test against.
    @param min_date: The minimum date of the transactions to test against.
    @param num_samples: The number of samples to test against.
    @param random_seed: The random seed to use.
    @return: The results of the test.

    The test returns the following metrics:
    - R2 score: The R2 score of the strategy.
    - MALE: The mean absolute log error of the strategy. This captures by how many orders of magnitude the strategy is off.
    """
    samples = db.query(
        where=f"`date_of_transfer` > {min_date}",
        orderby=f"RAND({random_seed})",
        limit=num_samples,
    )
    pred = []
    true = []
    for _, sample in tqdm(samples.iterrows()):
        true.append(sample.price)
        pred.append(
            strategy(
                latitude=sample.latitude,
                longitude=sample.longitude,
                date=sample.date_of_transfer,
                property_type=sample.property_type,
            )
        )

    return pd.DataFrame(
        [
            [
                male(
                    y_true=true,
                    y_pred=pred,
                ),
                r2_score(y_true=true, y_pred=pred),
            ]
        ],
        columns=["male", "r2_score"],
    )


def unpack(data):
    """
    Unpacks a row from the database into a format that can be used by the strategy.
    @param data: The data to unpack.
    @return: The unpacked data.
    """

    return {
        "latitude": data.latitude,
        "longitude": data.longitude,
        "date": data.date,
        "property_type": data.property_type,
    }
