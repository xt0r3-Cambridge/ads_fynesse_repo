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

from tqdm import tqdm
from geopandas import sjoin
import geopandas as gpd
import pandas as pd
from sklearn.metrics import r2_score

from .utils.io_utils import get_or_load
from .utils.osmnx_utils import features_from_bbox
from .utils.type_utils import coerce_args
from .utils.stats_utils import male

@coerce_args
def get_data_around(db, latitude, longitude, date, property_type, height=0.02, width=0.02, force_reload=False):
    def fetch_data_around(db, latitude, longitude, date, property_type, height, width):
        bbox_data = db.query(
            where=f"""(`latitude` BETWEEN {latitude - height} AND {latitude + height}) AND
    (`longitude` BETWEEN {longitude - height} AND {longitude + height}) AND
    (`property_type` = '{property_type}') AND
    (`date_of_transfer` < '{pd.Timestamp(date)}')
    """,
        )

        if len(bbox_data) == 0:
            return pd.DataFrame(columns=['amenity', 'geometry', 'building', 'nodes', 'index_right', 'price',
       'date_of_transfer', 'postcode', 'property_type', 'new_build_flag',
       'tenure_type', 'locality', 'town_city', 'district', 'county', 'country',
       'latitude', 'longitude', 'db_id', 'date', 'quarter', 'month', 'year',
       'outcode', 'plot_area'])
    
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
    
        bbox_with_buildings = sjoin(
            bbox_buildings, bbox_gdf, how='inner', predicate='contains'
        ).to_crs(crs=3857).assign(plot_area=lambda df: df.area).to_crs(crs=4326)
    
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
                latitude=float(sample.latitude),
                longitude=float(sample.longitude),
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