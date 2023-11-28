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
import numpy as np
from statsmodels import api as sm
from geopandas import sjoin
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from tqdm import tqdm

from .utils.io_utils import get_or_load
from .utils.osmnx_utils import features_from_bbox
from .utils.stats_utils import male, mape, train, test, standardise
from .utils.type_utils import coerce_args
from .utils.pandas_utils import aligned_concat
from .utils.plotting_utils import bin_plot
from .assess import load_place_tags, get_tags_per_district, load_place_prices, plot_heatmap


@coerce_args
def get_data_around(
    db,
    latitude,
    longitude,
    date,
    property_type,
    height=0.02,
    width=0.02,
    prevent_lookahead=False,
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
    def fetch_data_around(db, latitude, longitude, date, property_type, height, width, prevent_lookahead):
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
        date_clause = ""
        if prevent_lookahead:
            date_clause = f"AND (`date_of_transfer` < '{pd.Timestamp(date)}')"
            
        bbox_data = db.query(
            where=f"""(`latitude` BETWEEN {latitude - height} AND {latitude + height}) AND
    (`longitude` BETWEEN {longitude - height} AND {longitude + height}) AND
    (`property_type` = '{property_type}') 
    {date_clause}
        """
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
        prevent_lookahead=prevent_lookahead,
        cache_dir="../cache/pred_data",
        force_reload=force_reload,
    )

    return data


def get_same_buildings_across_time(
    db, min_count=10, property_types=["F", "D", "S", "T"], force_reload=False
):
    def fetch_same_buildings_across_time(db, min_count, property_types):
        """We are not returning any buildings of type Other"""
        df = pd.DataFrame(
            db.execute(
                f"""
    SELECT *
    FROM (
        SELECT
            price, `property_type`, `date_of_transfer`, postcode, latitude, longitude,
            ROW_NUMBER() OVER (PARTITION BY latitude, longitude, `property_type` ORDER BY RAND(70)) AS `row_num`
        FROM `prices_coordinates_data`
    ) AS `numbered_rows`
    WHERE `row_num` <= 2
        AND (latitude, longitude, `property_type`) IN (
            SELECT latitude, longitude, `property_type`
            FROM prices_coordinates_data
            WHERE {' OR '.join([f"`property_type` = '{property_type}'" for property_type in property_types])}
            GROUP BY latitude, longitude, `property_type`
            HAVING COUNT(*) >= {min_count}
        );
    """,
                return_results=True,
            )
        )

        idx_list = []
        row_list = []
        col_names = ["log_price1", "log_price2", "date_diff_days", "max_date"]
        for idx, grp in tqdm(df.groupby(["property_type", "latitude", "longitude"])):
            idx_list.append(idx)
            row_list.append(
                [
                    np.log(grp.iloc[0].price),
                    np.log(grp.iloc[1].price),
                    (grp.iloc[0].date_of_transfer - grp.iloc[1].date_of_transfer).days,
                    np.max(
                        [grp.iloc[0].date_of_transfer, grp.iloc[1].date_of_transfer]
                    ),
                ]
            )
        df_formatted = pd.DataFrame(
            row_list,
            index=pd.MultiIndex.from_tuples(
                idx_list, names=["property_type", "latitude", "longitude"]
            ),
            columns=col_names,
        )

        return df_formatted

    df = get_or_load(
        f"same_buildings_across_time_{min_count}",
        fetch_same_buildings_across_time,
        db=db,
        min_count=min_count,
        property_types=property_types,
        cache_dir="../cache/sql/",
        force_reload=force_reload,
        is_pandas=True,
    )
    return df


def fit_plot_area_model(world, db):
    resold_buildings = get_same_buildings_across_time(db, min_count=2, property_types=["S", "D"])
    resold_buildings = resold_buildings.sort_values(by="max_date").reset_index()
    resold_buildings = gpd.GeoDataFrame(
        resold_buildings, geometry=gpd.points_from_xy(resold_buildings.longitude, resold_buildings.latitude), crs=4326
    )
    
    world_price_data = (
        world.assign(plot_area=lambda df: df.area)[["building", "geometry", "plot_area"]]
        .to_crs(crs=4326)
        .sjoin(resold_buildings, how="right")
    )
    world_price_data = world_price_data.query("property_type in ['D', 'S']")
    world_price_data = world_price_data.dropna().drop_duplicates(subset=["latitude", "longitude"])
    world_price_data = world_price_data.assign(
        log_price1_per_sqm=lambda df: df.log_price1 - np.log(df.plot_area),
        log_price2_per_sqm=lambda df: df.log_price2 - np.log(df.plot_area),
    )
    world_price_data
        
    formatted_dataset = world_price_data[
        ["log_price1_per_sqm", "log_price2_per_sqm", "date_diff_days"]
    ].copy()
    X = formatted_dataset.pipe(train)
    X_test = formatted_dataset.pipe(test)
    
    y = X.pop("log_price1_per_sqm")
    y_test = X_test.pop("log_price1_per_sqm")
    
    model = sm.OLS(endog=y, exog=X)

    results = model.fit()
    
    print(f'''Out-of-sample Mean Absolute Log Error: {male(y_test, results.predict(X_test))}
Out-of-sample R2 Score: {r2_score(y_test, results.predict(X_test))}
    '''
    )
    
    return results

def fit_inflation_model(data):
    df_with_date = data.copy()[["log_price1", "log_price2", "date_diff_days"]]
    inflation_X = df_with_date.copy().iloc[: int(len(df_with_date) * 0.8)]
    inflation_X_test = df_with_date.copy().iloc[int(len(df_with_date) * 0.8) :]
    inflation_y = inflation_X.pop("log_price1")
    inflation_y_test = inflation_X_test.pop("log_price1")
    inflation_model = sm.OLS(
        exog=inflation_X,
        endog=inflation_y,
    ).fit()
    return inflation_model

def compare_regressions_with_and_without_inflation(data):
    no_date_df = data.copy()[["log_price1", "log_price2"]]
    no_date_X = no_date_df.copy().iloc[: int(len(no_date_df) * 0.8)]
    no_date_X_test = no_date_df.copy().iloc[int(len(no_date_df) * 0.8) :]
    no_date_y = no_date_X.pop("log_price1")
    no_date_y_test = no_date_X_test.pop("log_price1")
    reg_no_date = sm.OLS(
        exog=no_date_X,
        endog=no_date_y,
    ).fit()
    
    df_with_date = data.copy()[["log_price1", "log_price2", "date_diff_days"]]
    inflation_X = df_with_date.copy().iloc[: int(len(df_with_date) * 0.8)]
    inflation_X_test = df_with_date.copy().iloc[int(len(df_with_date) * 0.8) :]
    inflation_y = inflation_X.pop("log_price1")
    inflation_y_test = inflation_X_test.pop("log_price1")
    inflation_model = sm.OLS(
        exog=inflation_X,
        endog=inflation_y,
    ).fit()
    
    regression_metrics = [
    [r2_score(no_date_y_test, reg_no_date.predict(no_date_X_test)), male(no_date_y_test, reg_no_date.predict(no_date_X_test))],
    [r2_score(inflation_y_test, inflation_model.predict(inflation_X_test)), male(inflation_y_test, inflation_model.predict(inflation_X_test))],
    ]

    return pd.DataFrame(regression_metrics, index=['no inflation data', 'with inflation data'], columns = ['r2_score', 'male'])
    
    


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


def get_dataset_with_counts(
    world,
    tags,
    place,
    n_samples=10000,
    radius_metres=1000,
    batch_size=500,
    random_state=9102,
    id_vars=["db_id", "property_type", "price", "date_of_transfer"],
    db_key=["db_id"],
    force_reload=False,
):
    def fetch_dataset_with_counts(
        world,
        place,
        tags,
        n_samples,
        radius_metres,
        batch_size,
        random_state,
        id_vars,
        db_key,
    ):
        """place e.g. London"""
        pois = load_place_tags(place, tags=tags)

        def fetch_batch_with_counts(samples):
            # display(samples)
            counts_rows = samples.copy()
            counts_rows.columns = counts_rows.columns.get_level_values(-1)
            counts_rows.geometry = counts_rows.geometry.buffer(radius_metres)
            counts_rows = counts_rows.sjoin(
                pois.to_crs(crs=3857), how="left", predicate="contains"
            )

            counts_reshaped = (
                counts_rows.reset_index()
                .melt(
                    id_vars=id_vars,
                    value_vars=tags.keys(),
                )
                .dropna()
            )

            # display(samples)

            # display(counts_rows)
            # display(counts_reshaped)

            counts = (
                counts_reshaped.groupby(db_key + ["variable", "value"])
                .size()
                .unstack(["variable", "value"])
                .pipe(
                    lambda df: df[
                        df.columns.intersection(
                            pd.MultiIndex.from_tuples(
                                [(k, v) for (k, vs) in tags.items() for v in vs]
                            )
                        )
                    ]
                )
            )

            # display(counts)
            batch = samples.join(counts, how="left").fillna(0)
            # display(batch)
            return batch

        samples = (
            world.drop(columns=tags.keys(), errors="ignore")
            .drop(columns=["index_right"], errors="ignore")
            .query("property_type != 'O'")
        ).copy()
        samples = samples.sample(
            min(n_samples, len(samples)), random_state=random_state
        )
        samples.columns = pd.MultiIndex.from_product([["tx_data"], samples.columns])

        samples_list = []
        for i in tqdm(range(0, len(samples), batch_size)):
            counts = fetch_batch_with_counts(samples.iloc[i : i + batch_size])
            samples_list.append(counts)

        dataset = pd.concat(samples_list, join="outer")

        cols = [(k, v) for (k, vs) in tags.items() for v in vs] + [
            ("tx_data", col)
            for col in ["price", "date_of_transfer", "property_type", "postcode"]
        ]
        dataset = dataset[dataset.columns.intersection(cols)].fillna(0)

        return dataset

    dataset = get_or_load(
        "dataset_with_counts",
        fetch_dataset_with_counts,
        world=world,
        place=place,
        tags=tags,
        n_samples=n_samples,
        radius_metres=radius_metres,
        batch_size=batch_size,
        random_state=random_state,
        id_vars=id_vars,
        db_key=db_key,
        force_reload=force_reload,
        cache_dir="../cache/regression_data/",
    )

    return dataset

def model3_compare_lasso_and_standard_regression(dataset, tag_cols):
    reg_data = dataset.sort_values([("tx_data", "date_of_transfer")])[
        tag_cols + [("tx_data", "px_diff_from_median")]
    ].copy()
    
    X = reg_data.pipe(train)
    y = X.pop(("tx_data", "px_diff_from_median"))
    
    X_test = reg_data.pipe(test)
    y_test = X_test.pop(("tx_data", "px_diff_from_median"))
    
    model = sm.OLS(endog=y, exog=X.pipe(standardise))
    
    results = model.fit()
    results_lasso = model.fit_regularized(alpha=8002.16815)
    
    results_lasso.params.pipe(lambda ser: ser[ser > 0])

    regression_results = [
        [r2_score(y_test, results.predict(X_test.pipe(standardise, X=X))),
        ],
        [r2_score(y_test, results_lasso.predict(X_test.pipe(standardise, X=X))),
    ]
    ]

    return pd.DataFrame(regression_results, index=['OLS', 'LASSO'], columns=['outsample_r2_score']), results, results_lasso

def plot_pca_correlations_with_price(prior_counts):
    X = prior_counts.copy()
    y = aligned_concat(X.pop(("price_data", "price")), X.pop(("price_data", "price_adj")))
    pca = PCA(n_components=5).fit(X)
    components = aligned_concat(
        pd.DataFrame(pca.transform(X.div(X.std())), index=prior_counts.index).rename(
            columns=lambda name: f"PC{name:02d}"
        ),
        y,
    )
    return plot_heatmap(components.corr().loc[:, [('price_data', 'price'), ('price_data', 'price_adj')]]).set_title('Principle components vs price correlation')

def pca_vs_price_bin_plot(prior_counts):
    X = prior_counts.copy()
    y = aligned_concat(X.pop(("price_data", "price")), X.pop(("price_data", "price_adj")))
    pca = PCA(n_components=5).fit(X)
    components = aligned_concat(
        pd.DataFrame(pca.transform(X.div(X.std())), index=prior_counts.index).rename(
            columns=lambda name: f"PC{name:02d}"
        ),
        y,
    )
    return bin_plot(
        data=components.assign(
            log_px_adj=lambda df: df[("price_data", "price_adj")].pipe(np.log)
        ),
        x="PC00",
        y="log_px_adj",
    ).set_title('Inflation-adjusted log price vs the value of the first principal component')
    

    
def get_tag_counts_with_median_price(place, db, tags, threshold=0, suffix_length=2, property_type='F'):
    prior_data = get_tags_per_district(
        place,
        db,
        tags=tags,
        threshold=0,
        suffix_length=2,
    )[
        [(k, v) for k, vs in tags.items() for v in vs]
    ].fillna(0)

    place_prices = load_place_prices(place, db) 
    
    median_px = (
        place_prices.query("property_type == @property_type")
        .assign(area_code=lambda df: df.postcode.str[:-suffix_length])
        .groupby(["area_code"])[["price", "price_adj"]]
        .median()
    )
    median_px.columns = pd.MultiIndex.from_product([["price_data"], median_px.columns])

    return prior_data.join(median_px, how="right")


def get_dataset_with_counts_and_medians(
    world,
    tags,
    place,
    n_samples=10000,
    radius_metres=1000,
    batch_size=500,
    random_state=9102,
    id_vars=["db_id", "property_type", "price", "date_of_transfer"],
    db_key=["db_id"],
    force_reload=False,
    reload_counts=False,
):
    def fetch_dataset_with_counts_and_medians(
        world,
        place,
        tags,
        n_samples,
        radius_metres,
        batch_size,
        random_state,
        id_vars,
        db_key,
        reload_counts,
    ):
        dataset = get_dataset_with_counts(
            world=world,
            place=place,
            tags=tags,
            n_samples=n_samples,
            radius_metres=radius_metres,
            batch_size=batch_size,
            random_state=random_state,
            force_reload=reload_counts,
        )

        dataset[("tx_data", "area_code")] = dataset[("tx_data", "postcode")].str[:-2]

        # Add median price
        medians = (
            world.assign(area_code=lambda df: df.postcode.str[:-2])
            .groupby(["property_type", "area_code"])
            .price.median()
            .rename("nearby_median_per_property")
        )

        medians_frame = medians.to_frame().reset_index()
        medians_frame.columns = pd.MultiIndex.from_product(
            [["tx_data"], medians_frame.columns]
        )

        dataset = dataset.merge(
            medians_frame,
            left_on=[("tx_data", "property_type"), ("tx_data", "area_code")],
            right_on=[("tx_data", "property_type"), ("tx_data", "area_code")],
        )
        # medians_frame
        dataset[("tx_data", "log_px_diff_from_median")] = np.log(
            dataset_median[("tx_data", "price")]
        ) - np.log(dataset_median[("tx_data", "nearby_median_per_property")])

        dataset[("tx_data", "px_diff_from_median")] = (
            dataset_median[("tx_data", "price")]
            - dataset_median[("tx_data", "nearby_median_per_property")]
        )

        return dataset

    dataset = get_or_load(
        "dataset_with_counts_and_medians",
        fetch_dataset_with_counts_and_medians,
        world=world,
        place=place,
        tags=tags,
        n_samples=n_samples,
        radius_metres=radius_metres,
        batch_size=batch_size,
        random_state=random_state,
        id_vars=id_vars,
        db_key=db_key,
        force_reload=force_reload,
        reload_counts=reload_counts,
        cache_dir="../cache/regression_data/",
    )

    return dataset
