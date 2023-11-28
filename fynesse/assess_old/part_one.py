import math

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas.tools import sjoin
from tqdm import tqdm

from ..utils.io_utils import get_or_load
from ..utils.osmnx_utils import features_from_place, get_buildings
from ..utils.pandas_utils import aligned_concat
from ..utils.plotting_utils import subplots_iter
from ..utils.type_utils import hash_recur


def adjust_inflation(df):
    medians = df.groupby("quarter").price.median().rename("quarterly_median")
    y22q4 = medians.loc["2022Q4"]
    df_with_medians = df.merge(medians, left_on="quarter", right_index=True)
    df_with_medians = df_with_medians.assign(
        price_adj=lambda df: df.price.mul(y22q4).div(df.quarterly_median)
    ).drop(columns=["quarterly_median"])
    return df_with_medians


def load_place_prices(place, db, force_reload=False):
    def fetch_place_data(place, db):
        place_data = db.query(f"town_city = '{place.upper()}'")

        place_df = place_data.assign(
            date=lambda place_df: place_df.date_of_transfer.apply(pd.Timestamp)
        )
        place_df["quarter"] = pd.PeriodIndex(place_df.date, freq="Q").astype(str)
        place_df["month"] = pd.PeriodIndex(place_df.date, freq="M").astype(str)
        place_df["year"] = pd.PeriodIndex(place_df.date, freq="Y").astype(str)
        place_df["outcode"] = place_df.postcode.str[:-3].str.strip()

        geometry = gpd.points_from_xy(place_df.longitude, place_df.latitude)
        place_gdf = gpd.GeoDataFrame(place_df, geometry=geometry, crs=4326)

        place_buildings = features_from_place(
            f"{place}, United Kingdom",
            tags={"building": True},
        )

        place_with_buildings = sjoin(
            place_buildings, place_gdf, how="inner", predicate="contains"
        ).to_crs(crs=3857)

        place_with_buildings_inflation_adj = place_with_buildings.pipe(adjust_inflation)

        return place_with_buildings_inflation_adj

    return (
        get_or_load(
            f"{place}_prices",
            fetch_place_data,
            place=place,
            db=db,
            force_reload=force_reload,
            cache_dir="../cache/osmnx/prices",
        )
        .drop_duplicates(subset=["db_id"])
        .reset_index()
        .set_index("db_id")
    )


def plot_tx_count_over_time(db):
    dates = get_or_load(
        "date_of_transfer_cnt",
        db.query,
        cols="`date_of_transfer`, COUNT(*) as 'transaction count'",
        groupby="`date_of_transfer`",
        cache_dir="../cache/sql",
    )

    ax = subplots_iter()

    axis = next(ax)
    dates.set_index("date_of_transfer").rolling(120).mean().plot(
        ax=axis, title="Transaction Count Over Time"
    )

    axis = next(ax)
    dates.set_index("date_of_transfer").pipe(
        lambda df: df.rolling(30).mean().sub(df.rolling(120).mean())
    ).rename(columns={"transaction count": "tx_count change"}).plot(
        ax=axis, title="Transaction Count Change In Mean"
    )
    for i in range(2014, 2023):
        axis.axvspan(
            pd.Timestamp(f"{i}-06-01"),
            pd.Timestamp(f"{i}-09-01"),
            color="green",
            alpha=0.5,
        )


def get_null_counts(db, force_reload=False):
    def fetch_null_counts(db):
        null_cnts = []

        cols = [
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
        ]

        for col in tqdm(cols):
            nulls = db.query(cols=f"COUNT(*) as {col}", where=f"TRIM(`{col}`) = ''")
            null_cnts.append(nulls)
        total = db.query(cols="COUNT(*) as total_element_count")
        null_cnts.append(total)
        return aligned_concat(*null_cnts)

    null_count = get_or_load(
        "null_counts",
        fetch_null_counts,
        db,
        cache_dir="../cache/sql/",
        force_reload=force_reload,
    )

    return null_count.rename_axis("null_counts", axis=1)


def plot_transactions(db, seed=102, limit=100000):
    sample_transactions = get_or_load(
        f"heatmap_{seed}_{limit}",
        db.query,
        orderby=f"RAND({seed})",
        limit=limit,
        cache_dir="../cache/sql/heatmaps",
    )
    geometry = gpd.points_from_xy(
        sample_transactions.longitude, sample_transactions.latitude
    )
    sample_gdf = gpd.GeoDataFrame(sample_transactions, geometry=geometry, crs=4326)
    # sample_gdf["quarter"] = pd.PeriodIndex(sample_gdf.date_of_transfer, freq="Q").astype(str)
    # sample_gdf = sample_gdf.sort_values(by=['quarter'])

    # map_center = [
    #     sample_gdf.geometry.centroid.y.mean(),
    #     sample_gdf.geometry.centroid.x.mean(),
    # ]
    # mymap = folium.Map(location=map_center, zoom_start=7)

    # heat_data = [
    #     [
    #     [row.geometry.centroid.xy[1][0], row.geometry.centroid.xy[0][0], 1]
    #      for _, row in d.iterrows()
    #     ]
    #     for (_, d) in sample_gdf.groupby('quarter')
    # ]

    # time_index = sample_gdf.quarter.drop_duplicates().tolist()

    # folium.plugins.HeatMapWithTime(heat_data, auto_play=True, index=time_index, min_opacity=0, max_opacity=1).add_to(mymap)

    map_center = [
        sample_gdf.geometry.centroid.y.mean(),
        sample_gdf.geometry.centroid.x.mean(),
    ]
    mymap = folium.Map(location=map_center, zoom_start=7)

    heat_data = [
        [point.centroid.xy[1][0], point.centroid.xy[0][0]]
        for point in sample_gdf.geometry
    ]

    folium.plugins.HeatMap(heat_data, min_opacity=0).add_to(mymap)

    return mymap


def load_place_tags(place, tags={"building": True}, force_reload=False):
    def fetch_place_data(place, tags):
        features = features_from_place(
            f"{place}, United Kingdom",
            tags=tags,
        )

        return features

    results = []
    for tag in tags.keys():
        tag_result = get_or_load(
            f"{place}_tags__{hash_recur({tag: tags[tag]})}",
            fetch_place_data,
            place=place,
            tags={tag: tags[tag]},
            force_reload=force_reload,
            cache_dir="../cache/osmnx/tags",
        )
        results.append(tag_result)
    return pd.concat(results, axis=0, join="outer", copy=False)


def plot_outliers(db, force_reload=False):
    outlier_prices = get_or_load(
        "outlier_prices",
        pd.concat,
        [
            db.query(orderby="price DESC", limit=10000).assign(dir=lambda x: "HIGH"),
            db.query(orderby="price ASC", limit=10000).assign(dir=lambda x: "LOW"),
        ],
        cache_dir="../cache/sql",
        force_reload=force_reload,
    )

    outlier_prices

    ax = subplots_iter(max_subplots=5)
    outlier_prices.query("dir == 'HIGH'").price.plot(
        logy=True,
        ax=next(ax),
        xlabel="n-th highest price",
        ylabel="price",
        title="10000 highest prices",
    )
    outlier_prices.query("dir == 'LOW'").price.plot(
        logy=True,
        ax=next(ax),
        xlabel="n-th highest price",
        ylabel="price",
        title="10000 lowest prices",
    )


def get_around(center, world, radius=1000):
    return world[world.geometry.distance(center.geometry) < radius]


def plot_log_prices_around(gdf, world, radius=1000):
    """Radius is in meters"""
    axis = subplots_iter(
        math.floor(math.sqrt(len(gdf))),
        max_subplots=len(gdf),
        subplot_height=10,
        subplot_width=10,
    )
    for idx, data in gdf.iterrows():
        ax = next(axis)

        buildings = world[world.geometry.distance(data.geometry) < radius]

        ax.set_title(f"Log_10(price) around {data.postcode}")
        base = buildings.plot(
            ax=ax, color="white", edgecolor="black", alpha=0.4, aspect="equal"
        )
        buildings.assign(ax=ax, log_price=lambda df: df.price.pipe(np.log10)).plot(
            ax=base,
            column="log_price",
            edgecolor="black",
            legend=True,
        )
