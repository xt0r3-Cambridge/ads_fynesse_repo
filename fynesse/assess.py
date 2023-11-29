import math

import folium
import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import seaborn as sns
from geopandas import sjoin
from geopandas.tools import sjoin
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from .utils.io_utils import get_or_load
from .utils.osmnx_utils import features_from_place, get_buildings
from .utils.pandas_utils import aligned_concat
from .utils.plotting_utils import subplots_iter
from .utils.type_utils import coerce_args, hash_recur


def plot_heatmap(data, figsize=None):
    """
    Plots a heatmap of the data.
    @param data: The data to plot.
    @param figsize: The size of the figure.
    @return: The heatmap.
    """
    fig, ax = plt.subplots(figsize=(15, 15) if figsize is None else figsize)
    return sns.heatmap(
        data, annot=True, cmap=sns.diverging_palette(261, 20, s=91, l=45, as_cmap=True)
    )


def apply_pca(cnts, n_components=14):
    """
    Applies PCA to the data.
    @param cnts: The data to apply PCA to.
    @param n_components: The number of components to use.
    @return: The PCA applied to the data.
    """
    pca = PCA(n_components=n_components)
    cnts_top_preds = cnts
    cnts_top_normalised = cnts_top_preds.div(cnts_top_preds.std())
    pca.fit(cnts_top_normalised)
    return aligned_concat(
        pd.DataFrame(pca.components_, columns=cnts.columns).rename_axis(["component"]),
        pd.Series(pca.explained_variance_ratio_, name="explained_variance"),
    )


def get_prices_with_building_cnts(world, property_type):
    """
    Gets the prices with building counts fetched using OpenStreetMap.
    @param world: The world to get the prices from (e.g. a DataFrame containing the prices for all London properties)
    @param property_type: The property type to filter for.
    @return: The prices with building counts.

    The data is composed of the following columns:
    - price: The price of the property.
    - building: The building type (e.g. apartments).
    - outcode: The outcode of the property (e.g. W1B).
    - median_price: The median price of the property.
    - building_count: The number of buildings of the same type around the property.
    """
    building_counts = (
        world.groupby(["outcode", "building"])
        .price.count()
        .unstack("building")
        .fillna(0)
    )
    # Only look at building counts that occur more than once around a building on average - this is done to combat overfitting
    building_counts = building_counts.loc[:, building_counts.mean() > 1]

    median_prices = (
        world.query(f"property_type == '{property_type}'")
        .groupby("outcode")
        .price.median()
        .to_frame()
    )
    return median_prices.join(building_counts, how="left")


def get_prices_with_amenity_cnts(world, property_type):
    amenity_counts = (
        world.groupby(["outcode", "amenity"]).price.count().unstack("amenity").fillna(0)
    )
    # Only look at amenity counts that occur more than once around a amenity on average - this is done to combat overfitting
    amenity_counts = amenity_counts.loc[:, amenity_counts.mean() > 1]

    median_prices = (
        world.query(f"property_type == '{property_type}'")
        .groupby("outcode")
        .price.median()
        .to_frame()
    )
    return median_prices.join(amenity_counts, how="left")


def get_price_and_plot_area_correlation(data, property_types):
    """
    Gets the correlation between the price and plot area.
    The correlation is computed by taking the mean of the correlations between the price and plot area for each area.
    An area is defined as the first part of the postcode until the last 2 characters (e.g. the area of CB1 2AB is CB1 2).
    @param data: The data to get the correlation from.
    @param property_types: The property types to filter for.
    @return: The correlation between the price and plot area.

    The data is composed of the following columns:
    - price: The price of the property.
    - plot_area: The area of the plot.
    """
    data = data.assign(plot_area=lambda df: df.area)

    corrs_list = []
    for _, data in (
        data.query("property_type.isin(@property_types)")
        .assign(area_code=lambda df: df.postcode.str[:-2])
        .groupby("area_code")[["price", "plot_area"]]
    ):
        if len(data) < 5:
            continue
        corrs_list.append(data.corr().loc["price", "plot_area"])

    corrs = pd.Series(corrs_list)
    return corrs.mean()


@coerce_args
def plot_polygons_on_map(df, init_lat=None, init_lon=None, zoom_start=10):
    """
    Plots polygons on a map.
    @param df: The data to plot. Must be a GeoDataFrame.
                Note: if you want to plot a single polygon, use df.loc[[x]] instead of df.loc[x] to retain the type as GeoDataFrame.
    @param init_lat: The initial latitude of the map.
    @param init_lon: The initial longitude of the map.
    @param zoom_start: The initial zoom of the map.
    @return: The map.

    The data is composed of the following columns:
    - geometry: The geometry of the polygon.

    The map is composed of the following layers:
    - The polygons.
    """

    if init_lat is None:
        init_lat = float(df.geometry.centroid.y.mean())
    if init_lon is None:
        init_lon = float(df.geometry.centroid.x.mean())

    df = df.to_crs(crs=4326)
    m = folium.Map(
        location=[init_lat, init_lon], zoom_start=zoom_start, tiles="CartoDB positron"
    )
    for _, data in df.iterrows():
        sim_geo = gpd.GeoSeries(data.geometry).simplify(tolerance=0.001)
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(
            data=geo_j, style_function=lambda x: {"fillColor": "orange"}
        )
        geo_j.add_to(m)
    return m


def get_tags_per_district(
    place,
    db,
    tags,
    threshold=1,
    suffix_length=3,
    force_reload_districts=False,
    force_reload_tags=False,
):
    """
    Gets the tags per district.
    @param place: The place to get the tags from. (e.g. "London")
    @param db: The database to get the data from.
    @param tags: The tags to get. (e.g. {"building": ["apartments", "house"]} or {"amenity": ["restaurant"]})
    @param threshold: The threshold to use when filtering the tags. The threshold is over the mean of the tag counts.
    @param suffix_length: The length of the suffix to use when processing the postcodes to get the districts.
    (e.g. 2 for "CB1 2AB" -> "CB1 2")
    @param force_reload_districts: Whether to force reload the district boundaries.
    @param force_reload_tags: Whether to force reload the map containing the locations of the tags.
    @return: The tags per district.

    """
    districts = get_districts(
        place, db, suffix_length, force_reload=force_reload_districts
    )
    tags_df = load_place_tags(place, tags=tags, force_reload=force_reload_tags)
    tags_per_district = districts.sjoin(tags_df, how="left", predicate="contains")
    tag_counts = []
    for tag in tags.keys():
        tag_count = (
            tags_per_district.groupby(["area_code", tag])
            .geometry.count()
            .unstack(tag)
            .fillna(0)
            .pipe(lambda df: df.loc[:, df.mean() > threshold])
        )
        tag_count.columns = pd.MultiIndex.from_product([[tag], tag_count.columns])
        tag_counts.append(tag_count)
    return pd.concat(tag_counts, axis=1, join="outer")


def get_k_means_metrics_raw(world, n_clusters=100):
    """
    Gets the k-means metrics for the raw data and prints them.
    @param world: The world to get the metrics from. (e.g. a DataFrame containing the prices for all London properties)
    @param n_clusters: The number of clusters to use.
    @return: None

    The data is composed of the following columns:
    - price: The price of the property.
    - quarter: The quarter of the property.
    - area_code: The area code of the property.
    - cluster: The cluster of the property.
    - adjusted_price: The adjusted price of the property.
    """
    cluster = KMeans(n_clusters=n_clusters, max_iter=100, n_init=10)
    model = cluster.fit(world.price.pipe(np.log).to_numpy().reshape(-1, 1))

    for i in range(4):
        dfx = world.assign(area_code=world.postcode.str[:-i])
        dfx["cluster"] = model.predict(dfx.price.pipe(np.log).to_numpy().reshape(-1, 1))

        good_pairs = (
            dfx.groupby(["cluster", "area_code"])
            .price.apply(lambda x: len(x) * (len(x) - 1))
            .sum()
        )
        all_pairs = (
            dfx.groupby("area_code").price.apply(lambda x: len(x) * (len(x) - 1)).sum()
        )
        chance_pairs = all_pairs / n_clusters

        print(
            f"Fleiss Kappa for groups by all but the last {i} digits of post code: {(good_pairs - chance_pairs) / (all_pairs - chance_pairs)}"
        )


def get_k_means_metrics_inflation_adjusted(world, n_clusters=100):
    """
    Gets the k-means metrics for the inflation adjusted data and prints them.
    @param world: The world to get the metrics from. (e.g. a DataFrame containing the prices for all London properties)
    @param n_clusters: The number of clusters to use.
    @return: None

    The data is composed of the following columns:
    - price: The unadjusted price of the property.
    - quarter: The quarter of the property.
    - area_code: The area code of the property.
    - cluster: The cluster of the property.
    """
    quarterly_medians = (
        world.groupby(["quarter"]).price.median().rename("quarterly_median_price")
    )
    latest_median = quarterly_medians.iloc[-1]
    dfx = world.merge(
        quarterly_medians, how="left", left_on="quarter", right_index=True
    )
    dfx = dfx.assign(
        adjusted_price=dfx.price.mul(latest_median).div(dfx.quarterly_median_price)
    )

    cluster = KMeans(n_clusters=n_clusters, max_iter=100, n_init=10)
    model = cluster.fit(dfx.adjusted_price.pipe(np.log).to_numpy().reshape(-1, 1))

    dfx["cluster"] = model.predict(
        dfx.adjusted_price.pipe(np.log).to_numpy().reshape(-1, 1)
    )

    for i in range(4):
        dfy = dfx.assign(area_code=world.postcode.str[:-i])

        good_pairs = (
            dfy.groupby(["cluster", "area_code"])
            .price.apply(lambda x: len(x) * (len(x) - 1))
            .sum()
        )
        all_pairs = (
            dfy.groupby("area_code").price.apply(lambda x: len(x) * (len(x) - 1)).sum()
        )
        chance_pairs = all_pairs / n_clusters

        print(
            f"Fleiss Kappa for groups by all but the last {i} digits of post code: {(good_pairs - chance_pairs) / (all_pairs - chance_pairs)}"
        )


def count_freeholds_and_leaseholds(world):
    """
    Counts the number of freeholds and leaseholds.
    @param world: The world to get the counts from. (e.g. a DataFrame containing the prices for all London properties)
    @return: The number of freeholds and leaseholds.
    """
    return (
        world.reset_index()
        .groupby(["property_type", "tenure_type"])
        .db_id.count()
        .unstack()
        .rename(columns={"F": "Freehold", "L": "Leasehold"})
    )


def plot_other_price_data(world):
    """
    Plots the price data for properties that are not houses, flats, or terraced houses.
    @param world: The world to get the data from. (e.g. a DataFrame containing the prices for all London properties)
    """
    ax = subplots_iter(max_subplots=4, subplot_height=4, subplot_width=8, n_cols=2)
    axis = next(ax)
    sns.barplot(
        world.query("property_type == 'O'").groupby("building").price.median(), ax=axis
    )
    axis.tick_params(axis="x", rotation=90)
    axis.set_yscale("log")
    axis.set_title("Median price")
    axis = next(ax)
    sns.barplot(
        world.query("property_type == 'O'").groupby("building").price.mean(), ax=axis
    )
    axis.tick_params(axis="x", rotation=90)
    axis.set_yscale("log")
    axis.set_title("Mean price")
    axis = next(ax)
    sns.barplot(
        world.query("property_type == 'O'")
        .groupby("building")
        .price.apply(lambda grp: grp.max() / grp.min()),
        ax=axis,
    )
    axis.set_yscale("log")
    axis.tick_params(axis="x", rotation=90)
    axis.set_title("Max price / Min price")
    plt.tight_layout(pad=3)

    axis = next(ax)
    sns.barplot(
        world.query("property_type == 'O'").groupby("building").price.count(), ax=axis
    )
    axis.set_yscale("log")
    axis.tick_params(axis="x", rotation=90)
    axis.set_title("Count")
    plt.tight_layout(pad=2)


def plot_other_infos(world):
    """
    Compares the distributions of the prices with and without the 'other' property type.
    @param world: The world to get the data from. (e.g. a DataFrame containing the prices for all London properties)
    """
    ax = subplots_iter(n_cols=2)

    axis = next(ax)
    axis.set(xscale="log")
    sns.histplot(world.price, bins=100, ax=axis)
    sns.histplot(
        world.pipe(lambda data: data[data.property_type != "O"]).price,
        bins=100,
        alpha=0.7,
        ax=axis,
    )
    axis.set_title(
        "Distribution with and without 'other' (blue and orange respectively)"
    )

    axis = next(ax)
    axis.set(xscale="log")
    sns.histplot(
        world.pipe(lambda data: data[data.property_type == "O"]).price,
        bins=300,
        ax=axis,
    )
    axis.set_title("Price distribution of 'other'")
    return plt.tight_layout(pad=5)


def get_districts(place, db, suffix_length, force_reload=False):
    """
    Gets the districts for the place.
    @param place: The place to get the districts from. (e.g. "London")
    @param db: The database to get the data from.
    @param suffix_length: The length of the suffix to use when processing the postcodes to get the districts.
    (e.g. 2 for "CB1 2AB" -> "CB1 2")
    @param force_reload: Whether to force reload the districts.
    @return: The districts.

    Note: The districts are cached in ../cache/districts and are loaded from there if they exist,
    unless force_reload is set to True.
    """
    def fetch_districts(world, suffix_length):
        world = world.assign(area_code=lambda df: df.postcode.str[:-suffix_length])
        districts_list = []
        area_code_list = []
        for area_code, grp in tqdm(world.to_crs(crs=3857).groupby("area_code")):
            district = grp.buffer(1000).unary_union.convex_hull
            districts_list.append(district)
            area_code_list.append(area_code)

        districts = gpd.GeoDataFrame(
            area_code_list, columns=["area_code"], geometry=districts_list, crs=3857
        ).to_crs(4326)
        return districts

    place_prices = load_place_prices(place, db, force_reload=force_reload)

    place_districts = get_or_load(
        f"{place}_districts_{suffix_length}",
        fetch_districts,
        world=place_prices,
        suffix_length=suffix_length,
        cache_dir="../cache/districts",
        force_reload=force_reload,
    )
    return place_districts


def plot_log_prices_per_property_type(world):
    """
    Plots the log prices for each property type (e.g. detached, semi-detached, terraced, flat, other).
    @param world: The world to get the data from. (e.g. a DataFrame containing the prices for all London properties)
    
    The data is composed of the following columns:
    - property_type: The type of the property.
    - price: The price of the property.
    """
    ax = subplots_iter(3, max_subplots=20)
    for property_type in ["O", "D", "S", "T", "F"]:
        dfx = world.query("property_type == @property_type")
        axis = next(ax)
        sns.histplot(dfx.price.pipe(np.log), ax=axis).set_title(
            f"Price distribution | property_type = {property_type}"
        )
        axis.set(xlabel="log_price")
    return plt.tight_layout(pad=5)


def plot_log_prices_per_sq_m_per_property_type(world):
    """
    Plots the log prices per square meter for each property type (e.g. detached, semi-detached, terraced).
    Note that it doesn't make sense to plot the log prices per square meter for flats and other properties,
    as the plot area is often not the actual area of the property.
    @param world: The world to get the data from. (e.g. a DataFrame containing the prices for all London properties)
    
    The data is composed of the following columns:
    - property_type: The type of the property.
    - price: The price of the property.
    - area: The area of the property.
    """
    ax = subplots_iter(3, max_subplots=20)
    for property_type in ["D", "S", "T"]:
        dfx = world.query("property_type == @property_type")
        axis = next(ax)
        sns.histplot(dfx.price.div(dfx.area).pipe(np.log), ax=axis).set_title(
            f"Price per sq. m | property_type = {property_type}"
        )
        axis.set(xlabel="log_price")
    plt.tight_layout(pad=5)


def get_house_prices_per_town(db, min_count=300):
    """
    Gets the house prices per town.
    @param db: The database to get the data from.
    @param min_count: The minimum number of properties per town.
    @return: The house prices per town.
    
    Note: The house prices per town are cached in ../cache/sql and are loaded from there if they exist,
    unless force_reload is set to True.
    """
    def fetch_house_prices_per_town(db, min_count):
        """
        Gets the house prices per town.
        
        The SQL query is quite complex, so it is explained here:
        1. Get the median price per town.
        2. Get the count of properties per town.
        3. Join the two tables on the town.
        4. Filter out towns with less than min_count properties.
        5. Return the resulting table.
        """
        med_prices = pd.DataFrame(db.execute(
            """SELECT
                    `town_city`,
                    MEDIAN(price) OVER (PARTITION BY `town_city`) as `price_median`
                FROM
                    `prices_coordinates_data`
            """,
            return_results=True,
        ))

        counts = pd.DataFrame(db.execute(
            """SELECT
                    `town_city`,
                    COUNT(*) as count
                FROM
                    `prices_coordinates_data`
                GROUP BY
                    `town_city`""",
            return_results=True,
        ))

        df = med_prices.merge(counts, how='inner', left_on='town_city', right_on='town_city')

        df = df.query("town_city.str.len() > 0 & count > @min_count")

        return df

    data = get_or_load(
        "town_median_price",
        fetch_house_prices_per_town,
        db=db,
        min_count=min_count,
        cache_dir="../cache/sql/",
    )

    return data


def get_town_outlines(data):
    """
    Gets the outlines of the towns that are contained in the data.
    @param data: The data to get the outlines from.
    - Must contain a column called 'town_city' that contains the names of the towns.
    @return: The outlines of the towns.
    
    Note: The outlines of the towns are cached in ../cache/osmnx and are loaded from there if they exist,
    unless force_reload is set to True.
    """
    def fetch_town_outlines(town_df):
        poi_list = []
        for _, row in tqdm(list(town_df.iterrows())):
            try:
                town_info = ox.geocode_to_gdf({"city": row.town_city})
                poi_list.append(town_info)
            except Exception as e:
                print(e)
                continue
        pois = pd.concat(poi_list)
        return pois

    towns = get_or_load(
        "uk_town_outlines",
        fetch_town_outlines,
        town_df=data,
        cache_dir="../cache/osmnx",
    )

    towns = towns.query("display_name.str.contains('United Kingdom', case=False)")
    return towns


def plot_choropleth(towns, price_per_town, init_lat=None, init_lon=None):
    """
    Plots a choropleth (a map with different colours for different areas) of the towns.
    @param towns: The towns to plot.
    - Must contain a column called 'town_city' that contains the names of the towns.
    @param price_per_town: The price per town.
    - Must contain a column called 'town_city' that contains the names of the towns.
    @param init_lat: The initial latitude of the map.
    @param init_lon: The initial longitude of the map.
    @return: The map.
    """
    dfx = towns.assign(town_lower=lambda df: df["name"].str.lower().str.strip()).merge(
        price_per_town.assign(
            town_lower=lambda df: df.town_city.str.lower().str.strip(),
            log_price=lambda df: df.price_median.pipe(np.log),
        ),
        left_on="town_lower",
        right_on="town_lower",
    )

    if init_lat is None:
        init_lat = float(dfx.geometry.centroid.y.mean())
    if init_lon is None:
        init_lon = float(dfx.geometry.centroid.x.mean())

    m = folium.Map(
        location=[init_lat, init_lon], zoom_start=7, tiles="CartoDB positron"
    )

    folium.Choropleth(
        geo_data=dfx,
        name="choropleth",
        data=dfx,
        columns=["name", "price_median"],
        key_on="feature.properties.name",
        fill_color="YlGn",
        fill_opacity=1,
        line_opacity=0.3,
        legend_name="Median Price (GBP)",
    ).add_to(m)

    return m


def adjust_inflation(df):
    """
    Adjusts the inflation of the prices in the data by using the median price of the last quarter of 2022 as the base.
    @param df: The data to adjust the inflation of.
    - Must contain a column called 'quarter' that contains the quarters of the properties.
    - Must contain a column called 'price' that contains the prices of the properties.
    @return: The data with the inflation adjusted prices.

    """
    medians = df.groupby("quarter").price.median().rename("quarterly_median")
    y22q4 = medians.loc["2022Q4"]
    df_with_medians = df.merge(medians, left_on="quarter", right_index=True)
    df_with_medians = df_with_medians.assign(
        price_adj=lambda df: df.price.mul(y22q4).div(df.quarterly_median)
    ).drop(columns=["quarterly_median"])
    return df_with_medians


def load_place_prices(place, db, force_reload=False):
    """
    Loads the prices for the place.
    @param place: The place to get the prices from. (e.g. "London")
    @param db: The database to get the data from.
    @param force_reload: Whether to force reload the prices.
    @return: The prices for the place.
    
    Note: The prices are cached in ../cache/osmnx/prices and are loaded from there if they exist,
    unless force_reload is set to True.
    """
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
    """
    Plots the transaction counts over time.
    The measurement is done in 2 ways:
    1. The transaction count over time using an 120 day rolling mean.
    2. The change in the transaction count over time using a 30 day rolling mean minus a 120 day rolling mean.
    @param db: The database to get the data from.
    """
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
            label=f"{'_'*(i-2014)}prime season for home sales",
            alpha=0.5,
        )

    axis.legend()


def get_null_counts(db, force_reload=False):
    """
    Gets the number of null values for each column in the database.
    @param db: The database to get the data from.
    @param force_reload: Whether to force reload the null counts.
    @return: The null counts for each column.
    
    Note: The null counts are cached in ../cache/sql and are loaded from there if they exist,
    unless force_reload is set to True.
    """
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
    """
    Plots the transactions on a map.
    @param db: The database to get the data from.
    @param seed: The seed to use when sampling the transactions.
    @param limit: The number of transactions to sample.
    @return: The map.
    
    Note: The transactions are cached in ../cache/sql/heatmaps and are loaded from there if they exist,
    unless force_reload is set to True.
    """
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
    """
    Loads the tags for the place.
    @param place: The place to get the tags from. (e.g. "London")
    @param tags: The tags to get. (e.g. {"building": ["apartments", "house"]} or {"amenity": ["restaurant"]})
    @param force_reload: Whether to force reload the tags.
    @return: The tags for the place.

    Note: The tags are cached in ../cache/osmnx/tags and are loaded from there if they exist,
    unless force_reload is set to True.
    """
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
    """
    Plots the prices of the 10000 highest and lowest-priced properties.
    @param db: The database to get the data from.
    @param force_reload: Whether to use the cached outliers.
    @return: The outliers.
    """
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
        title="10000 highest prices (~4million total)",
    )
    outlier_prices.query("dir == 'LOW'").price.plot(
        logy=True,
        ax=next(ax),
        xlabel="n-th highest price",
        ylabel="price",
        title="10000 lowest prices (~4million total)",
    )


def get_around(center, world, radius=1000):
    """
    Gets the buildings within some distance of a point.
    @param center: The center to get the buildings around.
    @param world: The world to get the buildings from.
    @param radius: The radius to use when getting the buildings in meters.
    @return: The buildings around the center.
    """
    return world[world.geometry.distance(center.geometry) < radius]


def plot_log_prices_around(gdf, world, radius=1000):
    """
    Plots the log prices around the properties in the GeoDataFrame.
    @param gdf: The GeoDataFrame to get the properties from.
    @param world: The world to get the buildings from.
    @param radius: The radius to use when getting the buildings in meters.
    @return: None
    """
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
