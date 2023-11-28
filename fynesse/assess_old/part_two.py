import folium
import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import seaborn as sns
from geopandas import sjoin
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from ..utils.io_utils import get_or_load
from ..utils.pandas_utils import aligned_concat
from ..utils.plotting_utils import subplots_iter
from ..utils.type_utils import coerce_args
from .part_one import load_place_prices, load_place_tags


def plot_heatmap(data, figsize=None):
    fig, ax = plt.subplots(figsize=(15, 15) if figsize is None else figsize)
    return sns.heatmap(
        data, annot=True, cmap=sns.diverging_palette(261, 20, s=91, l=45, as_cmap=True)
    )


def apply_pca(cnts, n_components=14):
    pca = PCA(n_components=n_components)
    cnts_top_preds = cnts
    cnts_top_normalised = cnts_top_preds.div(cnts_top_preds.std())
    pca.fit(cnts_top_normalised)
    return aligned_concat(
        pd.DataFrame(pca.components_, columns=cnts.columns).rename_axis(["component"]),
        pd.Series(pca.explained_variance_ratio_, name="explained_variance"),
    )


def get_prices_with_building_cnts(world, property_type):
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
    """Note: instead of gdf.loc[x], use gdf.loc[[x]] to retain the type as GeoDataFrame"""

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
        # folium.Popup(data["BoroName"]).add_to(geo_j)
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
    return (
        world.reset_index()
        .groupby(["property_type", "tenure_type"])
        .db_id.count()
        .unstack()
        .rename(columns={"F": "Freehold", "L": "Leasehold"})
    )


def plot_other_price_data(world):
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
    def fetch_house_prices_per_town(db, min_count):
        data = db.execute(
            """SELECT DISTINCT
                med.town_city,
                price_median,
                count
            FROM (
                SELECT
                    town_city,
                    MEDIAN(price) OVER (PARTITION BY town_city) as price_median
                FROM
                    prices_coordinates_data
            ) med
            LEFT JOIN (
                SELECT
                    town_city,
                    COUNT(*) as count
                FROM
                    prices_coordinates_data
                GROUP BY
                    town_city
            ) grp ON (med.town_city = grp.town_city);""",
            return_results=True,
        )

        df = pd.DataFrame(data)

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
