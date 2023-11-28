import osmnx as ox
from geopandas.tools import sjoin
from osmnx import _overpass, geocoder, osm_xml, settings, utils, utils_geo
from shapely import MultiPolygon, Point, Polygon

from ..utils.io_utils import load

def process_property_type(property_type):
    """
    Processes the property type to a format that can be used by the OSMNX library.
    @param property_type: The property type to be processed.
    @return: The processed property type.
    """
    if property_type is None or property_type == "O":
        return "yes"
    elif property_type == "D":
        return "detached"
    elif property_type == "S":
        return "semidetached_house"
    elif property_type == "T":
        return "terrace"
    elif property_type == "F":
        return "apartments"
    raise ValueError(f"Unknown property type {property_type}")


# Retrieve POIs
def get_buildings(
    data,
    map,
    box_width=0.01,
    box_height=0.01,
    property_type=None,
):
    """
    Retrieves buildings from the OSMNX library.
    @param data: The data to be used for the retrieval. Must contain the following keys: town_city, country, latitude, longitude.
    @param map: The map to be used for the retrieval.
    @param box_width: The width of the bounding box.
    @param box_height: The height of the bounding box.
    @param property_type: The property type to be retrieved.
    @return: The retrieved buildings.
    """
    tags = {"building": True}
    if property_type is not None:
        tags = {"building": process_property_type(property_type)}
    place_name = f"{data['town_city']}, {data['country']}".lower()
    latitude = float(data["latitude"])
    longitude = float(data["longitude"])

    box_north = latitude + box_height / 2
    box_south = latitude - box_height / 2
    box_west = longitude - box_width / 2
    box_east = longitude + box_width / 2
    pois = features_from_bbox(
        box_north,
        box_south,
        box_east,
        box_west,
        tags=tags,
    )

    building_data = sjoin(pois, map, how="inner")

    # display(pois.sample(5, random_state=123124))
    return building_data


##############################################################################################################
############################### Extending the OSMNX library to use less memory ###############################
##############################################################################################################

"""
The OSMNX data takes up a lot of memory and most of it is random columns we don't care about.

It turns out that we can modify a small amount of internal methods to achieve a
data fetching function that requires about 1% of the space of the original dataframe.
This is what we are doing below.

Understanding the specifics of the modifications is not important for 
understanding the analysis process.
"""


def _download_truncated_overpass_features(polygon, tags):
    """
    Retrieve OSM features within boundary from the Overpass API.
    This function is slightly modified so that it truncates every response dataframe before
    returning it, significantly decreasing the memory requirements

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        boundaries to fetch elements within
    tags : dict
        dict of tags used for finding elements in the selected area

    Yields
    ------
    response_json : dict
        a generator of JSON responses from the Overpass server
    """
    # subdivide query polygon to get list of sub-divided polygon coord strings
    polygon_coord_strs = _overpass._make_overpass_polygon_coord_strs(polygon)
    # utils.log(f"Requesting data from API in {len(polygon_coord_strs)} request(s)")

    # pass exterior coordinates of each polygon in list to API, one at a time
    for polygon_coord_str in polygon_coord_strs:
        query_str = _overpass._create_overpass_query(polygon_coord_str, tags)
        response_json = _overpass._overpass_request(data={"data": query_str})
        truncated_json = {}
        for key in ["version", "generator", "osm3s"]:
            truncated_json[key] = response_json[key]
        truncated_json["elements"] = []
        for element in response_json["elements"]:
            if "tags" in element.keys():
                tags = element["tags"]
                truncated_tags = {
                    key: value
                    for (key, value) in tags.items()
                    if key
                    in [
                        "geometry",
                        "building",
                        "amenity",
                        "leisure",
                        "natural",
                        "office",
                        "public_transport",
                        "tourism",
                    ]
                }
                element["tags"] = truncated_tags
            truncated_json["elements"].append(element)
        yield truncated_json


def features_from_bbox(north, south, east, west, tags):
    """
    Create a GeoDataFrame of OSM features within a N, S, E, W bounding box.

    You can use the `settings` module to retrieve a snapshot of historical OSM
    data as of a certain date, or to configure the Overpass server timeout,
    memory allocation, and other custom settings.

    For more details, see: https://wiki.openstreetmap.org/wiki/Map_features

    Parameters
    ----------
    north : float
        northern latitude of bounding box
    south : float
        southern latitude of bounding box
    east : float
        eastern longitude of bounding box
    west : float
        western longitude of bounding box
    tags : dict
        Dict of tags used for finding elements in the selected area. Results
        returned are the union, not intersection of each individual tag.
        Each result matches at least one given tag. The dict keys should be
        OSM tags, (e.g., `building`, `landuse`, `highway`, etc) and the dict
        values should be either `True` to retrieve all items with the given
        tag, or a string to get a single tag-value combination, or a list of
        strings to get multiple values for the given tag. For example,
        `tags = {'building': True}` would return all building footprints in
        the area. `tags = {'amenity':True, 'landuse':['retail','commercial'],
        'highway':'bus_stop'}` would return all amenities, landuse=retail,
        landuse=commercial, and highway=bus_stop.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
    """
    # convert bounding box to a polygon
    polygon = utils_geo.bbox_to_poly(north, south, east, west)

    # create GeoDataFrame of features within this polygon
    return features_from_polygon(polygon, tags)


def features_from_point(center_point, tags, dist=1000):
    """
    Create GeoDataFrame of OSM features within some distance N, S, E, W of a point.

    You can use the `settings` module to retrieve a snapshot of historical OSM
    data as of a certain date, or to configure the Overpass server timeout,
    memory allocation, and other custom settings.

    For more details, see: https://wiki.openstreetmap.org/wiki/Map_features

    Parameters
    ----------
    center_point : tuple
        the (lat, lon) center point around which to get the features
    tags : dict
        Dict of tags used for finding elements in the selected area. Results
        returned are the union, not intersection of each individual tag.
        Each result matches at least one given tag. The dict keys should be
        OSM tags, (e.g., `building`, `landuse`, `highway`, etc) and the dict
        values should be either `True` to retrieve all items with the given
        tag, or a string to get a single tag-value combination, or a list of
        strings to get multiple values for the given tag. For example,
        `tags = {'building': True}` would return all building footprints in
        the area. `tags = {'amenity':True, 'landuse':['retail','commercial'],
        'highway':'bus_stop'}` would return all amenities, landuse=retail,
        landuse=commercial, and highway=bus_stop.
    dist : numeric
        distance in meters

    Returns
    -------
    gdf : geopandas.GeoDataFrame
    """
    # create bounding box from center point and distance in each direction
    north, south, east, west = utils_geo.bbox_from_point(center_point, dist)

    # convert the bounding box to a polygon
    polygon = utils_geo.bbox_to_poly(north, south, east, west)

    # create GeoDataFrame of features within this polygon
    return features_from_polygon(polygon, tags)


def features_from_address(address, tags, dist=1000):
    """
    Create GeoDataFrame of OSM features within some distance N, S, E, W of address.

    You can use the `settings` module to retrieve a snapshot of historical OSM
    data as of a certain date, or to configure the Overpass server timeout,
    memory allocation, and other custom settings.

    For more details, see: https://wiki.openstreetmap.org/wiki/Map_features

    Parameters
    ----------
    address : string
        the address to geocode and use as the central point around which to
        get the features
    tags : dict
        Dict of tags used for finding elements in the selected area. Results
        returned are the union, not intersection of each individual tag.
        Each result matches at least one given tag. The dict keys should be
        OSM tags, (e.g., `building`, `landuse`, `highway`, etc) and the dict
        values should be either `True` to retrieve all items with the given
        tag, or a string to get a single tag-value combination, or a list of
        strings to get multiple values for the given tag. For example,
        `tags = {'building': True}` would return all building footprints in
        the area. `tags = {'amenity':True, 'landuse':['retail','commercial'],
        'highway':'bus_stop'}` would return all amenities, landuse=retail,
        landuse=commercial, and highway=bus_stop.
    dist : numeric
        distance in meters

    Returns
    -------
    gdf : geopandas.GeoDataFrame
    """
    # geocode the address string to a (lat, lon) point
    center_point = geocoder.geocode(query=address)

    # create GeoDataFrame of features around this point
    return features_from_point(center_point, tags, dist=dist)


def features_from_place(query, tags, which_result=None, buffer_dist=None):
    """
    Create GeoDataFrame of OSM features within boundaries of some place(s).

    The query must be geocodable and OSM must have polygon boundaries for the
    geocode result. If OSM does not have a polygon for this place, you can
    instead get features within it using the `features_from_address`
    function, which geocodes the place name to a point and gets the features
    within some distance of that point.

    If OSM does have polygon boundaries for this place but you're not finding
    it, try to vary the query string, pass in a structured query dict, or vary
    the `which_result` argument to use a different geocode result. If you know
    the OSM ID of the place, you can retrieve its boundary polygon using the
    `geocode_to_gdf` function, then pass it to the `features_from_polygon`
    function.

    You can use the `settings` module to retrieve a snapshot of historical OSM
    data as of a certain date, or to configure the Overpass server timeout,
    memory allocation, and other custom settings.

    For more details, see: https://wiki.openstreetmap.org/wiki/Map_features

    Parameters
    ----------
    query : string or dict or list
        the query or queries to geocode to get place boundary polygon(s)
    tags : dict
        Dict of tags used for finding elements in the selected area. Results
        returned are the union, not intersection of each individual tag.
        Each result matches at least one given tag. The dict keys should be
        OSM tags, (e.g., `building`, `landuse`, `highway`, etc) and the dict
        values should be either `True` to retrieve all items with the given
        tag, or a string to get a single tag-value combination, or a list of
        strings to get multiple values for the given tag. For example,
        `tags = {'building': True}` would return all building footprints in
        the area. `tags = {'amenity':True, 'landuse':['retail','commercial'],
        'highway':'bus_stop'}` would return all amenities, landuse=retail,
        landuse=commercial, and highway=bus_stop.
    which_result : int
        which geocoding result to use. if None, auto-select the first
        (Multi)Polygon or raise an error if OSM doesn't return one.
    buffer_dist : float
        deprecated, do not use

    Returns
    -------
    gdf : geopandas.GeoDataFrame
    """
    if buffer_dist is not None:
        warn(
            "The buffer_dist argument as been deprecated and will be removed "
            "in a future release. Buffer your query area directly, if desired.",
            stacklevel=2,
        )

    # create a GeoDataFrame with the spatial boundaries of the place(s)
    if isinstance(query, (str, dict)):
        # if it is a string (place name) or dict (structured place query),
        # then it is a single place
        gdf_place = geocoder.geocode_to_gdf(
            query, which_result=which_result, buffer_dist=buffer_dist
        )
    elif isinstance(query, list):
        # if it is a list, it contains multiple places to get
        gdf_place = geocoder.geocode_to_gdf(query, buffer_dist=buffer_dist)
    else:  # pragma: no cover
        msg = "query must be dict, string, or list of strings"
        raise TypeError(msg)

    # extract the geometry from the GeoDataFrame to use in API query
    polygon = gdf_place["geometry"].unary_union
    utils.log("Constructed place geometry polygon(s) to query API")

    # create GeoDataFrame using this polygon(s) geometry
    return features_from_polygon(polygon, tags)


def features_from_polygon(polygon, tags):
    """
    Create GeoDataFrame of OSM features within boundaries of a (multi)polygon.

    You can use the `settings` module to retrieve a snapshot of historical OSM
    data as of a certain date, or to configure the Overpass server timeout,
    memory allocation, and other custom settings.

    For more details, see: https://wiki.openstreetmap.org/wiki/Map_features

    Parameters
    ----------
    polygon : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        geographic boundaries to fetch features within
    tags : dict
        Dict of tags used for finding elements in the selected area. Results
        returned are the union, not intersection of each individual tag.
        Each result matches at least one given tag. The dict keys should be
        OSM tags, (e.g., `building`, `landuse`, `highway`, etc) and the dict
        values should be either `True` to retrieve all items with the given
        tag, or a string to get a single tag-value combination, or a list of
        strings to get multiple values for the given tag. For example,
        `tags = {'building': True}` would return all building footprints in
        the area. `tags = {'amenity':True, 'landuse':['retail','commercial'],
        'highway':'bus_stop'}` would return all amenities, landuse=retail,
        landuse=commercial, and highway=bus_stop.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
    """
    # verify that the geometry is valid and a Polygon/MultiPolygon
    if not polygon.is_valid:
        msg = "The geometry of `polygon` is invalid"
        raise ValueError(msg)
    if not isinstance(polygon, (Polygon, MultiPolygon)):
        msg = (
            "Boundaries must be a shapely Polygon or MultiPolygon. If you "
            "requested features from place name, make sure your query resolves "
            "to a Polygon or MultiPolygon, and not some other geometry, like a "
            "Point. See OSMnx documentation for details."
        )
        raise TypeError(msg)

    # download the data from OSM - KORNEL: this downloads the truncated tags instead
    response_jsons = _download_truncated_overpass_features(polygon, tags)

    # create GeoDataFrame from the downloaded data
    df = ox.features._create_gdf(response_jsons, polygon, tags)
    try:
        df = df.set_crs(epsg=4326)
    except AttributeError:
        pass
    return df
