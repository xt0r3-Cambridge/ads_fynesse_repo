from .config import *

from . import access
from .utils.io_utils import load

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""

def query(data):
    pass

def data()
    pass

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError

def get_tags_around(
    data,
    tags,
    box_width=0.02,
    box_height=0.02,
):
    place_name = f"{data['town_city']}, {data['country']}".lower()
    latitude = float(data["latitude"])
    longitude = float(data["longitude"])

    box_north = latitude + box_height / 2
    box_south = latitude - box_height / 2
    box_west = longitude - box_width / 2
    box_east = longitude + box_width / 2
    try:
        pois = ox.features_from_bbox(box_north, box_south, box_east, box_west, tags)
    except InsufficientResponseError:
        pois = pd.DataFrame(columns=cols)

    return pois