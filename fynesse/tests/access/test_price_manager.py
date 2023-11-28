import pandas as pd
from nose.tools import assert_equal

def test_fetch_data_price():
    # Test that the data is fetched correct
    price_manager = PriceDatasetManager()
    price_paths = price_manager.fetch_data()

    assert_equal(len(price_paths), 18)
    for i, (year, part) in enumerate(zip(range(2014, 2023), [1, 2])):
        assert_equal(price_paths[i].name, f"{year}_{part}.csv")

    price_data = pd.read_csv(price_paths[0])
    assert price_data.shape == (100000, 16)
    for col in [
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
    ]:
        assert col in price_data.columns

