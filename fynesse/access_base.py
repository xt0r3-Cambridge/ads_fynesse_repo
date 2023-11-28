import shutil
import urllib.request
from abc import ABC, abstractmethod


class DatasetManager(ABC):
    """
    Abstract class for managing datasets. This class is responsible for downloading and unpacking
    datasets, and for providing a list of files to the final unpacked versions.
    """
    @staticmethod
    def fetch_url(url, filepath):
        """
        Downloads a file from a URL to a local filepath.
        @param url: The URL to download from.
        @param filepath: The filepath to download to.
        """
        # Add headers to request so that the site knows to serve it
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3"
        }
        req = urllib.request.Request(
            url=url,
            headers=headers,
        )
        # Download the file from `url` and save it locally under `filepath`:
        with urllib.request.urlopen(req) as response, open(filepath, "wb") as out_file:
            shutil.copyfileobj(response, out_file)

    @staticmethod
    @abstractmethod
    def fetch_data():
        """Downloads a dataset and returns a list of files to the final unpacked versions"""
        raise NotImplementedError
