import shutil
import urllib.request
from abc import ABC, abstractmethod

class DatasetManager(ABC):
    @staticmethod
    def fetch_url(url, filepath):
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