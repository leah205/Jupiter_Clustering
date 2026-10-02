from  webdav3.client import Client
from dotenv import load_dotenv
import os
import tempfile
from astropy.io import fits
import time
import requests



load_dotenv()

print(os.getenv("WEBDAV_HOST"))


options = {
    'webdav_hostname': os.getenv("WEBDAV_HOST"),
 'webdav_login':    os.getenv("WEBDAV_LOGIN"),
 'webdav_password': os.getenv("WEBDAV_PASSWORD")
}

client = Client(options)    

from config.config import cf



def get_dir_path(source):
    return f"HST GO 18055/{source[:10]}/{source}/L3/"


def get_fits_arr(keywords, dir_path):
    files = client.list(dir_path)
    fits_dict = {}
    
    for key in keywords:
        filename = next((f for f in files if key in f), None)
        filepath = dir_path + filename
        with tempfile.NamedTemporaryFile(delete_on_close=True) as fp:
                # print("downloading " + filepath)
            start = time.perf_counter()
            client.download_sync(remote_path = filepath, local_path = fp.name)


    #     url = (
    #     f"{os.getenv('WEBDAV_HOST')}"
    #     f"{filepath}"
    # )  
    #     print(url)
    #     with requests.get(
    #     url,
    #     auth=(
    #         os.getenv("WEBDAV_LOGIN"),
    #         os.getenv("WEBDAV_PASSWORD")
    #     ),
    #     stream=True,
    #     ) as response:
    #         response.raise_for_status()

    #         with open("downloaded_file", "wb") as f:
    #             start = time.perf_counter()
    #             for chunk in response.iter_content(chunk_size=1024 * 1024):
    #                 if chunk:
    #                     f.write(chunk)









    #     end = time.perf_counter()

                # print("elapsed is ", end - start)
                # print(os.listdir("."))

                # fits_dict["key"] = fits.open("." + fp.name)

    return fits_dict



def download_avi():
    start = time.perf_counter()
    with tempfile.NamedTemporaryFile(delete_on_close=True) as fp:

        client.download_sync(remote_path = "20260318UT/2026-03-18-0704_1-Jupiter_685NIR.avi", local_path = "test")
        end = time.perf_counter()
        print("elapsed is ", end - start)



if __name__ == "__main__":
    # print(client.list("/"))
    download_avi()

    # get_fits_arr(["NH3", "BLU"], get_dir_path("20251016UTa"))



