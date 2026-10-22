"""Download the raw UCI Online Retail file (dataset 352).

This project deliberately uses the file exactly as published, with
its real mess in it: missing customer IDs, cancelled invoices,
negative quantities, zero prices. Cleaning it first would delete
the very thing the engine is meant to measure.
Source: https://archive.ics.uci.edu/dataset/352/online+retail
License: CC BY 4.0.
"""
import hashlib
import pathlib
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    target = ROOT / "data" / "Online Retail.xlsx"
    if target.exists():
        print("exists", target)
        return
    zip_path = ROOT / "data" / "online_retail.zip"
    urllib.request.urlretrieve(URL, zip_path)
    print("zip sha256", hashlib.sha256(zip_path.read_bytes()).hexdigest())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(ROOT / "data")
    zip_path.unlink()
    print("saved", target)


if __name__ == "__main__":
    main()
