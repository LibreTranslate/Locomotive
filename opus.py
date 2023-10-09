import requests
import datetime
import json
import os

API_BASE = "https://opus.nlpl.eu/opusapi/"

def opus_datasets():
    r = requests.get(f"{API_BASE}?corpora=True")
    if r.status_code != 200:
        raise Exception(f"Cannot fetch dataset list from opus (status: {r.status_code})")
    res = r.json()
    return res["corpora"]

def get_opus_dataset_url(corpora, from_code, to_code, run_dir):
    current_dir = os.path.dirname(__file__)
    opus_cache = os.path.join(run_dir, "opus_cache.json")

    cache = {}
    if os.path.isfile(opus_cache):
        with open(opus_cache, "r", encoding="utf-8") as f:
            cache = json.loads(f.read())
    
    key = f"{corpora}-{from_code}-{to_code}"
    if key in cache:
        return cache[key]

    r = requests.get(f"{API_BASE}?corpus={corpora}&source={from_code}&target={to_code}&preprocessing=moses&version=latest")
    if r.status_code != 200:
        raise Exception(f"Cannot fetch dataset from opus (status: {r.status_code})")
    res = r.json()

    if not "corpora" in res:
        raise Exception(f"Invalid response from opus: {res}")
    if len(res["corpora"]) == 0:
        raise Exception(f"No corpora is available for {corpora} ({from_code}-{to_code})")
    if len(res["corpora"]) > 1:
        print(f"WARN: Multiple corpora found for {corpora} ({from_code}-{to_code}), using first")
    
    url = res["corpora"][0]["url"]
    cache[key] = url

    with open(opus_cache, "w", encoding="utf-8") as f:
        f.write(json.dumps(cache))

    return url


if __name__ == "__main__":
    print(f"# OPUS datasets")
    print("")
    print(f"Updated: {datetime.date.today()}")
    print("")

    ds = opus_datasets()
    for d in ds:
        print(f" * {d}")
