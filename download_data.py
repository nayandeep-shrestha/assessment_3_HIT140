from config_importation import *

def download_history(force=False):
    RAW_HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    if RAW_HISTORY_CSV.exists() and not force:
        print(f"Found {RAW_HISTORY_CSV.name}, skipping download.")
        return
    print("Downloading results.csv ...")
    urlretrieve(HISTORY_URL, RAW_HISTORY_CSV)
    print("Saved to", RAW_HISTORY_CSV)

if __name__ == "__main__":
    download_history()