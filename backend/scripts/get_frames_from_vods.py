from backend.services.vod_downloader import process_vod

if __name__ == "__main__":
    # Example usage
    vod_url = "https://www.youtube.com/watch?v=BXet4JEADb4"
    match_id = "m7"
    process_vod(vod_url, match_id)