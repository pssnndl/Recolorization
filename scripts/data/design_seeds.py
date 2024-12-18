import requests
from bs4 import BeautifulSoup
import os
import urllib.request
import time

# Set base URL and destination folder
base_url = "https://www.design-seeds.com"  # Adjust this to the actual URL structure of pages
output_folder = r"C:\Users\SowmyaG\Downloads\images_resized-20241102T201315Z-001\design seeds 18 pages"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Set the maximum number of pages to iterate through
max_pages = 18  # Set this as needed
current_page = 1

# Track downloaded images by filename
downloaded_images = set(os.listdir(output_folder))

while current_page <= max_pages:
    # Construct the URL for the current page
    url = f"{base_url}?page={current_page}"  # Modify this URL pattern if necessary

    # Fetch the page
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch page {current_page}. Status code: {response.status_code}")
        break

    print(f"Processing page {current_page}...")

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    images = soup.find_all("img")

    for img in images:
        src = img.get("src")
        if src and src.startswith("https://blogger.googleusercontent.com/img/b/R29vZ2xl/"):
            try:
                # Get the image filename from the URL
                image_name = src.split("/")[-1]
                
                # Check if the image has already been downloaded
                if image_name in downloaded_images:
                    print(f"Skipping duplicate: {image_name}")
                    continue

                # Download and save the image
                image_path = os.path.join(output_folder, image_name)
                urllib.request.urlretrieve(src, image_path)
                print(f"Downloaded: {image_name}")
                
                # Add to downloaded images set
                downloaded_images.add(image_name)
                
                # Delay to avoid overwhelming the server
                time.sleep(0.5)

            except Exception as e:
                print(f"Failed to download {src}. Error: {e}")

    # Move to the next page
    current_page += 1

print("Download completed.")
