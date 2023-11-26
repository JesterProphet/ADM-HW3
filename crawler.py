#!/usr/bin/env python
# coding: utf-8


import concurrent.futures
import math
import multiprocessing
import os
import random
from typing import List

import requests


def download_page(url: str) -> str:
    """
    Function that takes an url and returns its HTML text.

    Args:
        url (str): URL link from which the HTML has to be retrieved.

    Returns:
        html (str): HTML text from the given URL.
    """
    
    # Define a list with different user agents using different instances and web browser
    user_agents = [
        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'},
        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'},
        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.41'},
        {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15'},
        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; Trident/7.0; rv:11.0) like Gecko'},
        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 OPR/76.0.4017.107'},
        {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1'},
        {'User-Agent': 'Mozilla/5.0 (Linux; Android 10; Pixel 3 XL) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Mobile Safari/537.36'},
        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:78.0) Gecko/20100101 Thunderbird/78.12.0'},
        {'User-Agent': 'curl/7.68.0'}
    ]

    # Get response from url using a random user agent from the list
    user_agent = random.choice(user_agents)
    session = requests.Session()
    response = session.get(url, headers=user_agent)
        
    # Retrive HTML if url was successfully loaded by checking if the course title field exists
    if "text-white course-header__course-title" in response.text:
        html = response.text
    elif "FindAMasters Page Not Found" in html:
        html = "Page Not Found"
    else:
        html = None
        
    # Close session again
    session.close()

    return html


def process_urls(lines: List) -> None:
    """
    Function that takes a list of urls and indeces and saves every HTML of it inside a file in the corresponding page folder.

    Args:
        lines (List): List with URLs and indeces.

    Returns:
        None.
    """
    
    # Create and open text file to save all urls that failed to be downloaded
    failed_file = open("failed_files.txt", "a")
    
    # Number of processes using the available CPU
    processes = multiprocessing.cpu_count()

    # Process urls in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=processes) as executor:

        # Iterate through every line
        for line in lines:

            # Current index and url
            i, url = int(line[0]), line[1]

            # Submit the task to the executor
            html = executor.submit(download_page, url)

            # Download HTML if exists and if not save url with its index in the file that will be processed afterwards again
            if html.result():

                 # Page folder
                page_folder = f"pages/page_{str(math.ceil((i)/15)).zfill(3)}"

                # Save HTML in a file
                with open(f"pages/course_{i}.html", "w", encoding="utf-8") as file:
                    file.write(html.result())

            else:

                # Add url and its index to the file that will be processed afterwards
                failed_file.write(f"{i}, {url}\n")
                 
    # Close file with failed urls
    failed_file.close()
    
    # Check if any urls failed and if yes iterate through them again
    if not os.path.getsize("failed_files.txt") == 0:
        
        # Read urls that failed from text file
        with open("failed_files.txt", "r") as file:
            lines = file.read().splitlines()

        # Split the lines into their index and url
        lines = [line.split(", ", 1) for line in lines]
        
        # Close and delete the file
        failed_file.close()
        os.remove("failed_files.txt")
        
        # Restart the process with the failed urls
        process_urls(lines)
        
    # Close and delete the file
    os.remove("failed_files.txt")
    

if __name__ == "__main__":

    # Read urls from text file
    with open("links.txt", "r") as file:
        lines = file.read().splitlines()
        
    # Split the lines into their index and url
    lines = [line.split(", ", 1) for line in lines]
        
    # Save all HTMLs from each url
    process_urls(lines)
