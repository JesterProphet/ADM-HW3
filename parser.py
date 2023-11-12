#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
from typing import List, Dict

from bs4 import BeautifulSoup


def get_htmls(folder_path: str) -> List:
    """
    Function that takes a folder path, extracts all HTML files and sort them by course number.

    Args:
        folder_path (str): Folder path where all HTML files are saved.

    Returns:
        htmls (List): Sorted list by course number of all HTMLs.
    """
    
    file_paths = []

    # Iterate through all files including subfolders
    for root, dirs, files in os.walk(folder_path):
        for file in files:

            # Keep only the paths from all HTMLs
                if file.endswith(".html"):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)

    # Extract the course number from every path and sort the paths by the course number
    course_numbers = [path.split("/")[-1].split(".")[0].split("_")[1] for path in file_paths]
    tmp_list = list(zip(file_paths, course_numbers))
    sorted_list = sorted(tmp_list, key=lambda x: int(x[1]))
    htmls = [x[0] for x in sorted_list]

    return htmls


def parse_html(html: str) -> Dict:
    """
    Function that takes a file to a HTML file, loads it and extracts the values of interest.

    Args:
        html (str): File path of HTML to be parsed.

    Returns:
        data (Dict): Dictionary of extracted values of interest.
    """
    
    # Read HTML content from saved file
    with open(html, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Parse HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Find course name
    try:
        courseName = soup.find("h1", class_="text-white course-header__course-title").text.replace("\xa0\xa0", "")
    except:
        courseName = ""

    # Find university name
    try:
        universityName = soup.find("a", class_="course-header__institution text-white font-weight-light d-block d-md-inline").text
    except:
        universityName = ""

    # Find faculty name
    try:
        facultyName = soup.find("a", class_="course-header__department text-white font-weight-light d-block d-md-inline mt-2 mt-md-auto").text
    except:
        facultyName = ""

    # Determine if it's full time or part time
    try:
        isItFullTime = soup.find("a", class_="inheritFont concealLink text-decoration-none text-gray-600").text
    except:
        isItFullTime = ""

    # Find short description
    try:
        paragraphs = soup.find("div", class_="course-sections course-sections__description col-xs-24")
        paragraphs = paragraphs.find("div", id="Snippet").find_all("p")
        description = " ".join([paragraph.text.strip() for paragraph in paragraphs])
    except:
        description = ""

    # Find start date
    try:
        startDate = soup.find("span", title="Start dates").text
    except:
        startDate = ""

    # Find fees
    try:
        fees = soup.find("div", class_="course-sections course-sections__fees tight col-xs-24")
        fees = fees.find("div", class_="course-sections__content").find("p").text
    except:
        fees = ""

    # Find modality
    try:
        modality = soup.find("a", class_="inheritFont concealLink text-gray-600 text-decoration-none").text
    except:
        modality = ""

    # Find duration
    try:
        duration = soup.find("span", class_="key-info__content key-info__duration py-2 pr-md-3 d-block d-md-inline-block").text
    except:
        duration = ""

    # Find city
    try:
        city = soup.find("a", class_="card-badge text-wrap text-left badge badge-gray-200 p-2 m-1 font-weight-light course-data course-data__city").text
    except:
        city = ""

    # Find country
    try:
        country = soup.find("a", class_="card-badge text-wrap text-left badge badge-gray-200 p-2 m-1 font-weight-light course-data course-data__country").text
    except:
        country = ""

    # Find administration
    try:
        administration = soup.find("a", class_="card-badge text-wrap text-left badge badge-gray-200 p-2 m-1 font-weight-light course-data course-data__on-campus").text
    except:
        administration = ""
    
    # Find url
    url = soup.find("link", rel="canonical")["href"]
    
    data = {
            "courseName": courseName,
            "universityName": universityName,
            "facultyName": facultyName,
            "isItFullTime": isItFullTime,
            "description": description,
            "startDate": startDate,
            "fees": fees,
            "modality": modality,
            "duration": duration,
            "city": city,
            "country": country,
            "administration": administration,
            "url": url
           }

    return data
    

if __name__ == "__main__":

    # Folder path where all HTML files are saved
    folder_path = "/Users/andre/Data Science/Semester_1/ADM/HW3/pages"
    
    # Folder path where all course information are going to be saved
    courses_path = "/Users/andre/Data Science/Semester_1/ADM/HW3/courses"
    
    # Extract all file paths from every HTML saved
    htmls = get_htmls(folder_path)

    # Iterate through every HTML file and extract values of interest
    for i, html in enumerate(htmls):
        
        # Check if the HTML is not from an offline url and skip if yes
        if os.path.getsize(html) / 1024 > 10:
            
            # Specify column names
            column_names = [
                            "courseName",
                            "universityName",
                            "facultyName",
                            "isItFullTime",
                            "description",
                            "startDate",
                            "fees",
                            "modality",
                            "duration",
                            "city",
                            "country",
                            "administration",
                            "url"
                           ]

            # Create empty pandas dataframe
            df_courses = pd.DataFrame(columns=column_names)

            # Extract data
            data = parse_html(html)

            # Append data to dataframe
            df_courses = pd.concat([df_courses, pd.DataFrame([data], columns=df_courses.columns)], ignore_index=True)

            # Save data inside a tsv file
            df_courses.to_csv(f"{courses_path}/course_{i+1}.tsv", sep="\t", index=False)
