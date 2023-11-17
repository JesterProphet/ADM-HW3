#!/usr/bin/env python
# coding: utf-8


import glob
import heapq
import os
import pickle
import re
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess_text


def add_document(course_id: str, content: str, inverted_index: Dict) -> Dict:
    """
    Function that takes a course and its preprocessed description field and parses it into the inverted index.

    Args:
        course_id (str): Course ID derived from file name.
        content (str): Preprocessed description field.
        inverted_index (Dict): Current inverted index as dictionary.

    Returns:
        inverted_index (Dict): Updated inverted index as dictionary.
    """
    
    # Extract individual words from preprocessed description field
    words = re.findall(r"\b\w+\b", content.lower())
    
    # Parse words into the inverted index
    for word in set(words):
        
        # Create new index if word is not in the current inverted index
        if word not in inverted_index:
            inverted_index[word] = []
            
        # Map course to current word
        inverted_index[word].append(course_id)
    
    return inverted_index


def create_inverted_index1() -> None:
    """
    Function that creates an inverted index based on the preprocessed description field and saves it inside a 
    pickle file.   
    """

    inverted_index = {}
    
    # Create folder vocabularies inside the project
    os.makedirs(f"vocabularies", exist_ok=True)

    # Parse through all course files
    for root, dirs, files in os.walk("courses"):
        for file in files:

            # Check if its a tsv file
            if file.endswith(".tsv"):

                # Load data of current course into a pandas dataframe
                df = pd.read_csv(f"courses/{file}", sep="\t", encoding="utf-8")

                # Check if dataframe is not empty
                if not df.empty:

                    # Extract the preprocessed description field data
                    content = df["preprocessed_description"].values[0]

                    # Check if preprocessed description field is not empty
                    if pd.notna(content):

                        # Derive file name from path which is going to be the course id
                        file_name = file.replace(".tsv", "")

                        # Update the inverted index with current course file
                        inverted_index = add_document(file_name, content, inverted_index)

    # Save inverted index in a pickle file
    with open("vocabularies/vocabulary_preprocessed_description_score1.pkl", "wb") as file:
        pickle.dump(inverted_index, file)

        
def conjunctive_query(query: str):
    """
    Function that takes a query string and returns a pandas dataframe of all courses where every word of the query 
    is inside the course description.   

    Args:
        query (str): Query string.

    Returns:
        df_output (Dataframe): Output of all courses that match the query.
    """

    # Load inverted index from pickle file
    with open("vocabularies/vocabulary_preprocessed_description_score1.pkl", "rb") as file:
        vocabulary = pickle.load(file)

    # Split string into a list with all words
    words = preprocess_text(query).split()

    # Get all courses regarding the first word
    courses = set(vocabulary.get(words[0], []))

    # Search for the other words and always take the set intersection
    for word in words[1:]:
        courses.intersection_update(vocabulary.get(word, []))

    # Column names of output pandas dataframe
    column_names = ["courseName", "universityName", "description", "url", "preprocessed_description"]

    # Create an empty pandas dataframe
    df_output = pd.DataFrame(columns=column_names)

    # Get the information of interest for every course found
    for course in courses:

        # Load data of current course into a pandas dataframe
        df = pd.read_csv(f"courses/{course}.tsv", sep="\t", encoding="utf-8")

        # Add information of interest to output dataframe
        df_output = pd.concat([df_output, df[column_names]], ignore_index=True)
    
    return df_output


def get_courses(column_name: str) -> List:
    """
    Function that retrieves all values of interest given a column name and its course ids.
    
    Args:
        column_name (str): Column name of which the values of interest are being retrieved.
    
    Return:
        courses (List): List with all field values and its course ids.
    """

    courses = []

    # Parse through all course files
    for root, dirs, files in os.walk("courses"):
        for file in files:

            # Check if its a tsv file
            if file.endswith(".tsv"):

                # Load data of current course into a pandas dataframe
                df = pd.read_csv(f"courses/{file}", sep="\t", encoding="utf-8")

                # Check if dataframe is not empty
                if not df.empty:

                    # Check if column field is empty
                    if pd.notna(df[column_name].values[0]):

                        # Derive file name from path which is going to be the course id
                        course_id = int(file.replace(".tsv", "").replace("course_", ""))
                        courses.append((course_id, df[column_name].values[0]))

    # Sort list by the course id
    courses = sorted(courses, key=lambda x: x[0])
    
    return courses


def create_inverted_index2(column_name: str) -> Dict:
    """
    Function that creates an inverted index based on the given column name using the tfIdf score and saves it inside a pickle
    file.
    
    Args:
        column_name (str): Column name of which the inverted index has to be created of.    
    """
    
    # Create folder vocabularies inside the project
    os.makedirs(f"vocabularies", exist_ok=True)
    
    # Retrieve all course values considering the given column and its course ids
    courses = get_courses(column_name)
    
    # Preprocess the data if it's not the preprocessed description field
    if column_name != "preprocessed_description":
        courses = [(course[0], preprocess_text(course[1])) for course in courses]
    
    inverted_index = {}
    vectorizer = TfidfVectorizer()
    
    # Retrieve only the course ids
    course_ids = [course[0] for course in courses]
    
    # Retrieve only the descriptions
    values = [course[1] for course in courses]
    
    # Create tfidf matrix
    tfidf_matrix = vectorizer.fit_transform(values)

    # Keep only the scores larger than 0 and save in inverted index
    for word_id, word in enumerate(vectorizer.get_feature_names_out()):
        word_scores = list(zip(course_ids, tfidf_matrix[:, word_id].toarray().flatten()))
        word_scores = [(course_id, score) for course_id, score in word_scores if score > 0]
        if word_scores:
            inverted_index[word] = word_scores
    
    # Save inverted index in a pickle file
    with open(f"vocabularies/vocabulary_{column_name}.pkl", "wb") as file:
        pickle.dump(inverted_index, file)


def retrieve_courses(query, vocabulary, k=None):
    """
    Function that takes a query string and returns a pandas dataframe of the k courses where every word of the query is inside 
    the given vocabulary and is sorted by the cosine similarity in descending order.

    Args:
        query (str): Query string.

    Returns:
        df_output (Dataframe): Output of k courses that match the query and sorted by the cosine similarity.
    """

    # Preprocess the query string
    words = preprocess_text(query)

    # Calculate tfidf matrix from given query tring
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([words])

    # Retrieve words as feature names
    feature_names = vectorizer.get_feature_names_out()

    # Retrieve tfidf scores
    tfidf_scores = tfidf_matrix.toarray()[0]

    # Final output with each query word and its tfidf score
    tfidf_query = set(zip(feature_names, tfidf_scores))
    tfidf_query = [score[1] for score in tfidf_query]

    # Get all courses where at least one query word is inside
    matching_courses = [(word, vocabulary[word]) for word in words.split() if word in vocabulary]

    # Get all courses where all query words are inside
    matched_courses = set.intersection(*(set(course[0] for course in courses) for _, courses in matching_courses))

    # Reformate list
    matched_courses = [(course_id, item[0], score) for item in matching_courses for course_id, score in item[1] if course_id in matched_courses]

    # Sort list by course id
    matched_courses = set(sorted(matched_courses, key=lambda x: x[0]))

    # Retrive all course ids from matched courses
    course_ids = set([course[0] for course in matched_courses])

    result_heap = []

    # Itereate through every course, calculate the cosine similarity score, and save it inside a heap data structure
    for course_id in course_ids:
        
        # Retrieve tfidf scores and corresponding words from current course
        tfidf_course = [course for course in matched_courses if course[0] == course_id]
        
        # Sort by the words
        tfidf_course = sorted(tfidf_course, key=lambda x: x[1], reverse=False)
        
        # Retrieve only the tfidf score
        tfidf_course = [score[2] for score in tfidf_course]
        
        # Calculate cosine similarity between current course and query
        similarity = cosine_similarity(np.array(tfidf_course).reshape(1, -1), np.array(tfidf_query).reshape(1, -1))
        
        # Save result inside the heap data structure
        heapq.heappush(result_heap, (similarity[0][0], course_id))

    # Retrieve top-k courses if k is given
    if k:
        courses = [(course_id, score) for score, course_id in heapq.nlargest(k, result_heap)]
    else:
        courses = [(course_id, score) for score, course_id in result_heap]

    # Save indices and scores of top-k courses in lists
    indices = [course[0] for course in courses]
    scores = [course[1] for course in courses]

    # Build output inside a pandas dataframe
    columns = ["courseName",
               "universityName",
               "description",
               "url",
               "city",
               "country",
               "fees (€)",
               "startDate",
               "administration",
               "similarity"]
    df_output = pd.DataFrame(columns=columns)

    for index in indices:
        course_df = pd.read_csv(f"courses/course_{index}.tsv", sep="\t")
        course_df = course_df[columns[:-1]]
        df_output = pd.concat([course_df, df_output.dropna(axis=1, how="all")], ignore_index=True)

    df_output["similarity"] = scores

    return df_output


def get_user_input() -> Dict:
    """
    Function that lets a user input some parameters to create a query.

    Returns:
        query_params (Dict): Dictionary that includes all query parameters.
    """
    
    query_params = {}

    # Input for course name
    course_name = input("Enter Course Name (Press Enter to skip): ").strip()
    if course_name:
        query_params["course_name"] = course_name

    # Input for university name
    university_name = input("Enter University Name (Press Enter to skip): ").strip()
    if university_name:
        query_params["university_name"] = university_name

    # Input for city
    city = input("Enter City (Press Enter to skip): ").strip()
    if city:
        query_params["city"] = city

    # Input for fees (€) range
    while True:
        try:
            min_fee_input = input("Enter minimum fees in € (Press Enter to skip): ").strip()
            max_fee_input = input("Enter maximum fees in € (Press Enter to skip): ").strip()
            
            if min_fee_input or max_fee_input:
                min_fee = float(min_fee_input) if min_fee_input else None
                max_fee = float(max_fee_input) if max_fee_input else None
                
                # Check if max fee is larger than min fee
                if max_fee is not None and min_fee is not None and max_fee <= min_fee:
                    raise ValueError("Maximum fees must be larger than minimum fees. Please retry!")
                
                query_params["fees_range"] = (min_fee, max_fee)
            break
        except ValueError as e:
            print(f"Invalid input! {e}")

    # Input for list of countries
    countries = input("Enter a comma-separated list of countries (Press Enter to skip): ").strip()
    if countries:
        query_params["countries"] = [country.strip() for country in countries.split(",")]

    # Input based on courses that have already started
    started = input("Filter based on courses that have already started? (y/n): ").strip().lower()
    if started == "y":
        query_params["started"] = True
    else:
        query_params["started"] = False

    # Input based on the presence of online modality
    online = input("Filter based on the presence of online modality? (y/n): ").strip().lower()
    if online == "y":
        query_params["online"] = True
    else:
        query_params["online"] = False

    return query_params


def get_all_courses():
    """
    Function that returns all courses in a pandas dataframe.

    Returns:
        courses (Dataframe): Dataframe that includes all courses and the columns of interest.
    """   
    
    # Get a list of all tsv files
    tsv_files = glob.glob(f"courses/*.tsv")

    courses = []

    # Iterate through each tsv file and read into a pandas dataframe
    for tsv_file in tsv_files:
        course = pd.read_csv(tsv_file, sep="\t")
        courses.append(course)

    # Concatenate all dataframes
    courses = pd.concat(courses, ignore_index=True)[["courseName",
                                                     "universityName",
                                                     "url",
                                                     "fees (€)",
                                                     "country",
                                                     "startDate",
                                                     "administration"]]
    
    return courses


def course_started(months: List) -> bool:
    """
    Function that determines if a course already started or not given a list with its starting month(s).
    
    Args:
        months (List): List with starting month(s) of given course.

    Returns:
        started (bool): True if the course already started and False if not.
    """    
    
    # Mapping of all months from string to corresponding values
    month_to_number = {
        "Any Month": 0,
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
        "See Course": 99
    }

    # Get current month based on system date
    current_month = datetime.now().month
    
    # Retrieve the earliest starting date of a course if multiple are given
    start_month = min([month_to_number[month] for month in months])

    # Check if current month is in the past or future and return True if it is in the past and False if it is in the future
    if current_month >= start_month:
        started = True
    else:
        started = False
        
    return started


def complex_search_engine():
    """
    Function that first lets a user input some parameters to create a query. Based on this query the function return all courses
    based on the aggregated similarity between all three inverted indexes and applied filters from the query parameters.

    Returns:
        df_output (DataFrame): Pandas dataframe with all courses based on the aggregated similarity between all three inverted 
                               indexes and applied filters.
    """
    
    # Create a query from user input
    query = get_user_input()
    
    # Retrieve all courses if a course name was given
    try:
        with open("vocabularies/vocabulary_courseName.pkl", "rb") as file:
            vocabulary = pickle.load(file)
        courses1 = retrieve_courses(query["course_name"], vocabulary)
    except:
        courses1 = pd.DataFrame()

    # Retrieve all courses if a university name was given
    try:
        with open("vocabularies/vocabulary_universityName.pkl", "rb") as file:
            vocabulary = pickle.load(file)
        courses2 = retrieve_courses(query["university_name"], vocabulary)
    except:
        courses2 = pd.DataFrame()

    # Retrieve all courses if a city was given
    try:
        with open("vocabularies/vocabulary_city.pkl", "rb") as file:
            vocabulary = pickle.load(file)
        courses3 = retrieve_courses(query["city"], vocabulary)
    except:
        courses3 = pd.DataFrame()
    
    # Inner join on the columns courseName and universityName of all dataframes considering every possible combination
    if not courses1.empty:

        merged_courses = courses1[["courseName",
                                   "universityName",
                                   "url",
                                   "fees (€)",
                                   "country",
                                   "startDate",
                                   "administration",
                                   "similarity"]]
        merged_courses = merged_courses.copy()
        merged_courses.rename(columns={"similarity": "similarity_z"}, inplace=True)

        if not courses2.empty:
            merged_courses = pd.merge(merged_courses,
                                      courses2[["courseName", "universityName", "similarity"]],
                                      on=["courseName", "universityName"],
                                      how="inner")

        if not courses3.empty:
            merged_courses = pd.merge(merged_courses,
                                      courses3[["courseName", "universityName", "similarity"]],
                                      on=["courseName", "universityName"],
                                      how="inner")
        
    elif not courses2.empty:

        merged_courses = courses2[["courseName",
                                   "universityName",
                                   "url",
                                   "fees (€)",
                                   "country",
                                   "startDate",
                                   "administration",
                                   "similarity"]]
        merged_courses = merged_courses.copy()
        merged_courses.rename(columns={"similarity": "similarity_z"}, inplace=True)

        if not courses3.empty:
                merged_courses = pd.merge(merged_courses,
                                          courses3[["courseName", "universityName", "similarity"]],
                                          on=["courseName", "universityName"],
                                          how="inner")

    elif not courses3.empty:

        merged_courses = courses3[["courseName",
                                   "universityName",
                                   "url",
                                   "fees (€)",
                                   "country",
                                   "startDate",
                                   "administration",
                                   "similarity"]]
        merged_courses = merged_courses.copy()
        merged_courses.rename(columns={"similarity": "similarity_z"}, inplace=True)

    # Load all courses if all query fields are empty
    else:
        merged_courses = get_all_courses()

    # Retrieve all column names of the dataframe
    column_names = merged_courses.columns.tolist()

    # Those are the column names for similarity if more than one dataframe was originally retrieved and inner joined
    similarities = ["similarity_x", "similarity_y", "similarity_z"]

    # Calculate the arithmetic mean between all similarity scores and delete the other similarity columns
    if all(column_name in column_names for column_name in similarities):
        merged_courses["similarity"] = merged_courses[["similarity_x", "similarity_y", "similarity_z"]].mean(axis=1)
        merged_courses = merged_courses.drop(columns=["similarity_x", "similarity_y", "similarity_z"])
    elif all(column_name in column_names for column_name in similarities[:2]):
        merged_courses["similarity"] = merged_courses[["similarity_x", "similarity_y"]].mean(axis=1)
        merged_courses = merged_courses.drop(columns=["similarity_x", "similarity_y"])
    elif all(column_name in column_names for column_name in [similarities[0], similarities[2]]):
        merged_courses["similarity"] = merged_courses[["similarity_x", "similarity_z"]].mean(axis=1)
        merged_courses = merged_courses.drop(columns=["similarity_x", "similarity_z"])
    elif all(column_name in column_names for column_name in similarities[1:3]):
        merged_courses["similarity"] = merged_courses[["similarity_y", "similarity_z"]].mean(axis=1)
        merged_courses = merged_courses.drop(columns=["similarity_y", "similarity_z"])

    # Replace all values where we don't have a fee with 0
    merged_courses.loc[:, "fees (€)"] = merged_courses["fees (€)"].fillna(0)

    # Replace all values where administration is empty
    merged_courses.loc[:, "administration"] = merged_courses["administration"].fillna("Unknown")

    try:
        # Apply filter for minimun fee if given
        if query["fees_range"][0] is not None:
            merged_courses = merged_courses[merged_courses["fees (€)"] > query["fees_range"][0]]

        # Apply filter for maximum fee if given
        if query["fees_range"][1] is not None:
            merged_courses = merged_courses[merged_courses["fees (€)"] < query["fees_range"][1]]
    except:
        pass

    # Apply filter on country if given
    try:
        query["countries"] = [country.lower() for country in query["countries"]]
        merged_courses = merged_courses[merged_courses["country"].str.lower().isin(query["countries"])]
    except:
        pass

    # Apply filter on start date if given
    merged_courses["started"] = merged_courses["startDate"].str.split(", ").apply(course_started)
    if query["started"]:
        merged_courses = merged_courses[merged_courses["started"] == True]
    else:
        merged_courses = merged_courses[merged_courses["started"] == False]    

    # Apply filter on online modality if given
    if query["online"]:
        merged_courses = merged_courses[merged_courses["administration"].str.contains("online", case=False)]
    else:
        merged_courses = merged_courses[~merged_courses["administration"].str.contains("online", case=False)]

    # Print output of variable of interest
    df_output = merged_courses[["courseName", "universityName", "url"]]
    
    return df_output
