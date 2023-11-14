#!/usr/bin/env python
# coding: utf-8


import heapq
import numpy as np
import os
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from typing import Dict, List

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
    with open("vocabulary1.pkl", "wb") as file:
        pickle.dump(inverted_index, file)

        
def conjunctive_query(query: str) -> Dict:
    """
    Function that takes a query string and returns a pandas dataframe of all courses where every word of the query 
    is inside the course description.   

    Args:
        query (str): Query string.

    Returns:
        df_output (Dict): Output of all courses that match the query.
    """

    # Load inverted index from pickle file
    with open("vocabulary1.pkl", "rb") as file:
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


def get_courses() -> List:
    """
    Function that retrieves all courses preprocessed course descriptions and its course ids.
    
    Return:
        courses (List): List with all preprocessed course descriptions and its course ids.
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

                    # Check if preprocessed description field is empty
                    if pd.notna(df["preprocessed_description"].values[0]):

                        # Derive file name from path which is going to be the course id
                        course_id = int(file.replace(".tsv", "").replace("course_", ""))
                        courses.append((course_id, df["preprocessed_description"].values[0]))

    # Sort list by the course id
    courses = sorted(courses, key=lambda x: x[0])
    
    return courses


def create_inverted_index2() -> None:
    """
    Function that creates an inverted index based on the preprocessed course descriptions field using the tfIdf score and saves
    it inside a pickle file.
    
    Args:
        courses (List): List with all preprocessed course descriptions and its course ids.
    """
    
    courses = get_courses()
    
    inverted_index = {}
    vectorizer = TfidfVectorizer()
    
    # Retrieve only the course ids
    course_ids = [course[0] for course in courses]
    
    # Retrieve only the descriptions
    descriptions = [course[1] for course in courses]
    
    # Create tfidf matrix
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    # Keep only the scores larger than 0 and save in inverted index
    for word_id, word in enumerate(vectorizer.get_feature_names_out()):
        word_scores = list(zip(course_ids, tfidf_matrix[:, word_id].toarray().flatten()))
        word_scores = [(course_id, score) for course_id, score in word_scores if score > 0]
        if word_scores:
            inverted_index[word] = word_scores
    
    # Save inverted index in a pickle file
    with open("vocabulary2.pkl", "wb") as file:
        pickle.dump(inverted_index, file)


def retrieve_courses(query, k=10):
    """
    Function that takes a query string and returns a pandas dataframe of the k courses where every word of the query 
    is inside the course description and is sorted by the cosine similarity.   

    Args:
        query (str): Query string.

    Returns:
        df_output (Dict): Output of k courses that match the query and sorted by the cosine similarity.
    """
    
    # Load inverted index from pickle file
    with open("vocabulary2.pkl", "rb") as file:
        vocabulary = pickle.load(file)

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

    # Retrieve top-k courses
    top_k_courses = [(course_id, score) for score, course_id in heapq.nlargest(k, result_heap)]

    # Save indices and scores of top-k courses in lists
    indices = [course[0] for course in top_k_courses]
    scores = [course[1] for course in top_k_courses]

    # Build output inside a pandas dataframe
    columns = ["courseName", "universityName", "description", "url", "similarity"]
    df_output = pd.DataFrame(columns=columns)

    for index in indices:
        course_df = pd.read_csv(f"courses/course_{index}.tsv", sep="\t")
        course_df = course_df[columns[:-1]]
        df_output = pd.concat([course_df, df_output], ignore_index=True)

    df_output["similarity"] = scores

    return df_output