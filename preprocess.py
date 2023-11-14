#!/usr/bin/env python
# coding: utf-8


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os
import pandas as pd
import re
import string
from urllib.parse import urlparse

nltk.download("stopwords")
nltk.download("punkt")


def preprocess_text(text: str) -> str:
    """
    Function that takes a text as a string, removes stopwords and punctuation, and applies stemming.

    Args:
        text (str): Text to be preprocessed.

    Returns:
        result (str): Preprocessed text.
    """
    
    # Tokenize the text
    text = word_tokenize(text)

    # Remove stopwords
    text = [word for word in text if word.lower() not in stopwords.words("english")]

    # Remove punctuation
    text = [word for word in text if word not in string.punctuation]

    # Apply stemming
    text = [PorterStemmer().stem(word) for word in text]
    
    # Keep words that contain only alphabetical letters
    text = [word for word in text if word.isalpha()]

    # Preprocessed text
    processed_text = " ".join(text)
    
    return processed_text


def get_fee(fee_text: str) -> float:
    """
    Function that takes a string and extracts the fee in EUR.

    Args:
        fee_text (str): Text that includes the fee information.

    Returns:
        fee (float): Returns fee in EUR.
    """
    
    # Define exchange rates to EUR
    exchange_rates = {
                      "sek": 0.08588,
                      "chf": 1.03708,
                      "usd": 0.93600,
                      "gbp": 1.14430,
                      "rmb": 0.12892,
                      "jpy": 0.00618,
                      "qr": 0.25672
                     }

    # Retrieve all fees
    fees = re.sub(r"[.,]", "", fee_text)
    fees = re.findall(r"\d+", fees)
    fees = [int(fee) for fee in fees]
    
    # Remove all fees with is actually a year and values smaller than 50
    fees = [fee for fee in fees if fee not in list(range(1, 50)) + [2021, 2022, 2023]]
    
    # Retrive the larges fee and its currency and none otherwise
    if fees:
        fee = sorted(fees, reverse=True)[0]
        currencies = re.findall(r"[\£\$\€]|eur|sek|chf|usd|gbp|rmb|jpy|qr", fee_text.lower(), flags=re.IGNORECASE)
        currency = list(set([currency.replace("€", "eur").replace("£", "gbp").replace("$", "usd") for currency in currencies]))
        
        # Special case where no currency is provided but by checking it is GBP
        if len(currency) == 0 and "UK Fees: " in fee_text:
            currency = ["GBP"]
            
        # Remove the cases where we have fees with multiple currencies otherwise retrieve the fee in EUR
        if len(currency) > 1 or len(currency) == 0:
            fee = None
            currency = None
        else:
            currency = currency[0]
            fee = "{:.2f}".format(fee*exchange_rates.get(currency, 1.0), 2)
    else:
        fee = None
        currency = None
    
    return fee


def is_valid_link(link: str) -> bool:
    """
    Function that takes a string and checks if it is a valid url link.

    Args:
        link (str): Text to be checked.

    Returns:
        is_link (bool): Returns True or False.
    """
    
    try:
        result = urlparse(link)
        is_link = all([result.scheme, result.netloc])
    except ValueError:
        is_link = False
    
    return is_link
    

if __name__ == "__main__":
    
    # Define a list with keywords to exclude certain values inside the fees value
    no_fees_keywords = [
                        "find out about",
                        "find out fees",
                        "questions about fees",
                        "view your course fees",
                        "further information on fees",
                        "view course fees",
                        "more about course fees",
                        "find out more about fees",
                        "the engineering business management online msc curriculum includes 12 units, ",
                        "abdn.ac.uk/study/international/requirements-pg-266.php"
                       ]

    # Iterate through every course HTML file
    for root, dirs, files in os.walk("courses"):
        for file in files:

            # Read the current course HTML file into a pandas dataframe
            df = pd.read_csv(f"courses/{file}", sep="\t", encoding="utf-8")

            # Check if dataframe is empty
            if not df.empty: 
                
                df = df.drop("fees (€)", axis=1)
                df = df.drop("preprocessed_description", axis=1)

                # Create a new column "fees (€)"
                df["fees (€)"] = ""

                # Create a new column preprocessed_description
                df["preprocessed_description"] = ""

                # Retrieve the data of the fees field
                fees = df["fees"].values[0]

                # Retrive the data of the description field
                description = df["description"].values[0]

                # Check if the field of fees is not empty
                if pd.notna(fees):

                    # Check if the field has some special cases defined by keywords
                    if all(string not in fees.lower() for string in no_fees_keywords) and not is_valid_link(fees):

                        # Preprocess the fees field
                        fee = get_fee(fees)

                        # Check if a fee was retrieved
                        if fee:
                            df["fees (€)"] = fee

                # Check if the field of description is not empty
                if pd.notna(description):
                    df["preprocessed_description"] = preprocess_text(description)

                # Create an updated course file with the new column
                df.to_csv(f"courses/{file}", sep="\t", index=False)
