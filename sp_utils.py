import pandas as pd
import numpy as np
import os
import zipfile
from himl import ConfigProcessor
import yaml
from sklearn.model_selection import train_test_split


def yaml_to_dict(config_file):
    with open(config_file, 'r') as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    for key_ in cfg:
        try:
            key_, value_ = key_, cfg[key_].format(**cfg)
            cfg[key_] = value_
        except Exception as e:
            type(e)  # to avoid flake8 error
            key_, value_ = key_, cfg[key_]
    return (cfg)


def load_flat_file(filepath, format_cols=True, encoding='utf-8',**kwargs):
    """

    To read the flat files

    Parameters
    ----------
    filepath : str
        input file path
    format_cols : bool, optional
        True for modifying the column names, by default True
    encoding : str, optional
        by default 'utf-8'

    Returns
    -------
    pd.DataFrame
        Data frame with formated column names
    """
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath,encoding=encoding, **kwargs)
    if filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath, **kwargs)
    if filepath.endswith('.sas7bdat'):
        df = pd.read_sas(filepath,encoding=encoding,format='sas7bdat',**kwargs)
    
    if format_cols:
        df.columns = clean_col_names(df.columns,remove_chars_in_braces=False)
    return df


def clean_col_names(string_series, special_chars_to_keep="_", remove_chars_in_braces=True, strip=True, lower=True):
    
    """
    Function to clean strings.

    Removes special characters, multiple spaces

    Parameters
    ----------
        string_series : pd.Series
        special_chars_to_keep : 
            string having special characters that have to be kept
        remove_chars_in_braces: 
            Logical if to keep strings in braces. e.g: "Packages (8oz)" will be "Packages"
        strip : True(default), 
            if False it will not remove extra/leading/tailing spaces
        lower : False(default), 
            if True it will convert all characters to lowercase

    Returns
    -------
        pandas series
    """
    # FIXME: Handle Key Error Runtime Exception
    try:
        if(lower):
            # Convert names to uppercase
            string_series = string_series.str.upper()
        if(remove_chars_in_braces):
            # Remove characters between square and round braces
            string_series = string_series.str.replace(r"\(.*\)|\[.*\]", '')
        else:
            # Add braces to special character list, so that they will not be
            # removed further
            special_chars_to_keep = special_chars_to_keep + "()[]"
        if(special_chars_to_keep):
            # Keep only alphanumeric character and some special
            # characters(.,_-&)
            reg_str = "[^\\w"+"\\".join(list(special_chars_to_keep))+" ]"
            string_series = string_series.str.replace(reg_str, '', regex=True)
        if (strip):
            # Remove multiple spaces
            string_series = string_series.str.replace(r'\s+', ' ', regex=True)
            # Remove leading and trailing spaces
            string_series = string_series.str.strip()
        string_series = string_series.str.replace(' ', '_')
        string_series = string_series.str.replace('_+', '_', regex=True)
        return(string_series)
    except AttributeError:
        print("Variable datatype is not string")
    except KeyError:
        print("Variable name mismatch")


def create_out_dir(filepath):
    os.makedirs(filepath, exist_ok=True)



