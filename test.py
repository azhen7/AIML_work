from  periodictable import elements
import csv
import os
import pandas as pd
import re
import numpy as np

#for ele in elements:
#    print(ele.symbol)
def is_nan(x):
    return (x != x)


def create_template():
    num_elem = len(elements._element)
    element_list = list(range(0, num_elem))
    value_list = [0.0]*num_elem
    data_template = list(zip(element_list, value_list))
    res = np.array(data_template)
    return res

dataset = []


def split_elename_and_value(row_data):
    """input name with shape into name and shape."""
  
    """
        MATCHING NUMBER:
        1. matches first digit ('\d')
         - \d matches all digits
        2. if present, matches decimal point ('\.?')
         - \. matches '.'
         - '?' means 'match 1-or-0 times'
        3. matches subsequent decimal digits ('\d*')
         - \d matches all digits
         - '*' means 'match >=0 times'
        
        MATCHING ELEMENT NAME:
        1. matches letters
         - [a-z A-Z] matches all letters
         - + matches 1-or-more times
    """
    name_pattern = r"\d\.?\d*|[a-z A-Z]+"
    splits = re.findall(name_pattern, row_data)
 
    # splits is to in format of pair of element name, value (all in strng)
    # the number of pairs varies 
    num_pairs = len(splits)//2

    print(elements.symbol)

    for index in range(num_pairs):
        ele_sym = splits[index*2]
     
        try:
            ele = elements.symbol(ele_sym)
        except ValueError as msg:
            assert str(msg) == "unknown element:" + ele_sym 

        print(ele.number)
    return 


def parse_file(file, sheet):
    data = pd.read_excel(open(file, 'rb'), sheet_name=sheet)

    for index, row in data.iterrows():
        print(row.Compound)

        if(is_nan(row.Compound)==False):
            split_elename_and_value(row.Compound)

print(os.getcwd())

create_template()

parse_file('./AIML_work/superconductor.xlsx', 'Sheet1')
