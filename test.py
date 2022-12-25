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
    value_list = [0.0]*num_elem
    res = np.array(value_list)
    res = np.expand_dims(res, axis=1)
    return res

# check if string contain int or float number
# or it is a true string 
def string_or_number(s):
    try:
        z = int(s)
        return z
    except ValueError:
        try:
            z = float(s)
            return z
        except ValueError:
            return s

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
    print(elements.symbol)

    # locate all symbol positions
    symbol_pos = []
    index = 0
    while index < len(splits):
        ele_sym = splits[index]
        print(type(ele_sym))
        if type(string_or_number(ele_sym)) == str:
            try:
                ele = elements.symbol(ele_sym)
                symbol_pos.append(index)
            except ValueError as msg:
                print(str(msg))

        index = index+1

    element_id = []
    element_value = []
    total_value = 0
    for pos in symbol_pos:
        init_pos = pos
        ele = elements.symbol(splits[init_pos])
        element_id.append(ele.number)
        init_pos = init_pos+1
        if init_pos < len(splits):
            value = string_or_number(splits[init_pos])
            if type(value) == int or float:
                element_value.append(value)
                total_value = total_value + value
            else:
                element_value.append(-1)
        else:
            element_value.append(-1)

    if element_value.count(-1) > 0:
        split_value = (1-total_value)/element_value.count(-1)
        element_value = [split_value if item == -1 else item for item in element_value]
         

    return list(zip(element_id, element_value))


def parse_file(file, sheet):
    data = pd.read_excel(open(file, 'rb'), sheet_name=sheet)
    
    np_arr = np.empty([1, 119, 1])
    for index, row in data.iterrows():
        print(row.Compound)

        if(is_nan(row.Compound)==False):
            data_point = create_template()
            ret = split_elename_and_value(row.Compound)
            for item in ret:
                data_point[item[0]] = item[1]

            data_point = np.expand_dims(data_point, axis=0)
            np_arr= np.concatenate((np_arr, data_point), axis=0)

    
    return np_arr[1:, :]

print(os.getcwd())


ret = parse_file('./AIML_work/superconductor.xlsx', 'Sheet1')
print(ret[0])
