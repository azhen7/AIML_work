import pandas as pd;

def parse_csv(filename):
    parsed = pd.read_csv(filename)
    
    a = parsed.query('Tc == 0')
    
    a.insert(1, "Tc", 0, True)
    
    a.to_excel('fails.xlsx')

parse_csv("superconduct_2.csv")