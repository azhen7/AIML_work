import pandas as pd;

def parse_csv(filename):
    parsed = pd.read_csv(filename)
    
    fail = parsed.query('Tc == 0')
    succeed = parsed.query('Tc != 0')
    
    with pd.ExcelWriter('seperated_results.xlsx') as writer:
        fail.to_excel(writer, sheet_name="Fail")
        succeed.to_excel(writer, sheet_name="Success")
    
parse_csv("superconduct_2.csv")