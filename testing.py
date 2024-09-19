import math
import requests
import pandas as pd
from io import StringIO

def squareNumber(num): 
    return num * num

def solve_quadratic(a,b,c):
    if (b**2 - 4*a*c < 0):
        return [0,0]
    else:
        return [(math.sqrt(b**2 - 4*a*c) - b) / (2*a), -(math.sqrt(b**2 - 4*a*c) + b) / (2*a)]


#print(solve_quadratic(1, 0, -4))

def last_closing_price(webpage):
    dataFrames = pd.read_html(webpage)
    return dataFrames[0]

#print(last_closing_price('https://finance.yahoo.com/quote/GOOG'))

def countryCapitol(countryName):
    webpage2 = "https://www.boldtuesday.com/pages/alphabetical-list-of-all-countries-and-capitals-shown-on-list-of-countries-poster"

    # Needed to get html from HTTPS since security layer prevents get_html from reading to data frame
    r = requests.get(webpage2)
    html = StringIO(r.text)
    dataFrame = pd.read_html(html)
    
    dfNumber = 0
    if countryName[0] == 'A':
        dfNumber = 0
    elif countryName[0] == 'B':
        dfNumber = 1
    elif countryName[0] == 'C':
        dfNumber = 2
    elif countryName[0] == 'D':
        dfNumber = 3
    elif countryName[0] == 'E':
        dfNumber = 4
    elif countryName[0] == 'F':
        dfNumber = 5
    elif countryName[0] == 'G':
        dfNumber = 6
    elif countryName[0] == 'H':
        dfNumber = 7
    elif countryName[0] == 'I':
        dfNumber = 8
    elif countryName[0] == 'J':
        dfNumber = 9
    elif countryName[0] == 'K':
        dfNumber = 10
    elif countryName[0] == 'L':
        dfNumber = 11
    elif countryName[0] == 'M':
        dfNumber = 12
    elif countryName[0] == 'N':
        dfNumber = 13
    elif countryName[0] == 'O':
        dfNumber = 14
    elif countryName[0] == 'P':
        dfNumber = 15
    elif countryName[0] == 'Q':
        dfNumber = 16
    elif countryName[0] == 'R':
        dfNumber = 17
    elif countryName[0] == 'S':
        dfNumber = 18
    elif countryName[0] == 'T':
        dfNumber = 19
    elif countryName[0] == 'U':
        dfNumber = 20
    elif countryName[0] == 'V':
        dfNumber = 21
    elif countryName[0] == 'W':
        dfNumber = 22
    elif countryName[0] == 'X':
        dfNumber = 23
    elif countryName[0] == 'Y':
        dfNumber = 24
    elif countryName[0] == 'Z':
        dfNumber = 25
    else:
        return -1


    input_col = dataFrame[dfNumber][0]
    output_col = dataFrame[dfNumber][1]
    dictionary = dict( zip( input_col, output_col ) )

    print(dictionary[countryName])
    return 0

#countryCapitol('BURUNDI')

dataFrame = pd.read_csv('practice-project-dataset-1.csv')
keepArray = ['loan_amount', 'interest_rate', 'property_value', 'state_code', 'tract_minority_population_percent', 'derived_race', 'derived_sex','applicant_age']

dataFrame = dataFrame.loc[:, keepArray]

#print(dataFrame[0].info())

#dataFrame = dataFrame.astype({'loan_amount' : 'int64'})
dataFrame = dataFrame.astype({'interest_rate' : 'string'}, errors='ignore')
dataFrame = dataFrame.astype({'property_value' : 'float64'}, errors='ignore')
dataFrame = dataFrame.astype({'state_code' : 'string'})
#dataFrame = dataFrame.astype({'tract_minority_population_percent' : 'float64'})
dataFrame = dataFrame.astype({'derived_race' : 'string'})
dataFrame = dataFrame.astype({'derived_sex' : 'string'})
dataFrame = dataFrame.astype({'applicant_age' : 'int64'}, errors='ignore')

female_applicatns = dataFrame.loc[dataFrame['derived_sex'] == 'Female']
female_applicatns = female_applicatns.loc[female_applicatns['interest_rate'] != '<NA>']
female_applicatns = female_applicatns.loc[female_applicatns['interest_rate'] != 'Exempt']
female_applicatns = female_applicatns.loc[female_applicatns['property_value'] != 'NaN']
female_applicatns = female_applicatns.astype({'interest_rate' : 'float64'})
female_applicatns = female_applicatns.astype({'property_value' : 'float64'})
asian_applicants = dataFrame.loc[dataFrame['derived_race'] == 'Asian']
applicants_75_and_older = dataFrame.loc[dataFrame['applicant_age'] == ('>74')]


print(female_applicatns)
#print(asian_applicants)
#print(applicants_75_and_older)

female_applicatns['interest_first_year'] = female_applicatns['property_value'] * female_applicatns['interest_rate'] / 100

print(female_applicatns)


#print(dataFrame.info())
