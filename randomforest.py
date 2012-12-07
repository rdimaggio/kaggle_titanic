import csv as csv 
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def string_find_code(item, sub):
    for i in range(len(sub)):
        if item.find(sub[i]) >= 0:
            return float(i) 
    return 0.

def name_length(name):
    try:
        index = name.index(" (")
        return float(len(name[:index-1].split(" ")))
    except:
        return float(len(name.split(" ")))

def transform_data(data):
    for row in data:
        # change sex to float
        row[3] = 1. if row[3].lower()=="female" else 0.
        
        name = row[2].lower()
        # add a paranthesis flag
        row[7] = string_find_code(name, ["("])
        # add Miss.=1 vs Mrs.=2 vs Mr.=3 flag
        row[9] = string_find_code(name, ["miss.", "mrs.", "mr.", "master"])
        # add number of names flag
        row[2] = name_length(name)
        # change embarked into float
        row[10] = string_find_code(row[10].lower(), ["s", "c", "q"])
        # change cabin to two numbers
        # fill in empty numbers
        row[4] = 0. if row[4]=="" else float(row[4])

def get_data(filename):
    csv_file_object = csv.reader(open(filename, 'rU')) 
    header = csv_file_object.next()

    data=[]
    for row in csv_file_object:
        data.append(row)
    return transform_data(np.array(data))

train_data = get_data('train.csv')
test_data = get_data('train.csv')


Forest = RandomForestClassifier(n_estimators = 100)
Forest = Forest.fit(train_data[0::,1::],train_data[0::,0])
Output = Forest.predict(test_data)

print Output