import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn import cross_validation

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

csv_file_object = csv.reader(open('train.csv', 'rb')) #Load in the training csv file
header = csv_file_object.next() #Skip the fist line as it is a header
train_data=[] #Create a variable called 'train_data'
for row in csv_file_object: #Skip through each row in the csv file
    train_data.append(row) #adding each row to the data variable
train_data = np.array(train_data) #Then convert from a list to an array

#I need to convert all strings to integer classifiers:
#Male = 1, female = 0:
train_data[train_data[0::,3]=='male',3] = 1
train_data[train_data[0::,3]=='female',3] = 0
#embark c=0, s=1, q=2
train_data[train_data[0::,10] =='C',10] = 0
train_data[train_data[0::,10] =='S',10] = 1
train_data[train_data[0::,10] =='Q',10] = 2

#I need to fill in the gaps of the data and make it complete.
#So where there is no price, I will assume price on median of that class
#Where there is no age I will give median of all ages

#All the ages with no data make the median of the data
train_data[train_data[0::,4] == '',4] = np.median(train_data[train_data[0::,4]\
                                           != '',4].astype(np.float))
#All missing ebmbarks just make them embark from most common place
train_data[train_data[0::,10] == '',10] = np.round(np.mean(train_data[train_data[0::,10]\
                                                   != '',10].astype(np.float)))
#Add three columns for new data
train_data = np.append(train_data, np.zeros((train_data.shape[0],10)),1)

for row in train_data:
  name = row[2].lower()
  #Name has parenthesis
  row[11] = string_find_code(name, [" ("])
  #Name has mr., miss., mrs., master
  row[12] = string_find_code(name, ["miss.", "mrs.", "mr.", "master"])
  #Name length
  row[13] = name_length(name)

train_data[train_data[0::,10] =='C',14] = 1
train_data[train_data[0::,10] =='S',15] = 1
train_data[train_data[0::,10] =='Q',16] = 1
train_data[train_data[0::,12] ==1.0,17] = 1
train_data[train_data[0::,12] ==2.0,18] = 1
train_data[train_data[0::,12] ==3.0,19] = 1
train_data[train_data[0::,12] ==4.0,20] = 1

train_data = np.delete(train_data,[2,7,9],1) #remove the name data, cabin and ticket
#I need to do the same with the test data now so that the columns are in the same
#as the training data
train_data = np.delete(train_data,[7,9],1) #remove the port and salutation

test_file_object = csv.reader(open('test.csv', 'rb')) #Load in the test csv file
header = test_file_object.next() #Skip the fist line as it is a header
test_data=[] #Creat a variable called 'test_data'
for row in test_file_object: #Skip through each row in the csv file
    test_data.append(row) #adding each row to the data variable
test_data = np.array(test_data) #Then convert from a list to an array

#I need to convert all strings to integer classifiers:
#Male = 1, female = 0:
test_data[test_data[0::,2]=='male',2] = 1
test_data[test_data[0::,2]=='female',2] = 0
#ebark c=0, s=1, q=2
test_data[test_data[0::,9] =='C',9] = 0 #Note this is not ideal, in more complex 3 is not 3 tmes better than 1 than 2 is 2 times better than 1
test_data[test_data[0::,9] =='S',9] = 1
test_data[test_data[0::,9] =='Q',9] = 2

#All the ages with no data make the median of the data
test_data[test_data[0::,3] == '',3] = np.median(test_data[test_data[0::,3]\
                                           != '',3].astype(np.float))
#All missing ebmbarks just make them embark from most common place
test_data[test_data[0::,9] == '',9] = np.round(np.mean(test_data[test_data[0::,9]\
                                                   != '',9].astype(np.float)))
#All the missing prices assume median of their respectice class
for i in xrange(np.size(test_data[0::,0])):
    if test_data[i,7] == '':
        test_data[i,7] = np.median(test_data[(test_data[0::,7] != '') &\
                                             (test_data[0::,0] == test_data[i,0])\
            ,7].astype(np.float))

#Add three columns for new data
test_data = np.append(test_data, np.zeros((test_data.shape[0],10)),1)

for row in test_data:
  name = row[1].lower()
  #Name has parenthesis
  row[10] = string_find_code(name, [" ("])
  #Name has mr., miss., mrs., master
  row[11] = string_find_code(name, ["miss.", "mrs.", "mr.", "master"])
  #Name length
  row[12] = name_length(name)

test_data[test_data[0::,9] =='C',13] = 1
test_data[test_data[0::,9] =='S',14] = 1
test_data[test_data[0::,9] =='Q',15] = 1
test_data[test_data[0::,11] ==1.0,16] = 1
test_data[test_data[0::,11] ==2.0,17] = 1
test_data[test_data[0::,11] ==3.0,18] = 1
test_data[test_data[0::,11] ==4.0,19] = 1

test_data = np.delete(test_data,[1,6,8],1) #remove the name data, cabin and ticket
test_data = np.delete(test_data,[6,8],1) #remove the port and salutation
print test_data

train_data = train_data.astype(np.float)
test_data = test_data.astype(np.float)

"""
#Delete new columns
train_data = np.delete(train_data,[11,12,13],1) #remove the name data, cabin and ticket
test_data = np.delete(test_data,[10,11,12],1) #remove the name data, cabin and ticket
"""
print train_data
#The data is now ready to go. So lets train then test!
print 'Training'
forest = RandomForestClassifier(n_estimators=10,  min_samples_split=3, \
  min_samples_leaf=2, compute_importances=True, n_jobs=-1)
forest = forest.fit(train_data[0::,1::], train_data[0::,0])

extra_forest = ExtraTreesClassifier(n_estimators=10, max_depth=None, \
  min_samples_split=3, min_samples_leaf=2, compute_importances=True, n_jobs=-1)


print "Scoring"
scores = cross_validation.cross_val_score(forest, train_data[0::,1::], train_data[0::,0], cv=10)
print "RF Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
extra_scores = cross_validation.cross_val_score(extra_forest, train_data[0::,1::], train_data[0::,0], cv=10)
print "EF Accuracy: %0.2f (+/- %0.2f)" % (extra_scores.mean(), extra_scores.std() / 2)


"""
print 'Predicting'
output = forest.predict(test_data)
test_file_object = csv.reader(open('test.csv', 'rU')) #Load in the csv file


print 'Outputing'
open_file_object = csv.writer(open("myfirstforest.csv", "wb"))
test_file_object.next()
i = 0
for row in test_file_object:
    row.insert(0,output[i].astype(np.uint8))
    open_file_object.writerow(row)
    i += 1
"""