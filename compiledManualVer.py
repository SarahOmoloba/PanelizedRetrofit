# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 15:59:57 2023

@author: sarah
"""

import pandas as pd
#import numpy as np

dfOverbrook = pd.read_csv('extractedOverbrook.csv')
#df.head()
def fetch_image():
    address = dfOverbrook['FULL_ADDRESS_EN']
    api_key = "AIzaSyA5Uzg089JZInQHGg9TrP1Gz4ySqY1--k8"
    url_front = f"https://maps.googleapis.com/maps/api/streetview?size=600x300&location={address}&key={api_key}"
    #url_back = f"https://maps.googleapis.com/maps/api/streetview?size=600x300&location={address}&key={api_key}&heading=90&pitch=0"
    response_front = requests.get(url_front)
    #response_back = requests.get(url_back)
    with open("front.jpg", "wb") as f:
        f.write(response_front.content)
    #with open("back.jpg", "wb") as f:
        #f.write(response_back.content)
    image_front = Image.open("front.jpg")
    #image_back = Image.open("back.jpg")
    image_front = image_front.resize((600, 300), Image.Resampling.LANCZOS)
    #image_back = image_back.resize((600, 300), Image.Resampling.LANCZOS)
    photo_front = ImageTk.PhotoImage(image_front)
    #photo_back = ImageTk.PhotoImage(image_back)
    image_label_front.configure(image=photo_front)
    #image_label_back.configure(image=photo_back)
    image_label_front.image = photo_front
    #image_label_back.image = photo_back

def get_description():
    # implement your code to get image description here
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="ENTER YOUR PATH HERE"

    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # The name of the image file to annotate
    file_front = os.path.abspath('front.jpg')
    file_back = os.path.abspath('back.jpg')

    # Loads the image into memory
    with io.open(file_front, 'rb') as image_file:
        content_front = image_file.read()
    image_front = vision.Image(content=content_front)
    with io.open(file_back, 'rb') as image_file:
        content_back = image_file.read()
    image_back = vision.Image(content=content_back)
    
    # Performs label detection on the image file
    response_front = client.label_detection(image=image_front)
    response_back = client.label_detection(image=image_back)
    #labels = response.label_annotations

    # Get the labels detected in the image
    #label_text = 'Labels:\n'
    #for label in labels:
     #   label_text += label.description + '\n'
    #label_text += '\n'

    # Performs web detection on the image file
    response_front = client.web_detection(image=image_front)
    #response_back = client.web_detection(image=image_back)
    web_entities_front = response_front.web_detection.web_entities
    #web_entities_back= response_back.web_detection.web_entities
    # Get the web entities detected in the image
    
    web_entity_text_front = 'Image(1) "front view description":\n'
    #web_entity_text_back = 'Image(2) "back view description":\n'
    for entity in web_entities_front:
        web_entity_text_front += entity.description + '\t'
    #for entity in web_entities_back:
     #   web_entity_text_back += entity.description + '\t'

    
#enter the address
address_input = input("Please enter the address from the image search bar: ")

#use 698 Alesther St and 700 Alesther St
     
match = dfOverbrook.loc[dfOverbrook['FULL_ADDRESS_EN']== address_input]    
if not match.empty:
    print('The building information is: ', match.FULL_ADDRESS_EN.to_string(index=False, header=False),', ', match.POSTAL_CODE.to_string(index=False, header=False), ',', match.MUNICIPALITY.to_string(index=False, header=False), ', Ward',  match.WARD.values, 'and Minimum lotsize of ',  match.MINLOSIZELEFT.values)
else:
    print('No matches found.')
    
#declare all variables

isbuilding                     = 0
balcony_absent                 = 0
exterior_finish                = 0
building_age                   = 0
built_before_1990              = 0
building_height_less_than_four = 0

#subject to change
#web_entities_front = input("The descriptions are: ")
lowercase_entities = web_entities_front.lower()
lowercase_entities = web_entities_front.lower()

#list of specific words that relate 
specific_words = ["building", "home", "house", "residence", \
                  "dwelling", "domicile", "lodge", "manor", \
                  "villa", "bungalow", "cottage", "homestead", \
                  "farmhouse", "cabin", "apartment"]

#to train the model
import nltk

entities_token = nltk.word_tokenize(lowercase_entities)
#nltk.download('omw-1.4')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in entities_token]
print(lemmatized_tokens)

#check if the image returned is a building
match_found =False
#loop through each specific word and check if it's in the returned word set
for word1 in lemmatized_tokens:
    for word2 in specific_words:
        if word1 == word2:
            isbuilding = 1
            print("The image returned contains a building")
            match_found =True
            break
            
    if match_found:
        break    
else:
    print("Not a building!")
#if it is a building then do the following 
if isbuilding == 1:
    #check if balcony is absent   
    balcony_input = input("Is there a balcony? enter y for yes and n for no:")
    if balcony_input == "y":
        balcony_absent = 0
    if balcony_input == "n": 
        balcony_absent = 1
    else:
        print("Please enter y or n")
        balcony_input = input("Is there a balcony? enter y for yes and n for no:")    
    
    #checking the exterior finish of the building
    exterior_type       = ["wood","stucco","metal","vinyl","glass","brick","stone", "other"]
    print(exterior_type)
    exterior_type_input = input("Please enter the exterior finish of the building from the dispalyed list:")
    type_input          = exterior_type_input.lower()

    #checking if the exterior type meets the requirement
    if type_input in ("wood","stucco","stone", "vinyl"):
        exterior_finish = 1
    if type_input in ("metal","glass","brick", "other"):
        exterior_finish = 0    
    else:
        print("Please make sure to check your input")  
    #checking the age of the building
    
    from datetime import datetime
    #get current year
    current_year = datetime.now().year

    age_known_input   = input("Enter y for yes and n for no if the building age and or year built is known") 
    age_known         = age_known_input.lower()

    if age_known == "y":
        year_built_input = int(input("Please enter the year built: "))
        building_age = current_year - year_built_input
        if year_built_input <= 1990:
            built_before_1990 = 1
        else:
            built_before_1990 = 0
    else : 
        building_age      = 0
        built_before_1990 = 0
        print('The result might not be as accurate since the Year built is not known.')

    #check if the building is more less than four stories
    absolute_zones  = ('R1','R2','R3')
    column_to_check = dfOverbrook['RESIDENTIAL_ZONING']
    bool_column     = column_to_check == 'R4'
    if column_to_check.isin(absolute_zones).any():
        building_height_less_than_four = 1
    if bool_column.any():
        print("Ur here!")
        building_height_input = input("Is the building less than four (4) stories? y for yes and n for no: ")
        building_height       = building_height_input.lower()
    
        if building_height == "y":
            building_height_less_than_four = 1
        else:
            building_height_less_than_four = 0
    else:
        print("The zoning information does not fit into the preferred category ")
        building_height_less_than_four = 0
else:
    balcony_absent                 = 0
    exterior_finish                = 0
    building_age                   = 0
    built_before_1990              = 0
    building_height_less_than_four = 0
    
#create the input to feed into the model
model_input = [isbuilding,building_age,built_before_1990,building_height_less_than_four,balcony_absent,exterior_finish]
print(model_input)


#import matplotlib.pyplot as plt
#import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# import module
from sklearn.model_selection import train_test_split
# import Decision Tree model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# next load the data
df = pd.read_csv('ModelFourTrainingDataset.csv')

# drop all variables from the data that we won't need for the model.
df = df.drop(['buildingID', 'latitude', 'longitude', 'postalCode', 'propertyAddress','ImageClassID'], axis = 1)

# seperate input features in x
x = df.drop('suitabilityForRetrofit', axis = 1)

# store the target variable in y
y = df['suitabilityForRetrofit']


# Split the dataset
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.20, random_state=11)

# create an instane of the model
dt = DecisionTreeClassifier(max_depth = 4, max_features=6)

# train the model
dtmodel = dt.fit(xtrain,ytrain)

# Predict on x_test
y_pred = dt.predict(xtest)

# Evaluate the Model
print('Accuracy of Random Forest model on the train set: {:.2f}'.format(dt.score(xtrain, ytrain)))
print('Accuracy of Random Forest model on the test set: {:.2f}'.format(dt.score(xtest, ytest)))
confusion_matrix(ytest, y_pred)


#Save the model
import pickle

pickle.dump(dt,open('DECISIONTREE_MODEL_MANUAL','wb'))

#load the model
model4_manual = pickle.load(open('DECISIONTREE_MODEL_MANUAL','rb'))

result= model4_manual.predict([model_input])
print(result)