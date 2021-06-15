# Play_Prediction
Machine learning model and interactive web application to predict a football teams next play.

This repository contains the code files used to create a web application for predicting a football teams next play. 

The app can be used by a Defenseive Coordinator to predict the play an opponent is likely to run in a certain game situation. Alternatively, the app can be used by an Offensive Coordinator to see what play the opposing team may be expecting them to run.

The code is written in python and uses the Scikit-learn library to create the prediciton model, as well as the Streamlit library to create an app to interact with the data. The app is deployed using Heroku. The data used in this project came from:
https://www.kaggle.com/maxhorowitz/nflplaybyplay2009to2016?select=NFL+Play+by+Play+2009-2018+%28v5%29.csv.

The dataset was extremly large and reduced to only plays from the 2017 season using a Jupyter Notebook file. The data gets filtered by a specific team to analyze and then gets split into training and testsing sets. The independent variables used for the prediction model were:
- Distance from the endzone (yards)
- Down
- Distance needed for 1st down
- Game time remaining (seconds)
- Offensive formation (Shoutgun, Under Center, Special Teams)
- Huddle/Hurry-Up
- Offensive team score differential (- if losing)
- Offense Time Outs Left
- Defense Time Outs Left
The dependent variable the model will predict was the play (Pass, Run, Punt, or Field Goal).

From there a random forrest classifier model was created. The model gets fit with the training set and then predicts the testing set results. The accuracy of the predictions is then analyzed. The resulting code was downloaded as a .py file to be used as the web app.

The web app allows the user to select a team on offense and input the game situtation variables (independent varaiables). The app will then apply the user input to the random forrest classifier and predict what play the team will run in that situation.

The page is displayed with the user input are on a sidebar on the left side of the page. The main section of the page contains:
- Introduction and instructions
- A table of the teams previous games play data that was is used to make the play prediction
- The models prediction accuracy for the current team being analyzed
- The game situation being analyzed as input by the user
- The play prediction based on the user inputs.

The page can be viewed at: https://nfl-play-prediction.herokuapp.com/.


