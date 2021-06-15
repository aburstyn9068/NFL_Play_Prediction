import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st

# Upload the data only when first opening the app
flag = False
if flag == False:
    #plays = pd.read_csv("/Users/adamburstyn/Documents/Football Data/Play_Prediction/NFL_2017_Plays.csv")
    plays = pd.read_csv("NFL_2017_Plays.csv")


    # # Remove unncesary columns
    # plays = plays_all[["game_id", "game_date", "posteam", "defteam", "yardline_100", 
    #                     "game_seconds_remaining", "down", "ydstogo", "shotgun", "no_huddle",
    #                     "two_point_attempt", "score_differential_post", 
    #                     "posteam_timeouts_remaining", "defteam_timeouts_remaining", "play_type"]]

    # # Remove 2-point conversion plays as these should be analyzed seperately
    # plays = plays.loc[plays["two_point_attempt"]!=1]
    # plays = plays.drop(columns=["two_point_attempt"])

    # # Keep only remaining run, pass, field goal, and punt plays
    # plays = plays.loc[(plays["play_type"]=="run") | (plays["play_type"]=="pass") | 
    #                 (plays["play_type"]=="punt") | (plays["play_type"]=="field_goal")]

    # Get a list of the teams
    team_list = plays["posteam"].sort_values().unique()

    flag = True

# Create interactive dashboard

# Title
st.write("""
# Football Play Prediction for the 2017 Season
Predict a teams offensive play given the game situation
""")

# Get user input values
def user_input():
    #season = int(st.sidebar.slider("Distance to endzone", 2009, 2018, 2009))
    team = st.sidebar.selectbox("Pick the team on offense", team_list)
    endzone_distance = st.sidebar.slider("Distance to endzone", 0, 100, 50)
    game_seconds_remaining = st.sidebar.slider("Game time remaining (s)", 0, 3600, 3600)
    down = st.sidebar.slider("Down", 1, 4, 1)
    distance = st.sidebar.number_input("Distance to 1st down", 0, 100, 10)
    formation = st.sidebar.selectbox("Formation", ("Under Center", "Shotgun", "Special Teams"))
    if formation == "Shotgun":
        shotgun = 1
    else:
        shotgun = 0
    huddle = st.sidebar.selectbox("Huddle/Hurry-Up", ("Huddle", "Hurry-Up"))
    if huddle == "Huddle":
        no_huddle = 0
    else:
        no_huddle = 1
    score_differential_post = st.sidebar.number_input("Offensive team score differential (- if losing)", -100, 100, 0)
    posteam_timeouts_remaining = st.sidebar.selectbox("Offense TOL", (3, 2, 1, 0))
    defteam_timeouts_remaining = st.sidebar.selectbox("Defense TOL", (3, 2, 1, 0))

    # Store and reutrn input
    input_data = {
        #"season": season,
        "team": team,
        "endzone_distance": endzone_distance,
        "game_seconds_remaining": game_seconds_remaining,
        "down": down,
        "distance": distance,
        "formation": shotgun,
        "no_huddle": no_huddle,
        "score_differential_post": score_differential_post,
        "posteam_timeouts_remaining": posteam_timeouts_remaining,
        "defteam_timeouts_remaining": defteam_timeouts_remaining
    }
    features = pd.DataFrame(input_data, index=[0])
    return features

play_input = user_input()

# # Look at only games for a single season
# season = play_input["season"].values[0]
# season_plays = plays.loc[(plays["game_date"]>f"{season}-06-01") & (plays["game_date"]<f"{season+1}-06-01")]

# Look at plays for a single team
team = play_input["team"].values[0]
team_plays = plays.loc[plays["posteam"]==team]

# Show data
st.subheader("Previous Plays Data:")
st.dataframe(team_plays)

# Split data into independent and dependent variables
X = team_plays.iloc[:,4:13].values
y = team_plays.iloc[:,13].values

# Split the data into train and sets sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the independent variables
sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.fit_transform(X_test)

# Create and train the random forrest model
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Check accuracy
st.subheader("Prediction Model Accuracy")
st.write(str(round(accuracy_score(y_test, y_pred)*100,2))+"%")

# Display the input
st.subheader("Current game information:")
st.write(play_input)

# Get play prediction
###### Need the proper transformation ##########
X_predict = play_input.iloc[:,1:].values
#X_predict = sc.fit_transform(X_predict)
play_prediction = classifier.predict(X_predict)

# Display predicted play
st.subheader("Predicted Play: ")
st.write(play_prediction[0].title().replace("_", " "))