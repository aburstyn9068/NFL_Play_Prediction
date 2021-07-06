import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import time, timedelta
import streamlit as st

# Upload the data only when first opening the app
flag = False
if flag == False:
    plays = pd.read_csv("NFL_2017_Plays.csv")

    # Get a list of the teams
    team_list = plays["posteam"].sort_values().unique()

    flag = True

# Create interactive dashboard

# Title
st.write("""
# Football Play Prediction for the 2017 NFL Season
This app can be used by a Defensive Coordinator to predict the play an opponent is likely to
run in a certain game situation.\n
Alternatively, this app can be used by an Offensive Coordinator to see what play the opposing team
may be expecting them to run.\n
Use the fields on the sidebar on the left side of the page to input the team you are playing against 
and the game situation. The predicted play will be displayed at the bottom of the page.
""")

# Get user input values
def user_input():
    #season = int(st.sidebar.slider("Distance to endzone", 2009, 2018, 2009))
    team = st.sidebar.selectbox("Offensive Team", team_list)
    endzone_distance = st.sidebar.slider("Yards to endzone", 0, 100, 50)
    down = st.sidebar.number_input("Down", 1, 4, 1)
    if endzone_distance < 10:
        distance = st.sidebar.number_input("Distance needed for 1st down", 0, endzone_distance, endzone_distance)
    else:
        distance = st.sidebar.number_input("Distance needed for 1st down", 0, endzone_distance, 10)
    quarter = st.sidebar.slider("Quarter", 1, 4, 1)
    game_time_remaining = st.sidebar.slider("Time remaining in the quarter", time(0,0,0), time(0,15,0), time(0,15,0), timedelta(seconds=1), "mm:ss")
    time_q = (4-quarter)*15*60
    time_min = int(game_time_remaining.minute) * 60
    time_sec = int(game_time_remaining.second)
    game_seconds_remaining = time_q + time_min + time_sec
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
    posteam_timeouts_remaining = st.sidebar.slider("Offense TOL", 0, 3, 3)
    defteam_timeouts_remaining = st.sidebar.slider("Defense TOL", 0, 3, 3)

    # Store and reutrn input
    input_data = {
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

# Look at plays for a single team
team = play_input["team"].values[0]
team_plays = plays.loc[plays["posteam"]==team]

# Show data
st.subheader("Previous Games Play Data:")
st.dataframe(team_plays)

# Split data into independent and dependent variables
X = team_plays.iloc[:,4:13].values
y = team_plays.iloc[:,13].values

# Split the data into train and sets sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the random forrest model
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Check accuracy
st.subheader(f"Prediction model accuracy for {team}")
st.write(str(round(accuracy_score(y_test, y_pred)*100,2))+"%")

# Display the input
st.subheader("Game situation to predict:")
st.write(play_input)

# Get play prediction
X_predict = play_input.iloc[:,1:].values
play_prediction = classifier.predict(X_predict)

# Display predicted play
st.subheader("Predicted Play: ")
st.write(play_prediction[0].title().replace("_", " "))
