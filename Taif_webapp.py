import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st

# (Assuming you already have your 'data' DataFrame and label_encoders from before)
np.random.seed(42)
n = 300
data = pd.DataFrame({
    'Academic_Performance': np.random.randint(50, 100, n),
    'Motivation_Level': np.random.choice(['Low', 'Medium', 'High'], n),
    'Extracurricular_Participation': np.random.choice(['None', 'Some', 'Many'], n),
    'Peer_Influence': np.random.choice(['Positive', 'Neutral', 'Negative'], n),
    'Confidence_Level': np.random.choice(['Low', 'Medium', 'High'], n),
    'Previous_Mentorship_Experience': np.random.choice(['Yes', 'No'], n),
    'Mentorship_Type': np.random.choice(['Academic', 'Emotional', 'Career'], n)  # Target
})

label_encoders = {}
for col in ['Motivation_Level', 'Extracurricular_Participation', 'Peer_Influence', 
            'Confidence_Level', 'Previous_Mentorship_Experience', 'Mentorship_Type']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Prepare features and target
X = data.drop('Mentorship_Type', axis=1)
y = data['Mentorship_Type']

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Function to take user input and predict mentorship type
def predict_mentorship():
    # Get inputs from user
    academic_performance = st.number_input("Enter Academic Performance (50-100): ", 50,100)
    motivation_level = st.selectbox("Enter Motivation Level: ", ['Low', 'Medium', 'High'])
    extracurricular_participation = st.selectbox("Enter Extracurricular Participation : ",['None', 'Some', 'Many'])
    peer_influence = st.selectbox("Enter Peer Influence: ", ['Positive', 'Neutral', 'Negative'])
    confidence_level = st.selectbox("Enter Confidence Level: ", ['Low', 'Medium', 'High'])
    previous_mentorship_experience = st.selectbox("Previous Mentorship Experience?: ", ['Yes', 'No'])

    if st.button('Run prediction'):
        # Encode categorical inputs using the label_encoders
        motivation_level_enc = label_encoders['Motivation_Level'].transform([motivation_level])[0]
        extracurricular_participation_enc = label_encoders['Extracurricular_Participation'].transform([extracurricular_participation])[0]
        peer_influence_enc = label_encoders['Peer_Influence'].transform([peer_influence])[0]
        confidence_level_enc = label_encoders['Confidence_Level'].transform([confidence_level])[0]
        previous_mentorship_experience_enc = label_encoders['Previous_Mentorship_Experience'].transform([previous_mentorship_experience])[0]

        # Create feature array for prediction
        features = np.array([[academic_performance, motivation_level_enc, extracurricular_participation_enc,
                            peer_influence_enc, confidence_level_enc, previous_mentorship_experience_enc]])

        # Predict mentorship type
        pred_encoded = model.predict(features)[0]
        pred_label = label_encoders['Mentorship_Type'].inverse_transform([pred_encoded])[0]

        st.markdown(f"\nThe recommended mentorship type for you is: {pred_label}")

# Run the prediction function
predict_mentorship()

