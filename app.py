import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd

# Load model
with open("banglore_home_prices_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoder
with open("location_encoder.pkl", "rb") as f:
    location_encoder = pickle.load(f)

# Get all unique locations from OneHotEncoder
locations = sorted(location_encoder.categories_[0])  # Correct for OneHotEncoder


# Load columns
with open("columns.json", "r") as f:
    columns = json.load(f)["data_columns"]

locations = sorted(location_encoder.categories_[0])  # categories_ returns [array([...])]

# Load transformer
with open("column_transformer.pkl", "rb") as f:
    ct = pickle.load(f)


# Streamlit UI
st.title("🏠 Bengaluru House Price Prediction")

st.write("Enter the details below to predict house price:")

location = st.selectbox("Select Location", locations)
sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, step=50)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
bhk = st.number_input("Number of BHK", min_value=1, max_value=10, step=1)

# if st.button("Predict Price"):
#     # One-hot encode the location
#     loc_encoded = location_encoder.transform([[location]]).toarray()

#     # Combine sqft, bath, bhk with location vector
#     input_data = np.array([np.concatenate(([sqft, bath, bhk], loc_encoded[0]))])

#     # Prediction
#     prediction = model.predict(input_data)[0]
#     st.success(f"Estimated Price: ₹ {prediction:.2f} Lakhs")


# import pandas as pd

if st.button("Predict Price"):
    # Create dataframe like training
    new_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=columns[:4])

    # Transform with ColumnTransformer
    transformed_data = ct.transform(new_data)

    # Predict
    prediction = model.predict(transformed_data)[0]
    st.success(f"Estimated Price: ₹ {prediction:.2f} Lakhs")
