🏠 Bengaluru House Price Prediction

This project is a machine learning–powered web app built with Streamlit that predicts Bengaluru house prices (in Lakhs) based on location, total square feet, number of bathrooms, and BHK. It uses a pre-trained scikit-learn model along with a ColumnTransformer and location encoder saved as pickled files. 
app


🚀 Features

Interactive Streamlit UI for easy input of property details
Uses location, total_sqft, bath, bhk as features for prediction columns
Pre-trained scikit-learn regression model loaded from pickle
One-hot encoding of location using a saved location encoder
Real-time prediction of house price in ₹ Lakhs

📂 Project Structure
.
├── app.py                      # Streamlit app
├── banglore_home_prices_model.pkl
├── location_encoder.pkl
├── column_transformer.pkl
├── columns.json                # List of feature/column names
├── Bengaluru_House_Data.csv    # Original/raw dataset
├── d8_dataset.csv              # Processed/cleaned dataset (if used)
├── RealEstatePridiction.ipynb  # Model training & experimentation
└── requirements.txt            # Python dependencies

🧠 Model & Data

Training data: Bengaluru housing dataset (Bengaluru_House_Data.csv, d8_dataset.csv)
Target variable: price (in Lakhs)
Features used by the app: location, total_sqft, bath, bhk (as defined in columns.json) columns
The model, column transformer, and location encoder are pre-trained and stored as .pkl files.

🛠 Tech Stack

Python
Streamlit
NumPy, Pandas, Matplotlib, scikit-learn (from requirements.txt) 
requirements

📦 Installation

Clone this repository
git clone https://github.com/your-username/bengaluru-house-price-prediction.git
cd bengaluru-house-price-prediction


Create and activate a virtual environment (optional but recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate


Install dependencies
pip install -r requirements.txt

▶️ Run the App
streamlit run app.py


Then open the URL shown in the terminal (usually http://localhost:8501) in your browser.

💡 How to Use

Select a Location from the dropdown.
Enter Total Square Feet (e.g., 1200).
Enter Number of Bathrooms.
Enter Number of BHK.
Click "Predict Price".

The app will display the estimated price in ₹ Lakhs.

🧪 Experimenting / Retraining

Use RealEstatePridiction.ipynb to:
Explore the dataset
Clean and preprocess data
Train new models

Update and re-save:
banglore_home_prices_model.pkl
location_encoder.pkl
column_transformer.pkl
columns.json (if feature set changes)
After updating these files, restart the Streamlit app to use the new model.

📌 Future Improvements (Ideas)

Add more features (e.g., number of balconies, property age, area type)
Model performance comparison (Linear Regression, Random Forest, XGBoost, etc.)
Deploy the app to Streamlit Community Cloud, Render, or Heroku

Add APIs for programmatic access to predictions
