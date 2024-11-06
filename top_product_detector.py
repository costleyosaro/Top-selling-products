import streamlit as st
import pandas as pd
import joblib  # For loading the saved model
from sklearn.preprocessing import OneHotEncoder

# Load the trained model and encoder
model = joblib.load("best_seller_model.pkl")         # Path to your trained model
encoder = joblib.load("encoder.pkl")                 # Path to your saved OneHotEncoder for categorical features

# Streamlit app title and description
st.title("Best Seller Prediction Model")
st.write("Upload a dataset with 'Category Name', 'Item Name', 'Quantity', and 'Brand Name' columns to predict Best Seller status.")

# File uploader for CSV file input
uploaded_file = st.file_uploader("Upload CSV", type="csv")

# Function to preprocess input data
def preprocess_input(df):
    # Ensure columns are named correctly
    df.columns = [col.strip().title() for col in df.columns]  # Normalize column names
    
    # Handle missing columns if any
    required_columns = ["Category Name", "Item Name", "Quantity", "Brand Name"]
    for column in required_columns:
        if column not in df.columns:
            st.error(f"Please upload a file with '{column}' column.")
            return None
    
    # Use the encoder to transform categorical data
    category_encoded = encoder.transform(df[['Category Name', 'Item Name']])
    category_df = pd.DataFrame(category_encoded, columns=encoder.get_feature_names_out(['Category Name', 'Item Name']))
    
    # Combine encoded categories with quantity and retain original columns
    df_processed = pd.concat([category_df, df[['Quantity', 'Brand Name']].reset_index(drop=True)], axis=1)
    
    return df_processed, df[['Category Name', 'Item Name', 'Brand Name']]  # Return original info for best sellers

# Predict button and processing
if uploaded_file is not None:
    # Read CSV file
    input_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(input_df.head())

    # Preprocess the data
    processed_data, original_data = preprocess_input(input_df)
    if processed_data is not None:
        # Make predictions
        predictions = model.predict(processed_data)
        
        # Add predictions to the original DataFrame
        input_df['Best Seller'] = predictions
        
        # Filter for best sellers and sort by quantity
        best_sellers = input_df[input_df['Best Seller'] == 1]
        best_sellers_sorted = best_sellers.sort_values(by='Quantity', ascending=False)
        
        # Include original Item Name, Category Name, and Brand Name in the output
        best_sellers_final = best_sellers_sorted[['Item Name', 'Category Name', 'Brand Name', 'Quantity', 'Best Seller']]
        
        # Display the ranked best sellers
        st.write("Best Sellers Ranked by Quantity:")
        st.write(best_sellers_final)
        
        # Optional: download the best seller predictions as a CSV
        csv_output = best_sellers_final.to_csv(index=False)
        st.download_button("Download Best Sellers as CSV", data=csv_output, file_name="best_sellers_predictions.csv", mime="text/csv")
