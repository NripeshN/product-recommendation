import pandas as pd

def preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    # Convert InvoiceDate to datetime format
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    
    # Drop rows with missing CustomerID
    data = data.dropna(subset=['CustomerID'])
    
    # Converting CustomerID to integer
    data['CustomerID'] = data['CustomerID'].astype(int)
    
    # Save the cleaned data
    data.to_csv('cleaned_data.csv', index=False)

if __name__ == "__main__":
    preprocess_data('OnlineRetail.csv')