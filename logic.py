import pandas as pd
def import pandas as pd

def get_mock_data():
    # Mock Farmers: ID, Name, Land Size (in Acres)
    farmers_data = {
        'FarmerID': [101, 102, 103, 104, 105],
        'Name': ['Ramesh', 'Suresh', 'Mahesh', 'Ganesh', 'Dinesh'],
        'LandSize_Acres': [2.0, 5.0, 1.0, 10.0, 0.5] # 103 and 105 have small land
    }
    
    # Mock Transactions: Who bought what, from whom, and how much
    transactions_data = {
        'TransactionID': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'FarmerID': [101, 102, 103, 104, 105], # Matches farmers above
        'DealerID': ['D1', 'D1', 'D2', 'D2', 'D1'],
        'Fertilizer_Type': ['Urea', 'Urea', 'Urea', 'DAP', 'Urea'],
        'Quantity_KG': [50, 100, 500, 200, 45] # Look at T3 (500kg for 1 acre!)
    }
    
    return pd.DataFrame(farmers_data), pd.DataFrame(transactions_data)



def detect_land_mismatch(farmers_df, transactions_df):
    """
    Finds farmers buying more fertilizer than their land needs.
    Rule: Max 100kg allowed per Acre.
    """
    # 1. Merge tables so we know the Land Size for every transaction
    merged_df = pd.merge(transactions_df, farmers_df, on='FarmerID')
    
    # 2. Define the Rule (e.g., 4 bags (180kg) per acre is the max limit)
    limit_per_acre = 100 
    merged_df['Max_Allowed_KG'] = merged_df['LandSize_Acres'] * limit_per_acre
    
    # 3. Filter: Find where Quantity > Max_Allowed
    suspicious_df = merged_df[merged_df['Quantity_KG'] > merged_df['Max_Allowed_KG']]
    
    # Return specific columns to show in the UI
    return suspicious_df[['TransactionID', 'Name', 'LandSize_Acres', 'Quantity_KG', 'Max_Allowed_KG']]

def detect_dealer_volume(transactions_df):
    """
    Finds dealers selling abnormally high amounts.
    Rule: Any dealer with > 2 transactions (for this tiny dataset) is flagged.
    """
    # 1. Count transactions per Dealer
    dealer_counts = transactions_df['DealerID'].value_counts().reset_index()
    dealer_counts.columns = ['DealerID', 'Transaction_Count']
    
    # 2. Filter: Find dealers with too many sales
    # In a real hackathon, set this threshold to 100 or 1000
    threshold = 2 
    suspicious_dealers = dealer_counts[dealer_counts['Transaction_Count'] > threshold]
    
    return suspicious_dealers

if __name__ == "__main__":
    print("--- Running Logic Test ---")
    
    # Get fake data
    df_farmers, df_trans = get_mock_data()
    
    # Test Function 1
    print("\nSuspicious Farmers (Over-buying):")
    print(detect_land_mismatch(df_farmers, df_trans))
    
    # Test Function 2
    print("\nSuspicious Dealers (High Volume):")
    print(detect_dealer_volume(df_trans))