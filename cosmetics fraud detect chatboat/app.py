from flask import Flask, render_template, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import pandas as pd
import os
from dotenv import load_dotenv

# Load .env and API key
load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

# Load dataset
try:
    df = pd.read_excel("luxury_cosmetics_fraud_analysis_2025.xlsx")
except Exception as e:
    print("⚠️ Error loading dataset:", e)
    df = None

# Initialize Gemini model
chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)

# Store conversation memory
conversation_history = [SystemMessage(content="""You are a professional fraud detection analyst assistant. 
Your role is to analyze cosmetics transaction data and provide clear, well-structured insights about fraud patterns.

IMPORTANT GUIDELINES:
- Provide responses in a clear, professional format
- Use bullet points, numbered lists, and proper formatting
- Include specific statistics and percentages when relevant
- Highlight key patterns and anomalies
- Keep responses concise but comprehensive
- Use **bold text** for emphasis and section titles (NOT markdown headers like ## or ###)
- Use bullet points (*) and numbered lists for better readability
- Avoid using markdown headers (##, ###) - instead use **bold text** for section titles
- Avoid messy formatting or special characters like ^^
- Structure your responses with clear sections using bold text for headings""")]

@app.route("/")
def index():
    return render_template("index.html")

def get_dataset_summary():
    """Generate a structured summary of the dataset for better context"""
    if df is None:
        return "Dataset not available."
    
    # Calculate key statistics
    total_transactions = len(df)
    fraud_count = df['Fraud_Flag'].sum() if 'Fraud_Flag' in df.columns else 0
    fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
    
    # Get fraud transactions sample
    fraud_df = df[df['Fraud_Flag'] == 1] if 'Fraud_Flag' in df.columns else pd.DataFrame()
    
    summary = f"""
DATASET OVERVIEW:
- Total Transactions: {total_transactions:,}
- Fraudulent Transactions: {fraud_count} ({fraud_rate:.2f}%)
- Legitimate Transactions: {total_transactions - fraud_count} ({100 - fraud_rate:.2f}%)

KEY COLUMNS AVAILABLE:
- Transaction_ID, Customer_ID, Transaction_Date, Transaction_Time
- Customer_Age, Customer_Loyalty_Tier, Location, Store_ID
- Product_SKU, Product_Category, Purchase_Amount
- Payment_Method, Device_Type, IP_Address, Fraud_Flag, Footfall_Count

"""
    
    # Add comprehensive statistical summaries for ALL fraud transactions
    if len(fraud_df) > 0:
        summary += f"\n\nCOMPREHENSIVE FRAUD STATISTICS (Based on ALL {len(fraud_df)} fraudulent transactions):\n"
        
        # Loyalty Tier Analysis
        if 'Customer_Loyalty_Tier' in fraud_df.columns:
            loyalty_fraud = fraud_df['Customer_Loyalty_Tier'].value_counts()
            loyalty_percentages = (loyalty_fraud / len(fraud_df) * 100).round(2)
            summary += f"\nCustomer Loyalty Tier Distribution in Fraud:\n"
            for tier, count in loyalty_fraud.items():
                pct = loyalty_percentages[tier]
                summary += f"  - {tier}: {count} transactions ({pct}% of all fraud)\n"
        
        # Payment Method Analysis
        if 'Payment_Method' in fraud_df.columns:
            payment_fraud = fraud_df['Payment_Method'].value_counts()
            payment_percentages = (payment_fraud / len(fraud_df) * 100).round(2)
            summary += f"\nPayment Methods in Fraud:\n"
            for method, count in payment_fraud.items():
                pct = payment_percentages[method]
                summary += f"  - {method}: {count} transactions ({pct}%)\n"
        
        # Product Category Analysis
        if 'Product_Category' in fraud_df.columns:
            product_fraud = fraud_df['Product_Category'].value_counts()
            product_percentages = (product_fraud / len(fraud_df) * 100).round(2)
            summary += f"\nProduct Categories in Fraud:\n"
            for category, count in product_fraud.items():
                pct = product_percentages[category]
                summary += f"  - {category}: {count} transactions ({pct}%)\n"
        
        # Store Analysis
        if 'Store_ID' in fraud_df.columns:
            store_fraud = fraud_df['Store_ID'].value_counts()
            store_percentages = (store_fraud / len(fraud_df) * 100).round(2)
            summary += f"\nStores in Fraud (Top 10):\n"
            for store, count in store_fraud.head(10).items():
                pct = store_percentages[store]
                summary += f"  - {store}: {count} transactions ({pct}%)\n"
        
        # Location Analysis
        if 'Location' in fraud_df.columns:
            location_fraud = fraud_df['Location'].value_counts()
            location_percentages = (location_fraud / len(fraud_df) * 100).round(2)
            summary += f"\nLocations in Fraud (Top 10):\n"
            for location, count in location_fraud.head(10).items():
                pct = location_percentages[location]
                summary += f"  - {location}: {count} transactions ({pct}%)\n"
        
        # Device Type Analysis
        if 'Device_Type' in fraud_df.columns:
            device_fraud = fraud_df['Device_Type'].value_counts()
            device_percentages = (device_fraud / len(fraud_df) * 100).round(2)
            summary += f"\nDevice Types in Fraud:\n"
            for device, count in device_fraud.items():
                pct = device_percentages[device]
                summary += f"  - {device}: {count} transactions ({pct}%)\n"
        
        # Purchase Amount Statistics
        if 'Purchase_Amount' in fraud_df.columns:
            summary += f"\nPurchase Amount Statistics for Fraud:\n"
            summary += f"  - Average: ${fraud_df['Purchase_Amount'].mean():.2f}\n"
            summary += f"  - Median: ${fraud_df['Purchase_Amount'].median():.2f}\n"
            summary += f"  - Min: ${fraud_df['Purchase_Amount'].min():.2f}\n"
            summary += f"  - Max: ${fraud_df['Purchase_Amount'].max():.2f}\n"
            summary += f"  - Total Fraud Amount: ${fraud_df['Purchase_Amount'].sum():.2f}\n"
        
        # Customer Age Statistics
        if 'Customer_Age' in fraud_df.columns:
            summary += f"\nCustomer Age Statistics for Fraud:\n"
            summary += f"  - Average Age: {fraud_df['Customer_Age'].mean():.1f} years\n"
            summary += f"  - Min Age: {fraud_df['Customer_Age'].min():.0f} years\n"
            summary += f"  - Max Age: {fraud_df['Customer_Age'].max():.0f} years\n"
        
        # Time Analysis (if Transaction_Time exists)
        if 'Transaction_Time' in fraud_df.columns:
            summary += f"\nNote: Transaction times available for time-based pattern analysis\n"
    
    return summary

@app.route("/chat", methods=["POST"])
def chat_route():
    user_message = request.json["message"]

    # Add user message
    conversation_history.append(HumanMessage(content=user_message))

    # Add dataset context if available
    if df is not None:
        dataset_summary = get_dataset_summary()
        
        # Modify last user message with dataset context
        conversation_history[-1] = HumanMessage(
            content=f"""You have access to a cosmetics fraud detection dataset. Here's a summary:

{dataset_summary}

Now answer this question clearly and professionally:
{user_message}

Remember to:
- Provide well-structured, formatted responses
- Use specific numbers and statistics from the data
- Use **bold text** for section headings (NOT ## or ###)
- Use bullet points (*) and numbered lists for better readability
- Keep it clear and professional
- Avoid using markdown headers (##, ###) - use bold text instead
- Avoid messy formatting"""
        )

    # Get AI response
    try:
        response = chat.invoke(conversation_history)
        ai_response = response.content
    except Exception as e:
        ai_response = f"⚠️ Error: {str(e)}"

    # Add AI message
    conversation_history.append(AIMessage(content=ai_response))

    return jsonify({"reply": ai_response})

@app.route("/clear", methods=["POST"])
def clear_chat():
    global conversation_history
    conversation_history = [SystemMessage(content="""You are a professional fraud detection analyst assistant. 
Your role is to analyze cosmetics transaction data and provide clear, well-structured insights about fraud patterns.

IMPORTANT GUIDELINES:
- Provide responses in a clear, professional format
- Use bullet points, numbered lists, and proper formatting
- Include specific statistics and percentages when relevant
- Highlight key patterns and anomalies
- Keep responses concise but comprehensive
- Use **bold text** for emphasis and section titles (NOT markdown headers like ## or ###)
- Use bullet points (*) and numbered lists for better readability
- Avoid using markdown headers (##, ###) - instead use **bold text** for section titles
- Avoid messy formatting or special characters like ^^
- Structure your responses with clear sections using bold text for headings""")]
    return jsonify({"reply": "✅ Chat history cleared."})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
