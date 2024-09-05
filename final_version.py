import json
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

# Load data function using the updated caching mechanism
@st.cache_data
def load_reservations():
    url = "https://public.api.hospitable.com/v2/reservations"
    querystring = {
        "properties[]": ["1176448", "1176486", "1435074", "1176450", "1239502", "1392158"],
        "start_date": "2024-04-16",
        "end_date": "2025-12-31",
        "include": "financials,properties",
        "per_page": 50
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiI5YTYyNGRmMC0xMmYxLTQ0OGUtYjg4NC00MzY3ODBhNWQzY2QiLCJqdGkiOiI1OGI4N2I0MTBiNTVjZDYwMjY1ODUxMmNmNmRlNGE3YjE0NTBkZDEyY2QyNmY0ZTlmNzA4YmQ4MWUxY2ZlODA3YzBmYTYwMzE0ODliN2I4ZSIsImlhdCI6MTcxMTYyMjg0MC41NDU5NTYsIm5iZiI6MTcxMTYyMjg0MC41NDU5NiwiZXhwIjoxNzQzMTU4ODQwLjU0MjA5Nywic3ViIjoiMTY0NzU0Iiwic2NvcGVzIjpbInBhdDpyZWFkIiwicGF0OndyaXRlIl19.KKILeRg3ER-zxIUhEINK1mlHElCQJwMiB9XGh16HJLBA6OmxrB6boG86nC_rI710Im8le5ZfqQYb7PLdaJNu6epqLDPiHf_a9MsOQjdvppJkqV6NHSt7ejQ2offY8RN-qzc0SMucrXRGFxD5CWI3cuePskV7EY6SpfHLgQLOUiq07TfH7Nz5cz1vHh-Zu7PG-QUCHavVntlKe9vAWgFPP3u_743ybiVMqhdkWtauktYO3tytV3d5qAY0UVEvofbq0S_sR9OJ0vKKDYDMecynvxp58eubTLMSucp_eXiwyyPtkb8IM5rRxy4XibMIMUtVMSKmhiTsY4-MqqTSe95-sNlrEBJuv9VkHAwCvpYE5Do4hXUm05Ix7RWPoYefPUj8ceGkZClO40jzLDJDmbsfmwfemWzqCG2m-2h6uyns68ZsWVem9lj2k-_lKiyom0ichddaGKYvZbGOMYSJMfU3nPAqD6CU3lvdJU1anNLUWa1sIxucFTBS4yA1S96HSIi9EY72yBBlHfI4PDCXepuxpLtQVI6ngoD7zdYxDKiAytJf6SZ0mMzku8fbyHsvI2yZQ7hyZRFjqUsLDPX4W3afwVPz9Go5dujH5Z0OWjTa8f6uEKUFntq0YvaolP85DSkpV9RZOQvlkewXSKU0CHneCmrwvFSVfHv0BUp5EQ2hb-4"  # Replace with your actual API key
    }

    all_data = []
    current_page = 1
    total_pages = 1  # Initialize with a default

    while current_page <= total_pages:
        querystring["page"] = current_page

        response = requests.get(url, headers=headers, params=querystring)

        data = response.json()

        # Append the retrieved data to the all_data list
        all_data.extend(data['data'])

        # Extract pagination metadata
        meta = data.get('meta', {})
        current_page = meta.get('current_page', current_page)
        total_pages = meta.get('last_page', current_page)
        total_records = meta.get('total', len(all_data))

        # Move to the next page
        current_page += 1

    return all_data

def process_reservations(all_data):
    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Convert 'booking_date' to datetime format
    df['booking_date'] = pd.to_datetime(df['booking_date'])

    # Sort by 'booking_date' in ascending order
    df_sorted = df.sort_values(by='booking_date', ascending=True)

    # Reset index for clean output
    df_sorted.reset_index(drop=True, inplace=True)

    # Remove NA values
    df_sorted_clean = df_sorted.dropna()

    # Function to extract formatted revenue and property name
    def extract_revenue_and_name(row):
        financials = row['financials']
        properties = row['properties']

        # Extract formatted revenue from financials
        revenue_formatted = financials['host']['revenue']['formatted'] if 'financials' in row else None

        # Extract property name from properties
        if properties and isinstance(properties, list) and len(properties) > 0:
            property_name = properties[0]['name']
        else:
            property_name = None

        return pd.Series({
            'revenue_formatted': revenue_formatted,
            'property_name': property_name
        })

    # Apply the extraction function to each row
    df_sorted_clean[['revenue_formatted', 'property_name']] = df_sorted_clean.apply(extract_revenue_and_name, axis=1)

    # Filter columns and rename for clarity
    df_cleaned = df_sorted_clean.loc[:, ['id', 'code', 'platform', 'booking_date', 'check_in', 'check_out', 
                                         'revenue_formatted', 'property_name', 'status']]

    df_cleaned.rename(columns={'check_in': 'checkin_date', 'check_out': 'checkout_date', 
                               'revenue_formatted': 'revenue'}, inplace=True)

    # Convert 'booking_date' to naive datetime
    df_cleaned['booking_date'] = df_cleaned['booking_date'].dt.tz_localize(None)

    # Convert 'checkin_date' and 'checkout_date' to datetime and remove time zones
    df_cleaned['checkin_date'] = pd.to_datetime(df_cleaned['checkin_date'], errors='coerce', utc=True).dt.tz_localize(None)
    df_cleaned['checkout_date'] = pd.to_datetime(df_cleaned['checkout_date'], errors='coerce', utc=True).dt.tz_localize(None)

    # Calculate the number of nights per booking
    df_cleaned['nights'] = (df_cleaned['checkout_date'] - df_cleaned['checkin_date']).dt.days + 1

    # Remove currency symbols and commas from 'revenue' and convert to numeric
    df_cleaned['revenue'] = df_cleaned['revenue'].str.replace('[€,]', '', regex=True).str.replace(',', '', regex=True)
    df_cleaned['revenue'] = pd.to_numeric(df_cleaned['revenue'], errors='coerce')

    # Filter the DataFrame to include only necessary columns
    filtered_data_test = df_cleaned[df_cleaned['checkin_date'] >= '2024-04-17']
    columns_to_keep = ['code', 'platform', 'booking_date', 'checkin_date', 'checkout_date', 
                       'revenue', 'property_name', 'status', 'nights']
    filtered_data_test = filtered_data_test[columns_to_keep]

    return filtered_data_test

def load_and_filter_test_data():
    # Load the CSV file into a DataFrame
    test = pd.read_csv('test_data.csv')

    # Convert the 'checkin_date' to datetime format
    test['checkin_date'] = pd.to_datetime(test['checkin_date'])

    # Filter out rows where 'checkin_date' is after April 17, 2024
    filtered_test = test[test['checkin_date'] <= '2024-04-17']

    # Specify the columns to keep
    columns_to_keep = ['code', 'platform', 'booking_date', 'checkin_date', 'checkout_date',
                       'revenue', 'property_name', 'status', 'nights']

    # Keep only the specified columns
    filtered_test = filtered_test[columns_to_keep]

    # Return the filtered DataFrame
    return filtered_test

def combine_and_transform_datasets(filtered_test, filtered_data_test):
    # Combine the two datasets
    combined_dataset = pd.concat([filtered_test, filtered_data_test], ignore_index=True)

    # Convert 'booking_date' from object to datetime64[ns]
    combined_dataset['booking_date'] = pd.to_datetime(combined_dataset['booking_date'])

    # Convert 'checkout_date' from object to datetime64[ns]
    combined_dataset['checkout_date'] = pd.to_datetime(combined_dataset['checkout_date'])

    # Return the transformed combined dataset
    return combined_dataset

# KPI Calculations
def calculate_kpis(df):
    total_nights = df['nights'].sum()
    total_revenue = df['revenue'].sum()
    
    # Calculate the total available nights based on the data provided
    num_properties = len(df['property_name'].unique())
    period_days = (df['checkout_date'].max() - df['checkin_date'].min()).days + 1
    total_available_nights = period_days * num_properties
    
    occupancy_rate = round((total_nights / total_available_nights) * 100, 2)
    adr = round((total_revenue / total_nights), 2) if total_nights else 0
    revpar = round((total_revenue / total_available_nights), 2) if total_available_nights else 0

    # Ensure booking_date is converted to datetime format
    df['booking_date'] = pd.to_datetime(df['booking_date'])
    # Calculate average lead time in days
    lead_time = round((df['checkin_date'] - df['booking_date']).dt.days.mean(), 2)

    # Calculate the number of reservations
    number_of_reservations = df.shape[0]

    # Calculate the average length of stay
    average_length_of_stay = round(df['nights'].mean(), 2)

    return {
        'Occupancy Rate (%)': f"{occupancy_rate}%",
        'Average Daily Rate (ADR)': f"€{adr}",
        'Revenue Per Available Room (RevPAR)': f"€{revpar}",
        'Total Revenue': f"€{total_revenue:.2f}",
        'Average Lead Time (days)': f"{lead_time}",
        'Number of Reservations': number_of_reservations,
        'Average Length of Stay (nights)': average_length_of_stay,
    }

def plot_revenue_percentage_per_platform(df):
    revenue_per_platform = df.groupby('platform')['revenue'].sum().reset_index()
    fig = px.pie(revenue_per_platform, values='revenue', names='platform', title='Percentage of Revenue per Platform',
                 color='platform', color_discrete_map={'airbnb': 'red', 'booking.com': 'blue', 'vrbo': 'purple'},
                 hole=0.3)
    return fig

def plot_nights_booked_percentage_per_platform(df):
    nights_per_platform = df.groupby('platform')['nights'].sum().reset_index()
    fig = px.pie(nights_per_platform, values='nights', names='platform', title='Percentage of Nights Booked per Platform',
                 color='platform', color_discrete_map={'airbnb': 'red', 'booking.com': 'blue', 'vrbo': 'purple'},
                 hole=0.3)
    return fig

def plot_stacked_revenue_by_property_platform(df):
    # Step 1: Aggregate the data
    aggregated_df = df.groupby(['property_name', 'platform'], as_index=False)['revenue'].sum()

    # Step 2: Create the stacked bar chart
    fig = px.bar(aggregated_df, x='property_name', y='revenue', color='platform',
                 title='Stacked Revenue of Property by Platform',
                 color_discrete_map={'airbnb': 'red', 'booking': 'blue', 'vrbo': 'purple'})

    return fig

def plot_monthly_revenue_line(df):
    # Ensure date columns are in datetime format
    df['checkin_date'] = pd.to_datetime(df['checkin_date'])
    df['checkout_date'] = pd.to_datetime(df['checkout_date'])

    # Calculate the start of the month for grouping
    df['month'] = df['checkin_date'].dt.to_period('M').dt.start_time  # Convert to the first day of the month

    # Calculate the monthly revenue per property
    monthly_revenue = df.groupby(['month', 'property_name'])['revenue'].sum().reset_index()

    # Create the line chart
    fig = px.line(monthly_revenue, x='month', y='revenue', color='property_name', title='Monthly Revenue per Property')

    # Update y-axis to start at 0
    fig.update_yaxes(title='Revenue (€)',
                     range=[0, monthly_revenue['revenue'].max() * 1.1]
                     ,tickangle=90)

    # Format x-axis to show month names and ensure chronological order
    fig.update_xaxes(
        title='Month',
        dtick="M1",
        tickformat="%B %Y",  # Show month name and year
        categoryorder="category ascending"  # Ensure chronological order
    )

    return fig

def plot_monthly_occupancy_rate(df):
    # Ensure date columns are in datetime format
    df.loc[:, 'checkin_date'] = pd.to_datetime(df['checkin_date'])
    df.loc[:, 'checkout_date'] = pd.to_datetime(df['checkout_date'])

    # Calculate the start of the month for grouping
    df.loc[:, 'month'] = df['checkin_date'].dt.to_period('M').dt.start_time

    # Filter cancelled reservations
    #df = df[df['status'] != 'cancelled']
    df = df[~df['status'].isin(['cancelled', 'denied'])]


    # Calculate occupancy for each property each month
    monthly_occupancy = df.groupby(['month', 'property_name']).apply(
        lambda x: x['nights'].sum() / ((x['checkout_date'].max() - x['checkin_date'].min()).days + 1)
    ).reset_index(name='occupancy_rate')

    # Create the line chart
    fig = px.line(monthly_occupancy, x='month', y='occupancy_rate', color='property_name',
                  title='Monthly Occupancy Rate per Property')

    # Update y-axis to start at 0
    fig.update_yaxes(title='Occupancy Rate', tickformat=".0%", range=[0, 1.01],tickangle=90)

    # Format x-axis to show month names and ensure chronological order
    fig.update_xaxes(
        title='Month',
        dtick="M1",
        tickformat="%B %Y",  # Show month name and year
        categoryorder="category ascending"  # Ensure chronological order
    )

    return fig

def plot_monthly_adr_line(df):
    # Ensure date columns are in datetime format
    df['checkin_date'] = pd.to_datetime(df['checkin_date'])
    df['checkout_date'] = pd.to_datetime(df['checkout_date'])

    # Calculate the start of the month for grouping
    df['month'] = df['checkin_date'].dt.to_period('M').dt.start_time  # Convert to the first day of the month

    # Calculate the total revenue and total nights per month per property
    monthly_data = df.groupby(['month', 'property_name']).agg({
        'revenue': 'sum',
        'nights': 'sum'
    }).reset_index()

    # Calculate ADR (Average Daily Rate)
    monthly_data['adr'] = monthly_data['revenue'] / monthly_data['nights']

    # Create the line chart
    fig = px.line(monthly_data, x='month', y='adr', color='property_name', title='Monthly ADR (Average Daily Rate) per Property')

    # Update y-axis to start at 0
    fig.update_yaxes(title='ADR (€)', range=[0, monthly_data['adr'].max() * 1.1])

    # Format x-axis to show month names and ensure chronological order
    fig.update_xaxes(
        title='Month',
        dtick="M1",
        tickformat="%B %Y",  # Show month name and year
        categoryorder="category ascending"  # Ensure chronological order
    )

    return fig

# Main function where we define the app
def main():
    csv = load_and_filter_test_data()
    hospitable = load_reservations()
    clean_hospitable = process_reservations(hospitable)
    df = combine_and_transform_datasets(csv, clean_hospitable)
    # Streamlit title and introduction
    st.title("BI Dashboard for Short Term Rental Business")
    st.write("Interactive BI dashboard for analyzing booking data across multiple platforms.")

    # Sidebar - date range filter
    st.sidebar.header("Filters")
    if not df.empty:
        min_date = df['checkin_date'].min()
        max_date = df['checkout_date'].max()
        selected_dates = st.sidebar.date_input("Check-in Date", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        selected_dates = [pd.Timestamp(date) for date in selected_dates]  # Convert to Timestamp
        platform_filter = st.sidebar.multiselect("Platform", df['platform'].unique())
        property_filter = st.sidebar.multiselect("Property", df['property_name'].unique())

        # Filtering data based on selection
        filtered_df = df[(df['checkin_date'] >= selected_dates[0]) & (df['checkin_date'] <= selected_dates[1])]
        if platform_filter:
            filtered_df = filtered_df[filtered_df['platform'].isin(platform_filter)]
        if property_filter:
            filtered_df = filtered_df[filtered_df['property_name'].isin(property_filter)]

        # Display KPIs and progress bars
        kpis = calculate_kpis(filtered_df)
        for kpi, value in kpis.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.metric(label=kpi, value=f"{value}")
            with col2:
                if "Rating" in kpi:
                    st.progress(min(int((value / 5.0) * 100), 100))

            # Visualizations
        fig2 = plot_revenue_percentage_per_platform(filtered_df)
        st.plotly_chart(fig2)
        fig3 = plot_nights_booked_percentage_per_platform(filtered_df)
        st.plotly_chart(fig3)
        fig4 = plot_stacked_revenue_by_property_platform(filtered_df)
        st.plotly_chart(fig4)
        fig5 = plot_monthly_revenue_line(filtered_df)
        st.plotly_chart(fig5)
        fig6 = plot_monthly_occupancy_rate(filtered_df)
        st.plotly_chart(fig6)
        fig7 = plot_monthly_adr_line(filtered_df)
        st.plotly_chart(fig7) 

    else:
        st.write("No data loaded.")

# Run the app
if __name__ == "__main__":
    main()