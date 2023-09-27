# -*- coding: utf-8 -*-
"""
@author: jjcharzynski
"""

# WCA permalinks
# SQL: https://www.worldcubeassociation.org/export/results/WCA_export.sql.zip
# TSV: https://www.worldcubeassociation.org/export/results/WCA_export.tsv.zip

def createWCAdatabase():
    """
    Downloads the WCA competition data, extracts it, and creates an SQLite database.
    
    This function downloads the WCA competition data from the specified URL,
    extracts it into separate TSV and JSON files, and creates an SQLite database
    to store the data. It checks if the database already exists and if the
    export_date is not newer to avoid unnecessary recreation.

    Args:
        None

    Returns:
        None
    """
    import os
    import sys
    import requests
    import zipfile
    import json
    import pandas as pd
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import shutil
    
    # URL for the WCA data
    data_url = "https://www.worldcubeassociation.org/export/results/WCA_export.tsv.zip"
    
    # Path to save the downloaded ZIP file
    zip_file_path = "WCA_export.tsv.zip"
    
    # Path to extract the TSV files from the ZIP
    extracted_folder_path = "WCA_export"
    
    # Path for the SQLite database
    db_path = "wca_database.db"
    
    # Download the ZIP file
    response = requests.get(data_url)
    with open(zip_file_path, "wb") as f:
        f.write(response.content)
    
    # Extract the TSV files from the ZIP
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extracted_folder_path)
    
    # List all TSV and JSON files in the extracted folder
    tsv_files = [f for f in os.listdir(extracted_folder_path) if f.endswith(".tsv")]
    json_files = [f for f in os.listdir(extracted_folder_path) if f.endswith(".json")]
    
    # Read JSON file for export information
    json_file_path = os.path.join(extracted_folder_path, json_files[0])
    with open(json_file_path, "r") as json_file:
        json_data = json.load(json_file)
    
    # Check if the database already exists and export_date is not newer
    db_exists = os.path.exists(db_path)
    export_date = json_data["export_date"]
    
    if db_exists:
        engine = create_engine(f"sqlite:///{db_path}", echo=True)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        last_export_info = session.execute("SELECT * FROM export_info ORDER BY export_date DESC LIMIT 1").fetchone()
        last_export_date = last_export_info["export_date"]
        
        if last_export_date >= export_date:
            print("Database is up-to-date. No need to recreate.")
            session.close()
            
            # Clean up: remove downloaded files and extracted folder contents
            os.remove(zip_file_path)
            os.remove(json_file_path)
            shutil.rmtree(extracted_folder_path)
    
            sys.exit()
    
    # Create an SQLite database
    engine = create_engine(f"sqlite:///{db_path}", echo=True)
    
    # Create a table for export information
    export_info_df = pd.DataFrame([json_data])
    export_info_df.to_sql("export_info", con=engine, if_exists="replace", index=False)
    
    # Iterate through each TSV file, read into a DataFrame, and write to the database
    for tsv_file in tsv_files:
        tsv_file_path = os.path.join(extracted_folder_path, tsv_file)
        table_name = os.path.splitext(tsv_file)[0]
        
        # Read the TSV data into a DataFrame
        data = pd.read_csv(tsv_file_path, delimiter="\t")
        
        # Write the DataFrame to the SQLite database
        data.to_sql(table_name, con=engine, if_exists="replace", index=False)
    
    # Clean up: remove downloaded files and extracted folder contents
    os.remove(zip_file_path)
    for tsv_file in tsv_files:
        os.remove(os.path.join(extracted_folder_path, tsv_file))
    os.remove(json_file_path)
    os.remove(os.path.join(extracted_folder_path, "README.md"))
    os.rmdir(extracted_folder_path)
    
    print("Database created successfully.")

def pullalldataforpersonandevent(personId,eventId):
    """
    Retrieves and processes competition data for a specific person and event.

    This function queries the SQLite database for competition data associated with
    a particular person and event. It calculates and preprocesses relevant information
    including date, days since the first competition, and removes rows with specific
    values in "best" and "average" columns.

    Args:
        personId (str): The unique identifier of the person.
        eventId (str): The unique identifier of the event.

    Returns:
        pd.DataFrame or None: A DataFrame containing the processed competition data
        for the specified person and event, or None if no data is found.
    """
    import pandas as pd
    from sqlalchemy import create_engine
    
    # Path to your SQLite database file
    db_path = "wca_database.db"
    
    # Create an SQLAlchemy engine
    engine = create_engine(f"sqlite:///{db_path}", echo=True)
    
    # Specify the table names
    results_table_name = "WCA_export_Results"
    competitions_table_name = "WCA_export_Competitions"
    
    # Construct the SQL query to filter the data from results table
    results_query = f"SELECT * FROM {results_table_name} WHERE eventId = '{eventId}' AND personId = '{personId}'"
    
    # Read the filtered data from results table into a pandas DataFrame
    results_data = pd.read_sql(results_query, con=engine)
    if results_data.empty:
        # print("df is empty")
        return None
    else:
        # Construct the SQL query to fetch the year, month, and day for the competitionId
        competitions_query = f"SELECT id, year, month, day FROM {competitions_table_name}"
        
        # Read the competition data into a pandas DataFrame
        competitions_data = pd.read_sql(competitions_query, con=engine)
        
        # Merge the results_data and competitions_data DataFrames on competitionId
        merged_data = pd.merge(results_data, competitions_data, left_on="competitionId", right_on="id")
        
        # Calculate the date using year, month, and day fields and create a new "date" column
        merged_data["date"] = pd.to_datetime(merged_data[["year", "month", "day"]])
        
        # Calculate the earliest date
        earliest_date = merged_data["date"].min()
        
        # Calculate the DaysSinceFirstComp column and set the earliets date to 0.5
        merged_data["DaysSinceFirstComp"] = (merged_data["date"] - earliest_date).dt.days
        merged_data.at[0, "DaysSinceFirstComp"] = 0.5
        
        # Delete unecessary columns from the dataframe
        columns_to_remove = ["id", "year", "month", "day"]
        merged_data = merged_data.drop(columns=columns_to_remove)
        
        # Remove rows with specific values in "best" and "average" columns
        # The value `-1` means DNF (Did Not Finish).
        # The value `-2` means DNS (Did Not Start).
        # The value `0` means "no result".
        values_to_remove = [0, -1, -2]
        
        # Separate rows to remove into a separate DataFrame
        rows_to_remove = merged_data[(merged_data["best"].isin(values_to_remove)) | (merged_data["average"].isin(values_to_remove))]
        
        # Remove rows with specific values from the merged_data DataFrame
        filtered_data = merged_data[~((merged_data["best"].isin(values_to_remove)) | (merged_data["average"].isin(values_to_remove)))]
        
        # Convert from ms to s
        filtered_data["best"] = filtered_data["best"]/100
        filtered_data["average"] = filtered_data["average"]/100
        
        # Display the merged DataFrame
        pd.set_option("display.max_columns", None)
        # print("The following rows are removed:")
        # print(rows_to_remove)
        # print(filtered_data)
        return filtered_data

def pullpersonIds():
    """
    Retrieves a list of person IDs and names from the SQLite database.

    This function queries the SQLite database to fetch a list of person IDs and names.

    Args:
        None

    Returns:
        pd.DataFrame, tuple: A DataFrame containing person IDs and names, and a tuple
        containing the person IDs.
    """
    import pandas as pd
    from sqlalchemy import create_engine
    
    # Path to your SQLite database file
    db_path = "wca_database.db"
    
    # Create an SQLAlchemy engine
    engine = create_engine(f"sqlite:///{db_path}", echo=True)
    
    # Specify the table names
    persons_table_name = "WCA_export_Persons"
    
    # Construct the SQL query to filter the data from persons table
    persons_query = f"SELECT id, name FROM {persons_table_name}"
    
    # Read the persons data into a pandas DataFrame
    persons_data = pd.read_sql(persons_query, con=engine)
    # print(persons_data)
    persons_id_tuple = tuple(persons_data['id'].tolist())
    
    return persons_data, persons_id_tuple

def calculatepersonalprogressbyevent(personId,eventId):
    """
    Calculates and plots the personal progress of a person for a specific event.

    This function calculates and plots the personal progress of a person for a specific
    event. It uses linear regression and exponential curve fitting to model the progress
    trends over time.

    Args:
        personId (str): The unique identifier of the person.
        eventId (str): The unique identifier of the event.

    Returns:
        tuple: A tuple containing various data, including the raw and aggregated competition
        data, regression information, and trend lines.
    """
    # import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from scipy.optimize import curve_fit
    # import pandas as pd
    
    merged_data = pullalldataforpersonandevent(personId,eventId)
    # print(type(merged_data))
    if merged_data is None:
        # print("...........................................................................................................")
        aggregated_data = merged_data
        x, xe, y_pred_best, y_pred_average, x_trend, y_trend, x_trend2, y_trend2 = (0, 0, 0, 0, 0, 0, 0, 0)
    elif len(merged_data) != 0:

        # Aggregate the data by taking the mean of y values for each unique x value
        # print(merged_data)
        aggregated_data = merged_data.groupby("DaysSinceFirstComp").mean()
        
        # Adding trend line using linear regression
        x = merged_data["DaysSinceFirstComp"].values.reshape(-1, 1)
        y_best = merged_data["best"].values.reshape(-1, 1)
        y_average = merged_data["average"].values.reshape(-1, 1)
        
        # Fit linear regression models
        reg_best = LinearRegression().fit(x, y_best)
        reg_average = LinearRegression().fit(x, y_average)
        
        # Predict using the linear regression models
        y_pred_best = reg_best.predict(x)
        y_pred_average = reg_average.predict(x)
        
        # Adding exponential and logarithmic trend lines
        xe = aggregated_data.index.values
        y_best = aggregated_data["best"].values
        y_average = aggregated_data["average"].values
        
        # Define the curve function
        def exponential_curve(xe, A1, r1, A2, r2):
            return A1 * np.exp(r1 * xe) + A2 * np.exp(r2 * xe)
        
        try:
            # Fit the curve to the Best data
            initial_guess = [1, -0.01, 1, -0.01]  # Initial parameter guess
            params, covariance = curve_fit(exponential_curve, xe, y_best, p0=initial_guess)
            # Extract the fitted parameters
            A1_fit, r1_fit, A2_fit, r2_fit = params
            # Generate x values for the trendline
            x_trend = np.linspace(min(xe), max(xe), 100)
            y_trend = exponential_curve(x_trend, A1_fit, r1_fit, A2_fit, r2_fit)
        
            # Fit the curve to the Average data
            initial_guess = [1, -0.01, 1, -0.01]  # Initial parameter guess
            params, covariance = curve_fit(exponential_curve, xe, y_average, p0=initial_guess)
            # Extract the fitted parameters
            A1_fit, r1_fit, A2_fit, r2_fit = params
            # Generate x values for the trendline
            x_trend2 = np.linspace(min(xe), max(xe), 100)
            y_trend2 = exponential_curve(x_trend2, A1_fit, r1_fit, A2_fit, r2_fit)
        except (RuntimeError, TypeError):
            # print("Exponential trendline unable to be calculated.")
            x_trend, y_trend, x_trend2, y_trend2 = (0, 0, 0, 0)
    
    else:
        aggregated_data = merged_data
        x, xe, y_pred_best, y_pred_average, x_trend, y_trend, x_trend2, y_trend2 = (0, 0, 0, 0, 0, 0, 0, 0)
    
    return merged_data, aggregated_data, x, xe, y_pred_best, y_pred_average, x_trend, y_trend, x_trend2, y_trend2


def plotpersonalprogressbyevent(personId,eventId):
    """
    Plots the personal progress of a person for a specific event.

    This function plots the personal progress of a person for a specific event, including
    scatter plots of best and average times, trend lines, and fitted exponential curves.

    Args:
        personId (str): The unique identifier of the person.
        eventId (str): The unique identifier of the event.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    
    merged_data, aggregated_data, x, xe, y_pred_best, y_pred_average, x_trend, y_trend, x_trend2, y_trend2 = calculatepersonalprogressbyevent(personId,eventId)
    
    #Define Name and Event Variables (need to improve to Julian Charzynski/3x3x3 format)
    name=personId
    event=eventId
    
    # Plotting using matplotlib with scatter plot and trend lines
    plt.figure(figsize=(10, 6))
    plt.scatter(merged_data["DaysSinceFirstComp"], merged_data["best"], label="Best Time")
    plt.scatter(merged_data["DaysSinceFirstComp"], merged_data["average"], label="Average Time")
    # plt.scatter(aggregated_data.index, aggregated_data["best"], label="Mean Best Time", marker='x', color='red')
    # plt.scatter(aggregated_data.index, aggregated_data["average"], label="Mean Average Time", marker='x', color='green')
    plt.xlabel("Days Since First Competition")
    plt.ylabel("Time (s)")
    title1 = f"{name}'s Best and Average {event} Times"
    plt.title(title1)
    
    # Plot the linear trend lines
    plt.plot(x, y_pred_best, color="blue", linestyle="--", label="Best Trend Line")
    plt.plot(x, y_pred_average, color="orangered", linestyle="--", label="Average Trend Line")
    
    # Plot fitted exponential curves
    # if x_trend.any != 0:
    try:
        plt.plot(x_trend, y_trend, label="Best Fitted Exponential Curve", color='blue')
        plt.plot(x_trend2, y_trend2, label="Average Fitted Exponential Curve", color='orangered')
    # else:
    except AttributeError: 
        print("Exponential trendline unable to be calculated, and therefore not plotted.")

    plt.legend()
    plt.grid(True)
    plt.show()

def comparecubersbyevent(eventId, *args):
    """
    Compares the personal progress of multiple cubers for a specific event.

    This function compares the personal progress of multiple cubers for a specific event
    by plotting their average times and trend lines on the same graph.

    Args:
        eventId (str): The unique identifier of the event.
        *args (str): Variable-length list of person IDs to compare.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.xlabel("Days Since First Competition")
    plt.ylabel("Time (s)")
    title1 = f"{eventId} Times"
    plt.title(title1)
    
    for ar in args:
        merged_datac1, aggregated_datac1, xc1, xec1, y_pred_bestc1, y_pred_averagec1, x_trendc1, y_trendc1, x_trend2c1, y_trend2c1 = calculatepersonalprogressbyevent(ar,eventId)
        
        c1 = ar
        # plt.scatter(merged_datac1["DaysSinceFirstComp"], merged_datac1["best"], label=f"{c1} Best Time", alpha=0.5)
        plt.scatter(merged_datac1["DaysSinceFirstComp"], merged_datac1["average"], label=f"{c1} Average Time", alpha=0.5)
        # plt.plot(x_trendc1, y_trendc1, label=f"{c1} Best Trend",)
        plt.plot(x_trend2c1, y_trend2c1, label=f"{c1} Average Trend")

    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("Code not complete!")

def cubervseverybody(eventId, personId, *args):
    """
    Compares a cuber's progress with others for a specific event.

    This function compares a specific cuber's progress with other cubers for a given event
    by plotting their average times on the same graph.

    Args:
        eventId (str): The unique identifier of the event.
        personId (str): The unique identifier of the cuber.
        *args (str): Variable-length list of person IDs to compare.

    Returns:
        None
    """
    import time
    import matplotlib.pyplot as plt
    
    start_time = time.time()
    
    plt.figure(figsize=(10, 6))
    plt.xlabel("Days Since First Competition")
    plt.ylabel("Time (s)")
    plt.ylim(0, 60)
    title1 = f"{eventId} Times"
    plt.title(title1)

    for ar in args:
        print(ar)
        merged_datac1, aggregated_datac1, xc1, xec1, y_pred_bestc1, y_pred_averagec1, x_trendc1, y_trendc1, x_trend2c1, y_trend2c1 = calculatepersonalprogressbyevent(ar,eventId)
        if merged_datac1 is not None:
            c1 = ar
            # plt.scatter(merged_datac1["DaysSinceFirstComp"], merged_datac1["best"], label=f"_{c1} Best Time", alpha=0.5)
            plt.scatter(merged_datac1["DaysSinceFirstComp"], merged_datac1["average"], label=f"_{c1} Average Time", alpha=0.5, color="gray")
            # plt.plot(x_trendc1, y_trendc1, label=f"_{c1} Best Trend",)
            # plt.plot(x_trend2c1, y_trend2c1, label=f"_{c1} Average Trend", color="gray")

    merged_datac1, aggregated_datac1, xc1, xec1, y_pred_bestc1, y_pred_averagec1, x_trendc1, y_trendc1, x_trend2c1, y_trend2c1 = calculatepersonalprogressbyevent(personId,eventId)
    c1 = personId
    # plt.scatter(merged_datac1["DaysSinceFirstComp"], merged_datac1["best"], label=f"{c1} Best Time", alpha=0.5)
    plt.scatter(merged_datac1["DaysSinceFirstComp"], merged_datac1["average"], label=f"{c1} Average Time", alpha=0.5, color="red")
    # plt.plot(x_trendc1, y_trendc1, label=f"{c1} Best Trend",)
    # plt.plot(x_trend2c1, y_trend2c1, label=f"{c1} Average Trend", color="red") 
    
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("Code not complete!")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
