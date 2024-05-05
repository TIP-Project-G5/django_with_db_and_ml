from django.shortcuts import render
from django.http import HttpResponse
from django.db.models import Avg
from .models import Rainfall, Groundwater, Depot, Area, Prediction

import pandas as pd
import numpy as np
import os

from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.impute import SimpleImputer

def home_page_view(request):
    return HttpResponse("Hello, World!")

def ml_test(request):
    if request.method == 'POST' and 'get_predictions' in request.POST:
    
        # Create an empty predictions dataframe with headers only
        headers = [
            f'rainfall_forecast{i+1}' for i in range(7)
        ] + [
            f'groundwater_forecast{i+1}' for i in range(7)
        ] + [
            f'flood_forecast{i+1}' for i in range(7)
        ] + ['area_id']
        
        pd.DataFrame(columns=headers).to_csv('predictions.csv', index=False)
        
        # Fetch unique area IDs
        area_ids = Area.objects.values_list('area_id', flat=True).distinct()
        
        for area_id in area_ids:
            
            def preprocess_data(file_path, date_col, target_col, floor_value=None, cap_value=None):
                
                # Load the CSV file into a pandas dataframe
                df = pd.read_csv(file_path, header=0)
               
                # Rename the 'Date' column to 'ds' and convert its format
                df['ds'] = pd.to_datetime(df[date_col], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
                
                # Rename the target column to 'y'
                df.rename(columns={target_col: 'y'}, inplace=True)
                
                # Drop unneeded columns from the dataframe
                df.drop(columns=[col for col in df.columns if col not in ['ds', 'y']], inplace=True)

                # Set floor and cap values for logistic growth model
                if floor_value is None:
                    floor_value = df['y'].min() * 1.5
                if cap_value is None:
                    cap_value = df['y'].max() * 1.5

                df['floor'] = floor_value
                df['cap'] = cap_value
                
                return df

            def run_prophet_model(df, clip_negative=False):
                # Fit the model (use logistic growth rather than linear model to allow a saturated minimum to avoid negative predictions)
                m = Prophet(growth='logistic')
                m.fit(df)
                
                # Generate predictions dataframe with number of periods to predict ahead (df will also include historical data)
                future = m.make_future_dataframe(periods=7)
                
                # Ensure the floor and cap are also applied to the future dataframe
                future['floor'] = df['floor'].iloc[0]
                future['cap'] = df['cap'].iloc[0]
                
                # Generate predictions as well as upper and lower predicted levels and add to predictions df
                forecast = m.predict(future)
                
                # Clip negative predictions to zero
                if clip_negative:
                    forecast['yhat'] = forecast['yhat'].clip(lower=0)
                    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
                    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
                
                return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                
            # Get data for the current area
            rainfall_data = Rainfall.objects.filter(rf_area_id=area_id).values('timestamp', 'rainfall_value')
            groundwater_data = Groundwater.objects.filter(gw_depot_no__dp_area_id=area_id).values('gw_timestamp').annotate(avg_groundwater=Avg('groundwater_value'))
            
            # Convert data to df's
            df_rainfall = pd.DataFrame(list(rainfall_data))
            df_groundwater = pd.DataFrame(list(groundwater_data))

            # Rename columns for clarity
            df_rainfall.rename(columns={'timestamp': 'Date', 'rainfall_value': 'Rainfall'}, inplace=True)
            df_groundwater.rename(columns={'gw_timestamp': 'Date', 'avg_groundwater': 'Groundwater'}, inplace=True)

            # Merge the dataframes on Date
            combined_data = pd.merge(df_rainfall, df_groundwater, on='Date', how='outer')
            
            # Reformat the Date column
            combined_data['Date'] = pd.to_datetime(combined_data['Date'], format='%Y-%m-%d').dt.strftime('%d/%m/%Y')
            
            # Save the combined dataframe to a CSV file
            combined_data.to_csv('combined_data.csv', index=False)
            
            # Define file path and column names
            file_path = 'combined_data.csv'
            date_col = 'Date'
             
            # Run for 'rainfall' with floor of 0 and cap as max 'y', clipping negative values
            df_rainfall = preprocess_data(file_path, date_col, 'Rainfall', 0, None)
            forecast_rainfall = run_prophet_model(df_rainfall, clip_negative=True)

            # Run for 'Groundwater level' with floor as min 'y' and cap of 1, not clipping negative values
            df_groundwater = preprocess_data(file_path, date_col, 'Groundwater', None, 1)
            df_groundwater['floor'] = df_groundwater['y'].min() 
            forecast_groundwater = run_prophet_model(df_groundwater, clip_negative=False)
            
            def process_and_merge_forecasts(file_path, forecast_rainfall, forecast_groundwater):
                # Select the last 7 rows from both forecasts
                forecast_rainfall = forecast_rainfall.tail(7)
                forecast_groundwater = forecast_groundwater.tail(7)

                # Create a new dataframe that merges the two forecasts
                combined_forecast = pd.merge(
                    forecast_rainfall, 
                    forecast_groundwater, 
                    on='ds', 
                    suffixes=('_rainfall', '_groundwater')
                )

                # Rename columns in combined forecast df and convert the date format
                combined_forecast.rename(
                    columns={'ds': 'Date', 'yhat_rainfall': 'Rainfall', 'yhat_groundwater': 'Groundwater'}, 
                    inplace=True
                )
                combined_forecast['Date'] = pd.to_datetime(combined_forecast['Date']).dt.strftime('%d/%m/%Y')

                # Select columns from combined df
                combined_forecast = combined_forecast[['Date', 'Rainfall', 'Groundwater']]

                # Load the original CSV file and rename columns
                temp_df = pd.read_csv(file_path, header=0)
                temp_df.rename(columns={'Time': 'Date', 'rainfall': 'Rainfall', 'Groundwater level': 'Groundwater'}, inplace=True)

                # Append the forecasts to the original df
                final_df = pd.concat([temp_df, combined_forecast], ignore_index=True)
                
                return final_df

            df = process_and_merge_forecasts(file_path, forecast_rainfall, forecast_groundwater)
            
            # Standardise the data
            standardise = ['Rainfall', 'Groundwater']
            scaler = StandardScaler()
            df[standardise] = scaler.fit_transform(df[standardise])
            
            # Impute missing values for DBSCAN
            imputer = SimpleImputer(strategy='mean')
            df[['Rainfall', 'Groundwater']] = imputer.fit_transform(df[['Rainfall', 'Groundwater']])
            
            # Save the df to a CSV file for testing
            df.to_csv('df.csv', index=False)
                  
            def find_optimal_dbscan_params(df, feature_columns, eps_range, min_samples_range):
                # Standardize the selected features
                X = df[feature_columns].values
                X_scaled = StandardScaler().fit_transform(X)

                # Initialize the best score tracking variables
                best_score = -1
                best_eps = None
                best_min_samples = None

                # Perform grid search over eps and min_samples
                for eps in eps_range:
                    for min_samples in min_samples_range:
                        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
                        labels = db.labels_
                        
                        # Calculate the silhouette score only if valid clusters are found
                        if len(set(labels)) > 1 and -1 not in labels:
                            score = silhouette_score(X_scaled, labels)
                            if score > best_score:
                                best_score = score
                                best_eps = eps
                                best_min_samples = min_samples

                return best_eps, best_min_samples

            # Define parameters and dataframe
            feature_columns = ['Rainfall', 'Groundwater']

            #### Best eps_range so far: (0.25, 0.75, 0.01)
            eps_range = np.arange(0.25, 0.75, 0.01) # Adjust start, stop, and step values in arrange() to adjust size/number of clusters
            min_samples_range = range(1, 3)

            # Run DBSCAN and retrieve the best parameters along with the modified dataframe
            best_eps, best_min_samples = find_optimal_dbscan_params(df, feature_columns, eps_range, min_samples_range)
            
            # Select the columns to use for clustering
            X = df[['Rainfall', 'Groundwater']].values

            # Initialize and fit DBSCAN clustering
            db = DBSCAN(eps=best_eps, min_samples=best_min_samples).fit(X)
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            
            # Add cluster labels back to the dataframe for further analysis
            df['Cluster_Labels'] = labels
            
            # Convert 'Date' column to datetime format
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

            # Filter the dataframe to show flood dates based on a condition in 'Cluster_Labels'
            flood_dates = df.loc[df['Cluster_Labels'] != 0, ['Date', 'Rainfall', 'Groundwater']]

            # Order by 'Date'
            flood_dates = flood_dates.sort_values(by='Date')

            # Get 7-day forecasts for rainfall and groundwater
            forecast_rainfall = run_prophet_model(df_rainfall, clip_negative=True).tail(7)
            forecast_groundwater = run_prophet_model(df_groundwater, clip_negative=False).tail(7)
            
            # Determine flood dates
            flood_dates_df = df.loc[df['Cluster_Labels'] != 0, ['Date']].drop_duplicates()
            flood_dates_df['Date'] = pd.to_datetime(flood_dates_df['Date'])

            # Creating lists for each forecast to use in dataframe construction
            rainfall_forecasts = [forecast_rainfall.iloc[i]['yhat'] for i in range(7)]
            groundwater_forecasts = [forecast_groundwater.iloc[i]['yhat'] for i in range(7)]
            flood_forecasts = [
                1 if forecast_rainfall.iloc[i]['ds'] in flood_dates_df['Date'].values else 0 
                for i in range(7)
            ]

            # Create dictionaries for dataframe initialization
            rainfall_data = {f'rainfall_forecast{i+1}': [rainfall_forecasts[i]] for i in range(7)}
            groundwater_data = {f'groundwater_forecast{i+1}': [groundwater_forecasts[i]] for i in range(7)}
            flood_data = {f'flood_forecast{i+1}': [flood_forecasts[i]] for i in range(7)}

            # Merge the dictionaries and add the 'area_id'
            data = {**rainfall_data, **groundwater_data, **flood_data, 'area_id': [area_id]}


            # Prepare the predictions dataframe
            predictions = pd.DataFrame(data)
            
            # Append data to predictions.csv
            predictions.to_csv('predictions.csv', index=False, mode='a', header=False)

        # Append data to predictions.csv
        predictions = pd.read_csv('predictions.csv', header=0)
        
        # Delete all rows in the Prediction model
        Prediction.objects.all().delete()
        
        # Insert predictions into the database
        for _, row in predictions.iterrows():
            prediction = Prediction(
                p_area_id=Area.objects.get(area_id=row['area_id']),
                groundwater_forecast1=row['groundwater_forecast1'],
                groundwater_forecast2=row['groundwater_forecast2'],
                groundwater_forecast3=row['groundwater_forecast3'],
                groundwater_forecast4=row['groundwater_forecast4'],
                groundwater_forecast5=row['groundwater_forecast5'],
                groundwater_forecast6=row['groundwater_forecast6'],
                groundwater_forecast7=row['groundwater_forecast7'],
                rainfall_forecast1=row['rainfall_forecast1'],
                rainfall_forecast2=row['rainfall_forecast2'],
                rainfall_forecast3=row['rainfall_forecast3'],
                rainfall_forecast4=row['rainfall_forecast4'],
                rainfall_forecast5=row['rainfall_forecast5'],
                rainfall_forecast6=row['rainfall_forecast6'],
                rainfall_forecast7=row['rainfall_forecast7'],
                min_temp_forecast1=row.get('min_temp_forecast1', 0), 
                min_temp_forecast2=row.get('min_temp_forecast2', 0),
                min_temp_forecast3=row.get('min_temp_forecast3', 0),
                min_temp_forecast4=row.get('min_temp_forecast4', 0),
                min_temp_forecast5=row.get('min_temp_forecast5', 0),
                min_temp_forecast6=row.get('min_temp_forecast6', 0),
                min_temp_forecast7=row.get('min_temp_forecast7', 0),
                max_temp_forecast1=row.get('max_temp_forecast1', 0),
                max_temp_forecast2=row.get('max_temp_forecast2', 0),
                max_temp_forecast3=row.get('max_temp_forecast3', 0),
                max_temp_forecast4=row.get('max_temp_forecast4', 0),
                max_temp_forecast5=row.get('max_temp_forecast5', 0),
                max_temp_forecast6=row.get('max_temp_forecast6', 0),
                max_temp_forecast7=row.get('max_temp_forecast7', 0),
                flood_forecast1=row['flood_forecast1'],
                flood_forecast2=row['flood_forecast2'],
                flood_forecast3=row['flood_forecast3'],
                flood_forecast4=row['flood_forecast4'],
                flood_forecast5=row['flood_forecast5'],
                flood_forecast6=row['flood_forecast6'],
                flood_forecast7=row['flood_forecast7']
            )
            prediction.save() 
            
        context = {
            'predictions': predictions.to_html(classes='table table-striped', index=False)
        }
        return render(request, 'ml_test.html', context)

    return render(request, 'ml_test.html')