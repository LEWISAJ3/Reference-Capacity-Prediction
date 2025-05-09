#!/usr/bin/env python
# coding: utf-8

# In[1]:




# Load packages
import scipy
import pandas as pd
import numpy as np


# In[2]:

class Converter():
    # Load NASA data
    def __init__(self):
        print("Converter.convert requires mat filepath,output path, the cell number, and the cell key within the mat file")
        
        print("For example: Converter.convert('Files/B0025.mat','Files/Cell25_Cleaned.parquet', 'B0025',4)")
        print("You can also choose to return the finished dataframe by running:")
        print("df=Converter.convert('Files/B0025.mat','Files/Cell25_Cleaned.parquet', 'B0025',4,ret=True)")
        
    def convert(self,filepath,outpath,cellkey,cellnum,ret=True):
        
        nasa = scipy.io.loadmat(filepath, simplify_cells=True)


        # In[3]:


        nasa.keys()


        # In[4]:


        # Change into dataframe
        data = nasa[cellkey]
        df = pd.DataFrame(data)


        # In[8]:

        print("Converting to dataframe...")
        # Initialize an empty list to hold the rows for the new dataframe
        data_rows = []

        # Iterate through each row in the existing dataframe
        for index, row in df.iterrows():
            cycle_data = row['cycle']

            # Only include cycles where the type is NOT 'impedance'
            if cycle_data['type'] != 'impedance':
                cycle_number = index  # or use a specific cycle number if needed
                cycle_type = cycle_data['type']
                ambient_temperature = cycle_data['ambient_temperature']
                time = cycle_data['time']

                # Extract the data dictionary
                data_dict = cycle_data['data']

                # Use the get method to safely access keys
                voltage_measured = data_dict.get('Voltage_measured', None)
                current_measured = data_dict.get('Current_measured', None)
                temperature_measured = data_dict.get('Temperature_measured', None)
                current_charge = data_dict.get('Current_charge', None)  # Will be None if not present
                voltage_charge = data_dict.get('Voltage_charge', None)  # Will be None if not present
                time_data = data_dict.get('Time', None)
                capacity_data=data_dict.get('Capacity',None)
                # Append the valid cycle data to the list
                data_rows.append({
                    'cycle': cycle_number,
                    'type': cycle_type,
                    'ambient_temperature': ambient_temperature,
                    'time': time,
                    'Voltage_measured': voltage_measured,
                    'Current_measured': current_measured,
                    'Temperature_measured': temperature_measured,
                    'Current_charge': current_charge,
                    'Voltage_charge': voltage_charge,
                    'Time': time_data,
                    'Capacity':capacity_data
                })

        # Create the new dataframe from the valid cycles
        nasa_df = pd.DataFrame(data_rows)

        # Display the new dataframe



        # In[9]:





        # In[10]:


        type_values = df['cycle'].apply(lambda x: x['type'])

        # Get the value counts
        type_counts = type_values.value_counts()



        # In[11]:





        # In[13]:


        # Initialize an empty list to hold the new rows
        data_rows = []
        print("Restructuring DataFrame...")
        # Iterate through each row in nasa_df
        for index, row in nasa_df.iterrows():
            cycle_number = row['cycle']
            cycle_type = row['type']
            ambient_temperature = row['ambient_temperature']
            time = row['time']  # Assuming time is also a list

            # Get the lists from the current row
            voltage_measured = row['Voltage_measured']
            current_measured = row['Current_measured']
            temperature_measured = row['Temperature_measured']
            current_charge = row['Current_charge']
            voltage_charge = row['Voltage_charge']
            time_data = row['Time']
            capacity_data=row['Capacity']

            # Use the length of the lists to determine how many new rows to create
            num_rows = len(voltage_measured)

            for i in range(num_rows):
                # Prepare the row data
                new_row = {
                    'cycle': cycle_number,
                    'type': cycle_type,
                    'ambient_temperature': ambient_temperature,
                    'time': time[i] if i < len(time) else None,  # Handle different lengths
                    'Voltage_measured': voltage_measured[i],
                    'Current_measured': current_measured[i],
                    'Temperature_measured': temperature_measured[i],
                    'Time': time_data[i] if i < len(time_data) else None,
                    'Capacity':capacity_data
                }

                # Set Current_charge and Voltage_charge based on the cycle type
                if cycle_type == 'discharge':
                    new_row['Current_charge'] = None
                    new_row['Voltage_charge'] = None
                else:
                    new_row['Current_charge'] = current_charge[i] if i < len(current_charge) else None
                    new_row['Voltage_charge'] = voltage_charge[i] if i < len(voltage_charge) else None

                # Append the new row to the data_rows list
                data_rows.append(new_row)

        # Create the new DataFrame from the collected rows
        nasa_df_final = pd.DataFrame(data_rows)



        # In[19]:



        #capacity is still valuable even though it has some missing values


        # In[17]:


        nasa_df_final['Step']=np.repeat(None,len(nasa_df_final['type']))
        nasa_df_candidate=nasa_df_final.drop(columns=['time','Current_charge','Voltage_charge'])


        # In[20]:


        


        #______part2______________
        #!/usr/bin/env python
        # coding: utf-8

        # In[1]:




        # Read the Parquet file
        nasa = nasa_df_candidate
        del(nasa_df_candidate)
        # Create a column to indicate the data source
        nasa['tag'] = np.repeat('NASA', np.shape(nasa)[0])
        print("Fixing Columns...")
        # Define columns of the NASA data frame
        nasa_cols = [
            'cycle', 'type', 'ambient_temperature', 'Voltage_measured',
            'Current_measured', 'Temperature_measured', 'Time',
            'Capacity', 'Step'
        ]

        # Function to convert a data frame to match the NASA data frame format
        def convert_to_nasa(columns, df, tag):
            # Determine the size of the data frame
            size = np.shape(df)[0]
            # Initialize a new data frame with the 'tag' column
            new_df = pd.DataFrame({'tag': np.repeat(tag, size)})

            # Populate columns based on the provided mapping
            for index in range(len(columns)):
                if columns[index] is None:
                    new_df[nasa_cols[index]] = np.repeat(None, size)
                else:
                    new_df[nasa_cols[index]] = df[columns[index]]

            return new_df


        # In[2]:



        example=pd.DataFrame({'time':[1,2,3,4],
                             'Temp(C)':[1,2,3,4],
                             'Current':[1,2,3,4],
                             'Voltage':[1,2,3,4],
                             'Capacity(mAH)':[1,2,3,4],
                             'Cycle':[1,2,3,4]})
        #Enter the columns in your data frame that correspond to the order printed
        # for example, the 6th column in this data frame 'Cycle' corresponds to the first column 'cycle' in the nasa dataframe
        # This data frame doesnt have ambient temperature so I just put 'None'
        new=convert_to_nasa(columns=['Cycle',None,None,'Voltage','Current','Temp(C)','time','Capacity(mAH)',None],
                            df=example,tag="Example")



        # In[3]:


        
        #___part3___
        #!/usr/bin/env python
        # coding: utf-8

        # In[1]:





        # #### Import Data

        # In[2]:


        NASA = nasa
        del(nasa)






        # #### Add a Source and Cell Column

        # In[4]:


        NASA['Source'] = 'NASA'


        # In[5]:


        NASA['Cell'] = cellnum


        # In[ ]:





        # #### Rename Columns for a Standard Format

        # In[6]:


        NASA.rename(columns={
            'cycle': 'Cycle',
            'type': 'Cycle_Type',
            'Temperature_measured': 'Temperature',
            'capacity': 'Capacity',
            'Voltage_measured': 'Voltage',
            'Current_measured': 'Current',
            'Time': 'Cycle Time',
            'ambient_temperature': 'Ambient_Temperature'
        }, inplace=True)


        # #### Review Columns and Shape

        # In[7]:





        # In[8]:





        # #### Aligning Columns per Group Discussion (all new columns are filled with NaN)

        # In[9]:


        target_columns = [
            'Source', 'Chemistry', 'Charge_Type', 'Cell', 'Cycle', 'Step', 'Cycle_Type', 'Time',
            'Voltage', 'Capacity', 'Charge_Capacity', 'Discharge_Capacity',
            'Temperature', 'Ambient_Temperature', 'Current'
        ]


        # In[10]:


        def align_columns(df, dataset_name, default_cell='1'):
            # Add missing columns with NaN
            for col in target_columns:
                if col not in df.columns:
                    df[col] = np.nan

            # Fill Dataset and Cell columns if they exist or with default values
            df['Source'] = dataset_name
            if 'Cell' not in df.columns or df['Cell'].isna().all():
                df['Cell'] = default_cell

            # Reorder columns
            df = df[target_columns + [col for col in df.columns if col not in target_columns]]

            return df


        # In[11]:


        NASA_aligned = align_columns(NASA, 'NASA')
        del(NASA)

        # #### Converting Units (time = seconds, voltage = volts, capacity columns = mAh, temperture columns = celsius, current = amps)

        # In[12]:

        print("Fixing Units")
        def convert_units(df):
            # Check if Time is in seconds, assuming it's in minutes in some datasets
            if df['Time'].max() > 3600:  # Assuming time is in minutes if values are large
                df['Time'] = df['Time'] * 60  # Convert from minutes to seconds

            # Check if Voltage is in millivolts (mV) and convert to volts (V)
            if df['Voltage'].max() > 100:  # If max voltage is higher than expected, assume it's in mV
                df['Voltage'] = df['Voltage'] / 1000  # Convert from millivolts to volts

            # Capacity, Charge_Capacity, and Discharge_Capacoty should all be in mAh
            # Assuming they are in Ah in some datasets
            if 'Capacity' in df.columns and df['Capacity'].max() < 1:  # If values are small, assume Ah
                df['Capacity'] = df['Capacity'] * 1000  # Convert from Ah to mAh
            if 'Charge_Capacity' in df.columns and df['Charge_Capacity'].max() < 1:
                df['Charge_Capacity'] = df['Charge_Capacity'] * 1000  # Convert from Ah to mAh
            if 'Discharge_Capacity' in df.columns and df['Discharge_Capacity'].max() < 1:
                df['Discharge_Capacity'] = df['Discharge_Capacity'] * 1000  # Convert from Ah to mAh

            # Current should be in amps (same logic as Voltage if in milliamps)
            if df['Current'].max() > 100:  # If max current is higher than expected, assume it's in mA
                df['Current'] = df['Current'] / 1000  # Convert from milliamps to amps

            return df


        # In[13]:


        NASA_converted = convert_units(NASA_aligned)


        # In[14]:





        # In[61]:


        NASA_df = NASA_converted
        del(NASA_converted)

        # In[ ]:





        # In[ ]:





        # In[63]:

        print("Changing Cycles to Charge and Discharge")
        cycle=1
        tp="charge"
        for row in range(len(NASA_df)):
            if NASA_df.loc[row,"Cycle_Type"]=="discharge" and tp=="charge":
                tp="discharge"
                cycle+=1
                
            if NASA_df.loc[row,"Cycle_Type"]=="charge" and tp=="discharge":
                tp="charge"
                cycle+=1
                
            NASA_df.loc[row,"Cycle"]=cycle



        # Dropping rows where cycle type is NaN (this throws off all time calculations going forward)

        # In[17]:





        # #### Calculating Voltage Bounds

        # In[64]:

        print("Adding Reference Capacity")
        voltage_bounds = NASA_df.groupby('Cycle')['Voltage'].agg(lower_bound='min',upper_bound= 'max')


        # In[65]:





        # In[66]:


        NASA_df = NASA_df.merge(voltage_bounds, on='Cycle', how='left')


        # In[67]:


        NASA_df = NASA_df[
            (NASA_df['Voltage'] >= NASA_df['lower_bound']) &
            (NASA_df['Voltage'] <= NASA_df['upper_bound'])


        ]


        # In[68]:





        # #### Calculating Instantaneous Capacity (didn't convert to seconds like Stephen becuase it is already converted)

        # In[69]:


        charge_cc = scipy.integrate.cumulative_trapezoid(NASA_df['Current'], NASA_df['Cycle Time'], initial=0)


        # In[70]:





        # In[71]:


        for (i, data) in NASA_df.groupby('Cycle'):
          NASA_df.loc[NASA_df.Cycle == i, 'Capacity' ] = scipy.integrate.cumulative_trapezoid(data['Current'], data['Cycle Time'], initial=0)


        # In[72]:





        # In[73]:





        # In[74]:





        # #### Calculating Reference Capacity
        # 
        # 
        # 

        # Creating two new columns for capacity at upper and lower bounds for each cycle

        # In[75]:


        def find_bound_capacities(group):
            # Find the capacity where voltage equals the upper bound
            upper_bound_capacity = group.loc[group['Voltage'] == group['upper_bound'], 'Capacity']
            # Find the capacity where voltage equals the lower bound
            lower_bound_capacity = group.loc[group['Voltage'] == group['lower_bound'], 'Capacity']

            # Check if we have at least one match for both bounds
            if not upper_bound_capacity.empty:
                upper_capacity = upper_bound_capacity.values[0]
            else:
                upper_capacity = None  # Handle cases where no match is found for upper bound

            if not lower_bound_capacity.empty:
                lower_capacity = lower_bound_capacity.values[0]
            else:
                lower_capacity = None  # Handle cases where no match is found for lower bound

            group['capacity_at_upper_bound'] = upper_capacity
            group['capacity_at_lower_bound'] = lower_capacity
            return group

        NASA_df = NASA_df.groupby('Cycle').apply(find_bound_capacities)




        # Finding 'Reference Capacity' by subtracting capacity at lower bound from capacity at upper bound.

        # In[76]:


        NASA_df['Reference Capacity'] = NASA_df['capacity_at_upper_bound'] - NASA_df['capacity_at_lower_bound']




        # Dropping Columns from dataframe

        # In[77]:


        NASA_df = NASA_df.drop(columns=['capacity_at_upper_bound', 'capacity_at_lower_bound', 'lower_bound', 'upper_bound', 'Step'])


        # Changing Capacity to Instantaneous Capacity, changing cycle to step, and Cycle Time to Step Time

        # In[78]:


        NASA_df.rename(columns={
            'Capacity': 'Instantaneous Capacity',
            'Cycle': 'Step',
            'Cycle Time': 'Step Time'

        }, inplace=True)


        # In[79]:


        NASA_df = NASA_df.reset_index()


        # Making Reference Capacity postitive

        # In[80]:


        NASA_df['Reference Capacity'] =NASA_df['Reference Capacity'].abs()


        # Creating Cycle using Instantaneous Capacity

        # In[35]:





        # Dropping Level 1

        # In[36]:


        NASA_df = NASA_df.drop(columns=['level_1'])


        # Creating Real Cycle Time

        # In[81]:


        NASA_df['Cycle Time'] = 0


        # In[38]:


        cycle_time = 0


        # In[82]:





        # In[39]:

        print("Adjusting Time")
        for i in range(1, len(NASA_df)):
            # Check if the cycle has changed; reset cycle time if it does
            if NASA_df.loc[i, 'Cycle'] != NASA_df.loc[i - 1, 'Cycle']:
                cycle_time = 0  # Reset cycle time for the new cycle

            # If Step Time is zero, ignore it and continue
            if NASA_df.loc[i, 'Step Time'] == 0:
                NASA_df.loc[i, 'Cycle Time'] = cycle_time
                continue

            # Calculate the difference between the current and previous Step Time
            step_diff = NASA_df.loc[i, 'Step Time'] - NASA_df.loc[i - 1, 'Step Time']

            # Accumulate the cycle time
            cycle_time += step_diff
            NASA_df.loc[i, 'Cycle Time'] = cycle_time


        # Adding Time

        # In[40]:

        print("Adjusting Step Time")
        NASA_df['Time'] = NASA_df['Step Time'].copy()
        for i in range(1, len(NASA_df)):
            # If 'Step Time' is 0, carry forward the last 'Time' value
            if NASA_df.loc[i, 'Step Time'] == 0:
                NASA_df.loc[i, 'Time'] = NASA_df.loc[i - 1, 'Time']
            else:
                # Otherwise, add the difference to the previous 'Time' value
                NASA_df.loc[i, 'Time'] = NASA_df.loc[i - 1, 'Time'] + (NASA_df.loc[i, 'Step Time'] - NASA_df.loc[i - 1, 'Step Time'])


        # Rounding Down Time

        # In[41]:

        print("Rounding")
        NASA_df['Time'] = NASA_df['Time'].round(3)


        # In[42]:


        NASA_df['Cycle Time'] = NASA_df['Cycle Time'].round(3)


        # In[43]:


        NASA_df = NASA_df.applymap(lambda x: np.floor(x * 1000) / 1000 if isinstance(x, (int, float)) else x)


        # Adding Continuous Charge

        # In[44]:


        NASA_df['Charge_Type'] = 'Continuous'


        # In[45]:





        # In[46]:


        NASA_df = NASA_df.drop(columns=['tag'])


        # #### Review Dataframe

        # In[47]:


        # Select only the numeric columns
        numeric_columns = NASA_df.select_dtypes(include='number')

        # Find the minimum and maximum values for each numeric column
        min_values = numeric_columns.min()
        max_values = numeric_columns.max()




        # Export Data

        # In[48]:

        print(f"Saving NASA_df as{outpath}") 
        NASA_df.to_parquet(outpath, index=False)
        #NASA_df.to_csv('NASA_Time_Series_Example.csv', index=False)
        print("Returning NASA_df")
        if ret==True:
            return NASA_df


        # In[49]:











