import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns



# STREAMLIT CONFIGURATION
st.logo("tra_logo_rgb_HR.png", size='large')
page = 'Bus Planning Checker'  # Default page before any button is clicked

# SIDEBAR NAVIGATION
with st.sidebar:
    st.subheader('Navigation')
    
    if st.button("Bus Planning Checker", icon="🚍", use_container_width=True):
        page = 'Bus Planning Checker'
    if st.button('How It Works', icon="📖", use_container_width=True):
        page = 'How It Works'
    if st.button('Help', icon="❓", use_container_width=True):
        page = 'Help'

# VALIDITY FUNCTIONS
def check_battery_status(bus_planning, distance_matrix, SOH, min_SOC, consumption_per_km):
    """
    Validates battery status throughout the bus schedule and adds state of charge (SOC) as a column.
    
    Args:
        bus_planning (DataFrame): The bus schedule with 'starttijd', 'eindtijd', and other columns.
        distance_matrix (DataFrame): Distances between locations.
        SOH (float): State of Health of the battery as a percentage.
        min_SOC (float): Minimum state of charge required as a percentage.
        consumption_per_km (float): Energy consumption per kilometer in kWh.
    
    Returns:
        DataFrame: Rows from the schedule where battery status issues occur, with SOC as a column.
    """
    max_capacity = 300 * (SOH / 100)  # Max battery capacity in kWh
    min_battery = max_capacity * (min_SOC / 100)  # Minimum allowed battery level

    # Process times
    bus_planning['starttijd'] = pd.to_datetime(bus_planning['starttijd'], format='%H:%M')
    bus_planning['eindtijd'] = pd.to_datetime(bus_planning['eindtijd'], format='%H:%M')

    # Merge with distance matrix
    df = pd.merge(bus_planning, distance_matrix, on=['startlocatie', 'eindlocatie', 'buslijn'], how='left')

    # Calculate energy consumption
    df['consumption (kWh)'] = (df['afstand in meters'] / 1000) * max(consumption_per_km, 0.7)

    # Idle consumption
    df.loc[df['activiteit'] == 'idle', 'consumption (kWh)'] = 0.01

    # Charging speeds
    charging_speed_90 = 450 / 60  # Charging speed below 90% in kWh 
    charging_speed_10 = 60 / 60   # Charging speed above 90% in kWh 

    battery_level = max_capacity
    previous_loop_number = None

    issues = []
    state_of_charge = []  # Track SOC for each row

    for i, row in df.iterrows():
        next_start_time = bus_planning['starttijd'].iloc[i + 1] if i + 1 < len(bus_planning) else None

        # Reset battery for a new loop
        if row['omloop nummer'] != previous_loop_number:
            battery_level = max_capacity

        # Charging logic
        if row['activiteit'] == 'opladen':
            charging_duration = (row['eindtijd'] - row['starttijd']).total_seconds() / 60  # Charging time in minutes
            charge_power = (charging_speed_90 if battery_level <= (max_capacity * 0.9) else charging_speed_10) * charging_duration
            battery_level = min(battery_level + charge_power, max_capacity)
        else:
            battery_level -= row['consumption (kWh)']  # Subtract consumption for non-charging activities

        battery_level = max(battery_level, 0)  # Ensure battery level is not negative
        state_of_charge.append(battery_level / max_capacity * 100)  # Append SOC as percentage

        # Log issues if battery falls below minimum SOC
        if battery_level < min_battery:
            issues.append(row)

        previous_loop_number = row['omloop nummer']

    df['state_of_charge'] = state_of_charge  # Add SOC as a column

    # Filter out rows with issues
    if not issues:
        return pd.DataFrame()  # Return empty DataFrame if no issues

    failed_df = pd.DataFrame(issues)
    failed_df['state of charge'] = df.loc[failed_df.index, 'state_of_charge']  # Add SOC to issue rows
    required_columns = ['omloop nummer', 'starttijd', 'consumption (kWh)', 'state of charge']
    
    missing_columns = set(required_columns) - set(failed_df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in output DataFrame: {missing_columns}")

    return failed_df[required_columns]

def check_route_continuity(bus_planning):
    """
    Checks for route continuity issues within the same loop number.
    Args:
        bus_planning (DataFrame): The bus schedule.
    Returns:
        DataFrame: Rows with route continuity issues.
    """
    issues = []

    required_columns = {'omloop nummer', 'startlocatie', 'eindlocatie', 'starttijd'}
    if not required_columns.issubset(bus_planning.columns):
        missing_columns = required_columns - set(bus_planning.columns)
        st.error(f"Missing columns in 'bus_planning': {missing_columns}")
        return pd.DataFrame()

    bus_planning = bus_planning.sort_values(by=['omloop nummer', 'starttijd']).reset_index(drop=True)

    for i in range(len(bus_planning) - 1):
        current_row = bus_planning.iloc[i]
        next_row = bus_planning.iloc[i + 1]

        if (current_row['omloop nummer'] == next_row['omloop nummer']) & (current_row['eindlocatie'] != next_row['startlocatie']):
            issues.append({
                'omloop nummer': current_row['omloop nummer'],
                'current end location': current_row['eindlocatie'],
                'next start location': next_row['startlocatie'],
                'current end time': current_row['eindtijd'],
                'next start time': next_row['starttijd'],
                })

    return pd.DataFrame(issues)

def driven_rides(bus_planning):
    """
    Filters bus planning data for rides that include a bus line.
    Args:
        bus_planning (DataFrame): The bus schedule.
    Returns:
        DataFrame: Filtered DataFrame with rides containing a bus line.
    """
    return bus_planning[['startlocatie', 'starttijd', 'eindlocatie', 'buslijn']].dropna(subset=['buslijn'])

def every_ride_covered(bus_planning, timetable):
    """
    Checks if every Trip in the bus planning is covered in the timetable.
    Args:
        bus_planning (DataFrame): Planned rides.
        timetable (DataFrame): Timetable rides.
    Returns:
        DataFrame: Discrepancies between planning and timetable.
    """
    if 'vertrektijd' in timetable.columns:
        timetable = timetable.rename(columns={'vertrektijd': 'starttijd'})

    if 'starttijd' not in bus_planning.columns or 'starttijd' not in timetable.columns:
        st.error("Missing 'starttijd' column in bus planning or timetable.")
        return pd.DataFrame()

    bus_planning['starttijd'] = pd.to_datetime(bus_planning['starttijd'], errors='coerce')
    timetable['starttijd'] = pd.to_datetime(timetable['starttijd'], errors='coerce')

    differences = bus_planning.merge(
        timetable, on=['startlocatie', 'starttijd', 'eindlocatie', 'buslijn'], how='outer', indicator=True
    )

    issues = differences.query('_merge != "both"')

    if not issues.empty:
        return issues[['omloop nummer', 'startlocatie', 'activiteit', 'starttijd']]

    return pd.DataFrame(columns=['omloop nummer', 'startlocatie', 'activiteit', 'starttijd'])

def check_travel_time(bus_planning, distance_matrix):
    """
    Validates that travel times are within expected ranges.
    Args:
        bus_planning (DataFrame): Planned rides.
        distance_matrix (DataFrame): Expected travel time data.
    Returns:
        DataFrame: Discrepancies in travel times.
    """
    print("Initial columns:", bus_planning.columns)

    # Strip column names to avoid leading/trailing spaces
    bus_planning.columns = bus_planning.columns.str.strip()

    if 'starttijd' not in bus_planning.columns or 'eindtijd' not in bus_planning.columns:
        print("Missing columns:", bus_planning.columns)
        return pd.DataFrame()

    # Validate and convert to datetime
    try:
        bus_planning['starttijd'] = pd.to_datetime(bus_planning['starttijd'], format='%H:%M:%S')
        bus_planning['eindtijd'] = pd.to_datetime(bus_planning['eindtijd'], format='%H:%M:%S')
    except Exception as e:
        print("Datetime conversion error:", e)
        return pd.DataFrame()

    print("After datetime conversion:")
    print(bus_planning[['starttijd', 'eindtijd']].info())

    bus_planning['difference_in_minutes'] = (
        bus_planning['eindtijd'] - bus_planning['starttijd']
    ).dt.total_seconds() / 60

    # Merge and validate times
    merged_df = pd.merge(bus_planning, distance_matrix, on=['startlocatie', 'eindlocatie', 'buslijn'], how='inner')

    issues = []

    for _, row in merged_df.iterrows():
        if not (row['min reistijd in min'] <= row['difference_in_minutes'] <= row['max reistijd in min']):
            issues.append({
                'omloop nummer': row.get('omloop nummer', None),
                'startlocatie': row['startlocatie'],
                'eindlocatie': row['eindlocatie'],
                'reistijd': row['difference_in_minutes'],
                'starttijd': row['starttijd']
            })

    return pd.DataFrame(issues)

# VISUALISATION FUNCTION
def plot_schedule_from_excel(bus_planning):
    """Plot a Gantt chart for bus scheduling based on a DataFrame."""
    required_columns = ['starttijd', 'eindtijd', 'buslijn', 'omloop nummer', 'activiteit']
    if not all(col in bus_planning.columns for col in required_columns):
        st.error("One or more necessary columns are missing in bus planning.")
        return

    # Convert time columns to datetime
    bus_planning['starttijd'] = pd.to_datetime(bus_planning['starttijd'], errors='coerce')
    bus_planning['eindtijd'] = pd.to_datetime(bus_planning['eindtijd'], errors='coerce')

    # Drop rows with invalid datetime
    bus_planning = bus_planning.dropna(subset=['starttijd', 'eindtijd'])

    # Calculate duration in hours with a minimum value
    bus_planning['duration'] = ((bus_planning['eindtijd'] - bus_planning['starttijd']).dt.total_seconds() / 3600).clip(lower=0.05)

    # Map colors for different activities and bus lines
    color_map = {
        '400.0': 'blue',
        '401.0': 'yellow',
        'materiaal rit': 'green',
        'idle': 'red',
        'opladen': 'orange'
    }
    bus_planning['buslijn'] = bus_planning['buslijn'].astype(str)

    # Determine color per row
    def determine_color(row):
        return color_map.get(row['buslijn'], color_map.get(row['activiteit'], 'gray'))

    bus_planning['color'] = bus_planning.apply(determine_color, axis=1)

    # Plot Gantt chart
    fig, ax = plt.subplots(figsize=(12, 6))
    omloopnummers = bus_planning['omloop nummer'].unique()
    omloop_indices = {omloop: i for i, omloop in enumerate(omloopnummers)}

    for omloop, omloop_index in omloop_indices.items():
        trips = bus_planning[bus_planning['omloop nummer'] == omloop]

        if trips.empty:
            ax.barh(omloop_index, 1, left=0, color='black', edgecolor='black')  # Placeholder bar
            continue

        for _, trip in trips.iterrows():
            ax.barh(
                omloop_index, 
                trip['duration'], 
                left=trip['starttijd'].hour + trip['starttijd'].minute / 60, 
                color=trip['color'], 
                edgecolor='black'
            )

    # Add labels and legend
    ax.set_yticks(list(omloop_indices.values()))
    ax.set_yticklabels(list(omloop_indices.keys()))
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Bus Number')
    ax.set_title('Gantt Chart for Bus Scheduling')

    legend_elements = [
        Patch(facecolor=color_map['400.0'], edgecolor='black', label='Regular trip 400'),
        Patch(facecolor=color_map['401.0'], edgecolor='black', label='Regular trip 401'),
        Patch(facecolor=color_map['materiaal rit'], edgecolor='black', label='Deadhead trip'),
        Patch(facecolor=color_map['idle'], edgecolor='black', label='Idle'),
        Patch(facecolor=color_map['opladen'], edgecolor='black', label='Charging')
    ]
    ax.legend(handles=legend_elements, title='Legend')

    st.pyplot(fig)


def plot_activity_pie_chart(df):
    """
    Toont een cirkeldiagram van de verdeling van activiteiten in de totale planning.
    """
    df['starttijd'] = pd.to_datetime(df['starttijd'], format='%H:%M:%S', errors='coerce')
    df['eindtijd'] = pd.to_datetime(df['eindtijd'], format='%H:%M:%S', errors='coerce')

    # Bereken de tijdsduur voor elke activiteit
    df['duur'] = (df['eindtijd'] - df['starttijd']).dt.total_seconds() / 3600

    # Groepeer de data per activiteit en bereken de totale tijd per activiteit
    stapel_data = df.groupby('activiteit')['duur'].sum().reset_index()

    # Voeg ontbrekende activiteiten toe
    activiteit_labels = ['opladen', 'idle']
    for label in activiteit_labels:
        if label not in stapel_data['activiteit'].values:
            stapel_data = stapel_data.append({'activiteit': label, 'duur': 0}, ignore_index=True)

    nieuwe_labels = ['Regular Trip', 'Idle', 'Deadhead Trip', 'Charging']

    # Maak de figuur en plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        stapel_data['duur'], 
        labels=None, 
        autopct=lambda pct: f'{pct:.1f}%', 
        startangle=90, 
        colors=['blue', 'red', 'green', 'orange'], 
        textprops={'fontsize': 14, 'fontweight': 'bold'}
    )
    ax.legend(nieuwe_labels, loc="best")
    ax.set_title('Distribution of Activities in the Total Planning')
    st.pyplot(fig)

def plot_charging_heatmap(df):
    """
    Toont een heatmap van de activiteit 'opladen' per uur van de dag.
    """
    df['starttijd'] = pd.to_datetime(df['starttijd'], format='%H:%M:%S', errors='coerce')
    df['uur'] = df['starttijd'].dt.hour
    opladen_df = df[df['activiteit'] == 'opladen']
    heatmap_data = opladen_df['uur'].value_counts().reindex(range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_data.values.reshape(1, -1), 
        cmap='YlGnBu', 
        annot=True, 
        cbar=True, 
        ax=ax, 
        xticklabels=[f"{hour}:00" for hour in range(24)], 
        yticklabels=["Opladen"]
    )
    ax.set_title('Heatmap of Activity "Charging" per Hour')
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Activity')
    st.pyplot(fig)

def plot_activity_bar_chart(df):
    """
    Toont een stapelgrafiek van de totale tijd per activiteit.
    """
    df['starttijd'] = pd.to_datetime(df['starttijd'], format='%H:%M:%S', errors='coerce')
    df['eindtijd'] = pd.to_datetime(df['eindtijd'], format='%H:%M:%S', errors='coerce')
    df['duur'] = (df['eindtijd'] - df['starttijd']).dt.total_seconds() / 3600

    stapel_data = df.groupby('activiteit')['duur'].sum().reset_index()

    activiteit_labels = ['opladen', 'idle']
    for label in activiteit_labels:
        if label not in stapel_data['activiteit'].values:
            stapel_data = stapel_data.append({'activiteit': label, 'duur': 0}, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(stapel_data['activiteit'], stapel_data['duur'], color=['blue', 'red', 'green', 'orange'])
    nieuwe_labels = ['Regular Trip', 'Idle', 'Deadhead Trip', 'Charging']
    ax.set_xticks(range(len(nieuwe_labels)))
    ax.set_xticklabels(nieuwe_labels, fontsize=12)
    ax.set_title('Total Time per Activity')
    ax.set_xlabel('Activity')
    ax.set_ylabel('Total Time (Hours)')
    st.pyplot(fig)

# KPI FUNCTIONS
def count_buses(bus_planning):
    """Count unique 'omloop nummer' values in bus planning."""
    if 'omloop nummer' not in bus_planning.columns:
        raise ValueError("'omloop nummer' column not found in the data.")
    
    valid_omloop = bus_planning['omloop nummer'].dropna()
    return valid_omloop.nunique()

def calculate_deadhead_time(bus_planning):
    """Calculate total time spent on deadhead trips in minutes."""
    required_columns = ['starttijd datum', 'eindtijd datum', 'activiteit']
    if not all(col in bus_planning.columns for col in required_columns):
        raise ValueError("Required columns for deadhead time calculation are missing.")
    
    deadhead_trips = bus_planning[bus_planning['activiteit'] == 'materiaal rit']
    deadhead_trips['starttijd datum'] = pd.to_datetime(deadhead_trips['starttijd datum'], errors='coerce')
    deadhead_trips['eindtijd datum'] = pd.to_datetime(deadhead_trips['eindtijd datum'], errors='coerce')

    deadhead_trips['duration_minutes'] = (deadhead_trips['eindtijd datum'] - deadhead_trips['starttijd datum']).dt.total_seconds() / 60
    return round(deadhead_trips['duration_minutes'].sum(), 0)

def calculate_energy_consumption(bus_planning, distance_matrix, consumption_per_km):
    """Calculates the total energy consumed for the bus planning.

    Args:
        bus_planning (DataFrame): The bus schedule with details of trips and activities.
        distance_matrix (DataFrame): Matrix with distances between locations.
        consumption_per_km (float): Energy consumption per kilometer in kWh.

    Returns:
        float: Total energy consumed in kWh.
    """
    # Merge with distance matrix
    df = pd.merge(bus_planning, distance_matrix, on=['startlocatie', 'eindlocatie', 'buslijn'], how='left')

    # Calculate energy consumption (kWh)
    df['consumption (kWh)'] = (df['afstand in meters'] / 1000) * max(consumption_per_km, 0.7)

    # Adjust consumption for idle activities
    df.loc[df['activiteit'] == 'idle', 'consumption (kWh)'] = 0.01

    # Calculate total energy consumption
    total_energy_consumed = round(df['consumption (kWh)'].sum(), 0)

    return total_energy_consumed

# PAGINA'S DEFINIEREN
def bus_checker_page(): 
    st.header("Bus Planning Checker")

    tab1, tab2, tab3 = st.tabs(['Data and Parameters', 'Validity Checks', 'Your Data'])
    
    with tab1:
        # File uploaders
        st.subheader('Data')
        col1, col2 = st.columns(2)
    
        with col1:
            uploaded_file = st.file_uploader("Upload Your **Bus Planning** Here", type="xlsx")
        with col2:
            given_data = st.file_uploader("Upload Your **Timetable** Here", type="xlsx")
        
        st.subheader('Parameters')
        SOH =                   st.slider("**State Of Health** - %", 85, 95, 90)
        min_SOC =               st.slider("**Minimum State Of Charge** - %", 5, 25, 10)
        consumption_per_km =    st.slider("**Battery Consumption Per KM** - KwH", 0.7, 2.5, 1.6)

    with tab3:
        # Check if the required files are uploaded
        if not uploaded_file or not given_data:
            st.error("You need to upload your data in the 'Data and Parameters' tab.")
            return  # Stop execution if files are not uploaded
        
        if uploaded_file and given_data:
            with st.spinner('Your data is being processed...'): 
                try:
                    bus_planning = pd.read_excel(uploaded_file)
                    timetable = pd.read_excel(given_data, sheet_name='Dienstregeling')
                    distance_matrix = pd.read_excel(given_data, sheet_name="Afstandsmatrix")
                except Exception as e:
                    st.error(f"Error reading Excel files: {str(e)}")
                    return

                st.write('Your Bus Planning:')
                st.dataframe(bus_planning, hide_index=True)

                st.write('Gantt Chart Of Your Bus Planning:')
                plot_schedule_from_excel(bus_planning) 

                st.write("")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("Pie Chart:")
                    plot_activity_pie_chart(bus_planning)

                with col2:
                    st.write("Heatmap:")
                    plot_charging_heatmap(bus_planning)

                with col3:
                    st.write("Bar Chart:")
                    plot_activity_bar_chart(bus_planning)
            
                if bus_planning.empty or timetable.empty or distance_matrix.empty:
                    st.error("One or more DataFrames are empty. Please check the uploaded files.")
                    return
                
                st.write('(Click on the graph to expand)')

    with tab2:
            # Dislay KPIs
            st.subheader('KPIs')
            met_col1, met_col2, met_col3 = st.columns(3)

            try:
                buses_used = count_buses(bus_planning)  
                met_col1.metric('Total Buses Used', buses_used, delta=(buses_used - 20), delta_color="inverse")
            except Exception as e:
                st.error(f'Something went wrong displaying buses: {str(e)}')

            try:
                deadhead_minutes = calculate_deadhead_time(bus_planning)  
                met_col2.metric('Total Deadhead Trips In Minutes', deadhead_minutes)
            except Exception as e:
                st.error(f'Something went wrong displaying deadhead time: {str(e)}')
            
            try: 
                energy_cons = calculate_energy_consumption(bus_planning, distance_matrix, consumption_per_km)
                met_col3.metric('Total Energy Consumed in kW', energy_cons)
            except Exception as e:
                st.error(f'Something went wrong displaying energy consumption: {str(e)}')
                
            st.divider()
            
            # Check Batterij Status
            st.subheader('Battery Status')
            try: 
                battery_problems = check_battery_status(bus_planning, distance_matrix, SOH, min_SOC, consumption_per_km)
                if battery_problems.empty:
                    st.write('No problems found!')
                else:
                    st.markdown(':red[Battery dips below minimum State Of Charge]')
                    with st.expander('Click to see the affected rows'):
                        st.dataframe(battery_problems)       
            except Exception as e:
                st.error(f'Something went wrong checking battery: {str(e)}')
            
            # Check Route Continuiteit
            st.subheader('Route Continuity')
            try:
                continuity_problems = check_route_continuity(bus_planning)
                if continuity_problems.empty:
                    st.write('No problems found!')
                else:
                    st.markdown(':red[Start and end location do not line up]')
                    with st.expander('Click to see the affected rows'):
                        st.dataframe(continuity_problems)
            except Exception as e:
                st.error(f'Something went wrong checking route continuity: {str(e)}')

            # Gereden Ritten
            try:
                bus_planning = driven_rides(bus_planning)
            except Exception as e:
                st.error(f'Something went wrong checking driven rides: {str(e)}')

            # Iedere Nodige Rit Wordt Gereden
            st.subheader('Trip Coverage')
            try:
                ride_coverage = every_ride_covered(bus_planning, timetable)
                if ride_coverage.empty:
                    st.write('No problems found!')
                else:
                    st.markdown(':red[Some trips included in timetable are not present in bus pallning, or vice versa]')
                    with st.expander('Click to see the affected rows'):
                        st.dataframe(ride_coverage)             
            except Exception as e:
                st.error(f'Something went wrong checking if each trip is covered: {str(e)}')

            # Check Reistijd
            st.subheader('Travel Time')
            try:
                travel_time = check_travel_time(bus_planning, distance_matrix)
                if travel_time.empty:
                    st.write('No problems found!')
                else:
                    st.markdown(':red[Travel time outside of bound as specified in distance matrix]')
                    with st.expander('Click to see the affected rows'):
                        st.dataframe(travel_time)  
            except Exception as e:
                st.error(f'Something went wrong checking the travel time: {str(e)}')
                       
def how_it_works_page():
    st.header("How It Works")

    st.write("**The app checks the following conditions:**")

    st.markdown("""
    1. **Battery Status**: the app checks and ensures that the battery level of the bus does not drop below **10%** of the State of Health, which is **30 kWh**. 
    The system accounts for both driving and idle time consumption and models charging times at two rates: a higher rate for charging up to **90%** and a slower rate beyond that. 

    2. **Route Continuity**: the app checks that the endpoint of each route aligns with the starting location of the following route to maintain continuity in the bus's journey. 

    3. **Travel Time**: the app confirms that the travel time for each route falls within the predefined range. 

    4. **Coverage of Scheduled Rides**: the app ensures that every Trip listed in the **timetable** is matched in the **bus planning** records. 

    5. **Data Consistency**: the app verifies that all critical columns are present in your data. 

    6. **Error Reporting**: in cases where errors or discrepancies are found, the app provides detailed error messages. These messages include specific 
    information about the issue, such as route numbers, times, and locations, allowing for easy adjusting.
    """)
    
def help_page():
    st.header("Help")

    tab1, tab2, tab3 = st.tabs(['How To Use', 'Troubleshooting', 'Error Interpretation'])

    with tab1:
        st.subheader("**Need assistance?**")
        st.write("**This is how to use the app**")

        st.markdown("""1. Go to the navigation panel and select ‘Bus Planning Checker’.""")
        st.image('Picture1.png', width=200)

        st.markdown("""2. You should be presented with the following page. Here you can upload your bus planning and your timetable.""")
        st.image('Picture2.png', width=600)

        st.markdown("""
        ---
        Note:
        - Do not refresh the page after uploading files, this will clear all data
        - Follow the correct upload sequence  to ensure accurate results
        - Both files must be .xlsx files
        ---
        3. Results appear on the same page after uploading both files. You will find:
            - The uploaded bus planning for easy viewing and verification
            - A visualization of the planning to help you identify issues at a glance.
            - A list of detected issues or inconsistencies in your planning""")
        st.image('Picture3.png', width=400)

    with tab2:
        st.subheader("**Troubleshooting**")
        st.markdown("""
        **Things to do if you are having trouble uploading your files**
        - Ensure that the files are .xlsx files. Any other file format will not work
        - Verify that you uploaded the files in the correct order. The bus planning at the top, the timetable at the bottom
        - Verify that the files are complete and contain all required fields. Missing data or headers may result in errors during analysis
        - If the issue persists, try refreshing the page and re-uploading the files""")

    with tab3:
        st.subheader("**Error interpretation**")
        st.write("**Not sure what an error means? Here you can find some more explanation**")
        st.markdown("""
        - **Battery under minimum threshold detected**: check route timing and ensure that sufficient charging time is allocated.
        - **Route continuity issue found**: ensure that the endpoint of the previous route matches the start location of the next route.
        - **Some rides may be missing bus line entries**: Make sure all routes are clearly labeled with their bus lines to avoid mismatches.
        - **Inconsistencies found between bus planning and timetable data**: ensure that all timetable rides are included in the bus planning and vice versa.
        - **Missing start time column in either bus planning or timetable file**: verify both files contain start times for accurate matching.
        - **The calculated travel time for bus line from start location to end location is outside the expected range**: check timing and distance data to ensure accuracy.
        - **Invalid start or end time detected**: check entries for accurate time formats (HH:MM:SS) and ensure times are complete.
        - **Essential columns are missing in the bus planning data**: confirm that all rides have start and end times for reliable analysis.""")


# PAGE SELECTOR
if page == 'Bus Planning Checker':
    bus_checker_page()
elif page == 'How It Works':
    how_it_works_page()
elif page == 'Help':
    help_page()