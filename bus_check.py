import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from io import StringIO
from datetime import datetime, timedelta

# We are creating an app that checks a trip schedule and verifies if it meets all constraints. 
# If not, an error will be shown saying: Sorry, but your stupid file doesn't work. Here's why: .... What an idiot.

circuit_planning = pd.read_excel('trip_schedule.xlsx')

# Parameters
max_capacity = 300  # Maximum capacity in kWh
SOH = [85, 95]  # State of Health
charging_speed_90 = 450 / 60  # kWh per minute for charging up to 90%
charging_time_10 = 60 / 60  # kWh per minute for charging from 90% to 100%
actual_capacity = max_capacity * 0.9
daytime_limit = actual_capacity * 0.9
consumption_per_km = (0.7 + 2.5) / 2  # kWh per km
min_idle_time = 15

# Battery charging simulation
def charging(battery, actual_capacity, current_time, start_time, end_time):
    """Charge the battery based on the current time and bus schedule."""
    min_battery = 0.10 * actual_capacity
    max_battery_day = 0.90 * actual_capacity
    max_battery_night = actual_capacity
    charging_per_min = charging_speed_90

    if current_time < start_time or current_time > end_time:
        max_battery = max_battery_night
    else:
        max_battery = max_battery_day

    charged_energy = min_idle_time * charging_per_min
    new_battery = battery + charged_energy if battery <= min_battery else battery
    return min(new_battery, max_battery)

# Function to calculate battery status
def simulate_battery(circuit_planning, actual_capacity, start_time, end_time):
    """
    Simulate the battery status during the trip schedule.
    Parameters:
        - circuit_planning: DataFrame with the trip schedule.
        - actual_capacity: Battery capacity of the bus.
        - start_time: First departure time of the regular trip.
        - end_time: Last end time of the regular trip.
    Output: Battery percentage after the simulation.
    """
    battery = actual_capacity * 0.9  # Start with 90% battery
    min_battery = actual_capacity * 0.1  # Minimum battery percentage
    max_battery_day = actual_capacity * 0.9  # Maximum 90% during the day
    max_battery_night = actual_capacity  # Maximum 100% at night
    min_charging_time = 15  # Minimum 15 minutes of charging

    for i, row in circuit_planning.iterrows():
        # Convert start and end times to datetime
        start_time = datetime.strptime(row['starttijd'], '%H:%M:%S')
        end_time = datetime.strptime(row['eindtijd'], '%H:%M:%S')

        # Check if the trip is a regular trip or deadhead trip
        if row['activiteit'] in ['dienst rit', 'materiaal rit']:
            consumption = row['energieverbruik']
            battery -= consumption

            # Check if battery status has dropped below 10%
            if battery < min_battery:
                print(f"Warning: Battery too low after trip {row['buslijn']} at {row['starttijd']} from {row['startlocatie']} to {row['eindlocatie']}.")
                return battery  # Stop simulation if battery is too low

        # Check if the bus is idle and has enough time to charge
        if row['activiteit'] == 'opladen':
            idle_start_time = datetime.strptime(row['starttijd'], '%H:%M:%S')
            idle_end_time = datetime.strptime(row['eindtijd'], '%H:%M:%S')
            idle_time = (idle_end_time - idle_start_time).total_seconds() / 60  # Idle time in minutes

            # Check if the idle time is at least 15 minutes
            if idle_time >= min_charging_time:
                battery = charging(battery, actual_capacity, idle_start_time, start_time, end_time)
            else:
                print(f"Warning: Charging time too short from {row['startlocatie']} to {row['eindlocatie']} at {row['starttijd']} to {row['eindtijd']}, only {idle_time} minutes.")

        # Check battery status after each step
        if battery < min_battery:
            print(f"Warning: Battery too low after {row['starttijd']}.")
            break

    return battery

# Function to check route continuity
def check_route_continuity(circuit_planning):
    """
    Check if the endpoint of trip n matches the start point of trip n+1.
    Parameters:
        - circuit_planning: DataFrame with trip data.
    Output: Print messages if there are inconsistencies.
    """
    for i in range(len(circuit_planning) - 1):
        current_end_location = circuit_planning.iloc[i]['eindlocatie']
        next_start_location = circuit_planning.iloc[i + 1]['startlocatie']
        if current_end_location != next_start_location:
            print(f"Warning: Route continuity issue between {circuit_planning.iloc[i]['buslijn']} ending at {current_end_location} and next trip starting at {next_start_location}.")
            return False
    return True

st.title("🚌 Bus Planning Checker")
st.write(
    "Instantly validate your circulation planning for compliance!"
)

# File upload (CSV or Excel)
uploaded_file = st.file_uploader("Drag and drop your circulation planning (CSV or Excel file)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read the uploaded file (CSV or Excel)
        if uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_csv(uploaded_file)
        
        st.write("**Your Bus Planning:**")
        st.dataframe(data)
        
        # Validation function (example)
        def validate_schema(df):
            errors = []
            # Check for missing values
            if data.isnull().values.any():
                errors.append("There are missing values in the bus planning.")
                
            # Check if battery status is above 10% after the last trip
            final_battery = simulate_battery(data, actual_capacity, start_time, end_time)
            if final_battery < (actual_capacity * 0.1):
                errors.append("The battery is below 10% at the end of the trip.")
            
            # Check charging time
            for i, row in data.iterrows():
                if row['activiteit'] == 'opladen':
                    idle_start_time = datetime.strptime(row['starttijd'], '%H:%M:%S')
                    idle_end_time = datetime.strptime(row['eindtijd'], '%H:%M:%S')
                    idle_time = (idle_end_time - idle_start_time).total_seconds() / 60  # Idle time in minutes
                    if idle_time < 15:
                        errors.append(f"Charging time is too short in trip {row['buslijn']} from {row['startlocatie']} to {row['eindlocatie']}.")

            # Check teleportation
            if data.iloc[-1]['eindlocatie'] != data.iloc[0]['startlocatie']:
                errors.append("The endpoint of the last trip does not match the start location of the next trip.")
            
            # Check travel times, coverage, and idle time
            for i in range(len(data) - 1):
                current_end_location = data.iloc[i]['eindlocatie']
                next_start_location = data.iloc[i + 1]['startlocatie']
                # Check if the endpoint of the current trip matches the start location of the next trip
                if current_end_location != next_start_location:
                    errors.append(f"Route continuity problem between trip {data.iloc[i]['buslijn']} and trip {data.iloc[i + 1]['buslijn']}.")
            
            return errors
        
        # Perform validation
        validation_errors = validate_schema(data)
        
        if validation_errors:
            st.error("File processed successfully. A few mistakes were found in the bus planning:")
            for error in validation_errors:
                st.error(error)
        else:
            st.success("File processed successfully. No errors found in the bus planning.")
                     
            # Optional visualization
            st.write("**Visualization of the Trip Schedule:**")
            fig, ax = plt.subplots()
            ax.scatter(data['speed'], data['energy'])
            ax.set_xlabel('Speed (km/h)')
            ax.set_ylabel('Energy Consumption (kWh)')
            ax.set_title('Speed vs Energy Consumption')
            st.pyplot(fig)
    
    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")
def plot_schedule(scheduled_orders):
    """Plots a Gantt chart of the scheuled orders

    Args:
        scheduled_orders (dict): every order, their starting time, end time, on which machine and set-up time
        method (str): method used to calculate schedule
    """    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = 0
    
    # Colors for visualization
    color_map = {
         '400': 'blue',
       '401': 'yellow',
    }
    for machine, orders in scheduled_orders.items():
        y_pos += 1  # Voor elke machine
        for order in orders:
            order_color = order['colour']
            processing_time = order['end_time'] - order['start_time'] - order['setup_time']
            setup_time = order['setup_time']
            start_time = order['start_time']
            
            # Teken verwerkingstijd
            ax.barh(y_pos, processing_time, left=start_time + setup_time, color=color_map[order_color], edgecolor='black')
            ax.text(start_time + setup_time + processing_time / 2, y_pos, f"Order {order['order']}", ha='center', va='center', color='black', rotation=90)

            # Teken setup tijd
            if setup_time > 0:
                ax.barh(y_pos, setup_time, left=start_time, color='gray', edgecolor='black', hatch='//')
    
    ax.set_yticks(range(1, len(scheduled_orders) + 1))
    ax.set_yticklabels([f"Machine {m}" for m in scheduled_orders.keys()])
    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_title(f'Gantt Chart for Paint Shop Scheduling')
    plt.show()