# Functie om het busplanningsschema te valideren
#def validate_bus_planning(data):
    #errors = []
    
    # Voorbeeldcontroles
    # Controleer op ontbrekende waarden
    #if data.isnull().values.any():
        #errors.append("Er zijn lege waarden in het schema.")
        
    # Controleer of starttijd minder is dan eindtijd
    #if 'starttijd' in data.columns and 'eindtijd' in data.columns:
        #invalid_times = data[data['starttijd'] >= data['eindtijd']]
        #if not invalid_times.empty:
            #errors.append(f"Onjuiste tijden: {invalid_times[['starttijd', 'eindtijd']].to_string(index=False)}")

    # Verdere validatielogica kan hier worden toegevoegd

    #return errors
    
def start_day(line):
        """
        Deze functie kijkt naar de kolom 'startlocatie' in time_table. 
        Voor elke lijn wordt de tijd van een rit met materiaal (buslijn = NaN) vanuit 'ehvgar' 
        naar de startlocatie bepaald voor beide startlocaties ('ehvapt' en 'ehvbst').
    
        input: de lijn waarvoor de starttijden worden berekend (bijv. 400 of 401).
        output: de berekende starttijden worden toegevoegd aan de lijst start_tijden.
        """
        if line in time_table["buslijn"].values:
            # Verkrijg de startlocaties van de buslijn
            start_locaties = time_table.loc[time_table["buslijn"] == line, "startlocatie"].unique()

            for start_locatie in start_locaties:
                if start_locatie in ["ehvapt", "ehvbst"]:
                    # Maak een mask om de juiste rij uit de distance_matrix te filteren
                    mask = (
                        (distance_matrix["eindlocatie"] == start_locatie) &
                        (distance_matrix["startlocatie"] == "ehvgar") &
                        (distance_matrix["buslijn"].isna())
                    )
                    # Controleer of er resultaten zijn
                    if not distance_matrix[mask].empty:
                        # Verkrijg de reistijd
                        reistijd = distance_matrix.loc[mask, "min reistijd in min"].iloc[0]
                        # Converteer reistijd naar een Timedelta in minuten
                        reistijd_delta = pd.Timedelta(minutes=reistijd)

                        # Bereken de starttijd van de dag
                        start_day = time_table.loc[time_table["buslijn"] == line, "vertrektijd"].iloc[0] - reistijd_delta
                    
                        # Voeg de starttijd toe aan de lijst
                        start_tijden.append((line, start_locatie, start_day))  # Voeg lijn en locatie toe
                    else:
                        st.error(f"Row {index + 2}: No matching rides from ehvgar to {start_locatie} for line {line}")

                else:
                    st.error("Starting location not recognized:", start_locatie)
        else:
            st.error("No riseds found for bus line", line)

    def battery_consumption(distance, current_time, start_times, end_times):
        """Calculate battery consumption based on distance and current time."""
    
        # Assume max_capacity and consumption_per_km are defined globally
        battery_capacity = max_capacity * 0.9
        consumption = distance * np.mean(consumption_per_km)
        remaining_battery = battery_capacity - consumption
    
        # Get valid start and end times
        start_time = None
        end_time = None
    
        for line, locatie, tijd in start_times:
            if current_time >= tijd.time():
                start_time = tijd
    
        for line, locatie, tijd in end_times:
            if current_time >= tijd.time():
                end_time = tijd
    
        if start_time is None:
            st.error(f"No valid start time found for current time {current_time}")
        if end_time is None:
            st.error(f"No valid end time found for current time {current_time}")
    
        # Call the charging function to update the remaining battery
        return charging(remaining_battery, battery_capacity, current_time, start_times, end_times)

    def charging(battery, actual_capacity, current_time, start_times, end_time):
        """
        Simuleert het opladen van de batterij op basis van de huidige tijd en start- en eindtijden van de dienstregeling.
    
        Parameters:
            battery (float): Huidige batterijcapaciteit.
            actual_capacity (float): Totale capaciteit van de batterij.
            current_time (datetime.time): Huidige tijd in het schema.
            start_times (list): Lijst van tuples met (line, locatie, tijd) voor starttijden.
            end_time (list): Lijst van tuples met (line, locatie, tijd) voor eindtijden.
    
        Returns:
            float: Nieuwe batterijcapaciteit na opladen.
        
        Raises:
            ValueError: Als er geen geldige start- of eindtijd wordt gevonden voor de huidige tijd.
        """
    
        min_battery = 0.10 * actual_capacity
        max_battery_day = 0.90 * actual_capacity
        max_battery_night = actual_capacity
        charging_per_min = charging_speed_90
    
        # Zoek de juiste starttijd
        start_time = None
        for line, locatie, tijd in start_times:
            if current_time >= tijd.time():
                start_time = tijd
    
        # Zoek de juiste eindtijd
        end_time = None
        for line, locatie, tijd in end_time:
            if current_time >= tijd.time():
                end_time = tijd
    
        # Controleer of start_time en end_time zijn gevonden
        if start_time is None:
            raise ValueError(f'Geen geldige starttijd gevonden voor de huidige tijd: {current_time}')
        if end_time is None:
            raise ValueError(f'Geen geldige eindtijd gevonden voor de huidige tijd: {current_time}')
    
        # Bepaal maximum batterijlimiet op basis van de tijd
        if current_time < start_time.time() or current_time > end_time.time():
            max_battery = max_battery_night
        else:
            max_battery = max_battery_day

        # Bereken de nieuwe batterijcapaciteit
        charged_energy = min_idle_time * charging_per_min
        new_battery = battery + charged_energy if battery <= min_battery else battery
        return min(new_battery, max_battery)


    def battery_consumption(distance, current_time, start_times, end_time):
        """
        Bereken het batterijverbruik op basis van de afstand en huidige tijd.
    
        Parameters:
            distance (float): Afstand in kilometers.
            current_time (datetime.time): Huidige tijd in het schema.
            start_times (list): Lijst van tuples met (line, locatie, tijd) voor starttijden.
            end_time (list): Lijst van tuples met (line, locatie, tijd) voor eindtijden.
    
        Returns:
            float: Resterende batterijcapaciteit na verbruik en opladen.
        
        Raises:
            ValueError: Als er geen geldige start- of eindtijd wordt gevonden voor de huidige tijd.
        """
    
        # Bepaal batterijcapaciteit voor de dag
        battery_capacity = max_capacity * 0.9
    
        # Bereken het verbruik op basis van de afstand
        consumption = distance * np.mean(consumption_per_km)
        remaining_battery = battery_capacity - consumption
    
        # Zoek de juiste starttijd
        start_time = None
        for line, locatie, tijd in start_times:
            if current_time >= tijd.time():
                start_time = tijd
    
        # Zoek de juiste eindtijd
        end_time = None
        for line, locatie, tijd in end_time:
            if current_time >= tijd.time():
                end_time = tijd
    
        # Controleer of start_time en end_time zijn gevonden
        if start_time is None:
            raise ValueError(f'Geen geldige starttijd gevonden voor de huidige tijd: {current_time}')
        if end_time is None:
            raise ValueError(f'Geen geldige eindtijd gevonden voor de huidige tijd: {current_time}')
    
        # Roep de charging-functie aan om het resterende batterijpercentage bij te werken
        return charging(remaining_battery, battery_capacity, current_time, start_time, end_time)


    def check_route_continuity(bus_planning): # de bus kan niet vliegen
        """
        Check if the endpoint of route n matches the start point of route n+1.
        Parameters:
         - bus_planning: DataFrame with route data.
        Output: Print messages if there are inconsistencies.
        """
        for i in range(len(bus_planning) - 1): 
            current_end_location = bus_planning.iloc[i]['eindlocatie']
            next_start_location = bus_planning.iloc[i + 1]['startlocatie']
            if current_end_location != next_start_location:
                st.error(f'Route continuity issue between {bus_planning.iloc[i]['omloop nummer']:.0f} ending at {current_end_location} and next route starting at {next_start_location}.')
                return False
           
        return True

def remove_startingtime_endtime_equal(bus_planning): 
        """ If the starting time and end time are equal, the row is removed.
        Parameters: 
            bus_planning: DataFrame
                Whole bus planning.
        Output: DataFrame
            Clean DataFrame.
        """
        clean_bus_planning = bus_planning[bus_planning['starttijd'] != bus_planning['eindtijd']]
        return clean_bus_planning