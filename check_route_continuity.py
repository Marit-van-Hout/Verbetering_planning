    # Something went wrong checking route continuity: 'omloop nummer'
    def check_route_continuity(bus_planning):
        """ Check if the endpoint of route n matches the start point of route n+1.
        Parameters:
            - bus_planning: DataFrame with route data.
        Output: Print messages if there are inconsistencies.
        """
    
      # Controleer op NaN-waarden in 'omloop nummer'
    if bus_planning['omloop nummer'].isna().any():
        st.error("NaN values found in 'omloop nummer' column.")
        return False

    # Controleer de continuïteit van de routes
    for i in range(len(bus_planning) - 1):
        current_end_location = bus_planning.at[i, 'eindlocatie']
        next_start_location = bus_planning.at[i + 1, 'startlocatie']
        omloop_nummer = bus_planning.at[i, 'omloop nummer']
        next_start_time = bus_planning.at[i + 1, 'starttijd'].time() # Haal de starttijd van de volgende route op

        if current_end_location != next_start_location:
            st.error(f"Route continuity issue between bus number {omloop_nummer:.0f} at {next_start_time}: "
                     f"ends at {current_end_location} and next route starts at {next_start_location}.")
        