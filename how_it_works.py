import streamlit as st

st.title("How It Works")
st.write("This is how the **Bus Planning Checker** works.")

# Gebruik Markdown voor een nettere opsomming
st.write("The app checks the following conditions:")

# Maak gebruik van een genummerde lijst met Markdown
st.markdown("""
1. **Battery Status**: Ensures the battery status is not under **10%** of the State of Health, which is **30 kWh**.
2. **Route Endpoint**: Verifies if the endpoint of route **n** matches the start point of the next route.
3. **Travel Time**: Confirms that the actual travel time is within the specified minimum and maximum travel time.
""")