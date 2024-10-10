import streamlit as st
import pandas as pd

# Display the logo
st.image("logo_transdev_klein.png", width=200)

# Define pages
def bus_checker_page():
    st.title("🚌 Bus Planning Checker")
    st.write("Deze pagina stelt je in staat om het busplanningsschema te controleren.")

    # Bestand uploaden
    uploaded_file = st.file_uploader("Upload een Excel-bestand (xlsx)", type=["xlsx"])

    if uploaded_file is not None:
        try:
            # Probeer het Excel-bestand te lezen
            data = pd.read_excel(uploaded_file)
            st.write("Geüpload bestand:")
            st.dataframe(data)

            # Valideer de data
            validation_errors = validate_bus_planning(data)

            if validation_errors:
                st.error("Er zijn fouten gevonden in het oploopschema:")
                for error in validation_errors:
                    st.error(error)
            else:
                st.success("Het oploopschema is geldig!")
        except Exception as e:
            st.error(f"Fout bij het uploaden of lezen van het bestand: {str(e)}")

def how_it_works_page():
    st.title("📖 How It Works")
    st.write("Deze sectie legt uit hoe de applicatie werkt.")

def help_page():
    st.title("❓ Help")
    st.write("Deze pagina biedt hulp en ondersteuning.")

# Functie om het busplanningsschema te valideren
def validate_bus_planning(data):
    errors = []
    
    # Voorbeeldcontroles
    # Controleer op ontbrekende waarden
    if data.isnull().values.any():
        errors.append("Er zijn lege waarden in het schema.")
        
    # Controleer of starttijd minder is dan eindtijd
    if 'starttijd' in data.columns and 'eindtijd' in data.columns:
        invalid_times = data[data['starttijd'] >= data['eindtijd']]
        if not invalid_times.empty:
            errors.append(f"Onjuiste tijden: {invalid_times[['starttijd', 'eindtijd']].to_string(index=False)}")

    # Verdere validatielogica kan hier worden toegevoegd

    return errors

# Hoofd pagina selector
page = st.sidebar.selectbox("Selecteer een pagina", ["Bus Planning Checker", "How It Works", "Help"])

if page == "Bus Planning Checker":
    bus_checker_page()
elif page == "How It Works":
    how_it_works_page()
elif page == "Help":
    help_page()
