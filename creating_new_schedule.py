import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

# Laad de Excel-data (zorg dat je eigen pad correct is)
df = pd.read_excel('omloopplanning.xlsx')

# Zorg dat de tijd in datetime formaat staat
df['starttijd'] = pd.to_datetime(df['starttijd'])
df['eindtijd'] = pd.to_datetime(df['eindtijd'])

# Definieer een kleurenmap voor de activiteiten
kleur_map = {'materiaal rit': 'blue', 'dienst rit': 'green', 'idle': 'red', 'opladen': 'orange'}

# Begin met het plotten van de Gantt-grafiek
fig, ax = plt.subplots(figsize=(10, 6))

# Ga door elke rij in de dataframe en plot de activiteiten als een bar
for i, row in df.iterrows():
    start = row['starttijd']
    eind = row['eindtijd']
    omloop = row['omloop nummer']
    activiteit = row['activiteit']
    
    # Plot een horizontale balk voor elke activiteit
    ax.barh(omloop, (eind - start).total_seconds()/3600, left=start, color=kleur_map[activiteit])

# X-as instellen voor 24 uur
ax.set_xlim([df['starttijd'].min().replace(hour=0, minute=0, second=0),
             df['starttijd'].min().replace(hour=23, minute=59, second=59)])

# Stappen per uur op de x-as
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Label de assen en de grafiek
ax.set_xlabel('Tijd (per uur)')
ax.set_ylabel('Omloop Nummer')
ax.set_title('Gantt Chart voor Activiteiten')
plt.xticks(rotation=45)

# Maak een legenda op basis van de activiteiten
handles = [mpatches.Patch(color=color, label=label) for label, color in kleur_map.items()]
ax.legend(handles=handles)

# Toon de grafiek
plt.tight_layout()
plt.show()

# droppen als startlocatie niet eindlocatie is, dus die rare met idle
df_filtered = df[df['startlocatie'] != df['eindlocatie']]

# afstand erachter 
uploaded_file = pd.read_excel('omloopplanning.xlsx')
distance_matrix = pd.read_excel("Connexxion data - 2024-2025.xlsx", sheet_name="Afstandsmatrix")
time_table = pd.read_excel("Connexxion data - 2024-2025.xlsx", sheet_name="Dienstregeling")
df = pd.merge(df_filtered, distance_matrix , on=['startlocatie', 'eindlocatie', 'buslijn'], how='left')

# consumptie erbij 
consumption_per_km = (0.7 + 2.5) / 2  
df['consumptie_kWh'] = (df['afstand in meters']/1000) * consumption_per_km

