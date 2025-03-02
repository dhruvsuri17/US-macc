import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import zarr
import time
import s3fs
import os
from pathlib import Path

# ================================
# Helper functions for the SR model
# ================================

def rect(i, w, s, e, n):
    """Create rectangle coordinates for grid cell i."""
    x = [w[i], e[i], e[i], w[i], w[i]]
    y = [s[i], s[i], n[i], n[i], s[i]]
    return x, y

def poly(sr):
    """Create polygons for all grid cells."""
    ret = []
    w = sr["W"][:]
    s = sr["S"][:]
    e = sr["E"][:]
    n = sr["N"][:]
    for i in range(52411):
        x, y = rect(i, w, s, e, n)
        ret.append(Polygon([[x[0],y[0]],[x[1],y[1]],[x[2],y[2]],
                            [x[3],y[3]],[x[4],y[4]]]))
    return ret

def run_sr(emis, model="isrm", emis_units="tons/year"):
    """Run the Source-Receptor model for given emissions."""
    start = time.time()
    print(f"Starting SR model run for {len(emis)} facilities...")
    
    # Load spatial receptor grid (SR)
    url = 's3://inmap-model/isrm_v1.2.1.zarr/'
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-east-2"})
    sr = zarr.open(
        store=url,
        mode="r",
        storage_options={"anon": True, "client_kwargs": {"region_name": "us-east-2"}}
    )   

    # Build the grid geometry
    p = poly(sr)
    print("Grid polygons created.")

    # Create grid GeoDataFrame
    df = pd.DataFrame({'Location': range(52411)})
    gdf = gpd.GeoDataFrame(df, geometry=p, crs="+proj=lcc +lat_1=33.000000 +lat_2=45.000000 +lat_0=40.000000 +lon_0=-97.000000 +x_0=0 +y_0=0 +a=6370997.000000 +b=6370997.000000 +to_meter=1")
    
    # Ensure emis has CRS set correctly
    if emis.crs is None:
        print("Warning: emis CRS is None. Assigning default CRS (WGS84).")
        emis = emis.set_crs("EPSG:4326")

    # Convert emissions to match grid CRS
    emis = emis.to_crs(gdf.crs)

    # Spatial join (match emissions to grid)
    join_right_df = gdf.sjoin(emis, how="right")

    # Debugging: Print missing locations
    missing_count = join_right_df.Location.isna().sum()
    print(f"Spatial join complete. Missing locations: {missing_count}")

    # Drop NaN locations if any exist
    join_right_df = join_right_df.dropna(subset=["Location"])
    
    index = join_right_df.Location.astype(int).tolist()  # Ensure integer type

    # Get unique indices for emissions
    ppl = np.unique(index).tolist()

    # Create dictionary for mapping locations to index
    dictionary = {ppl[i]: i for i in range(len(ppl))}

    print("Loading SR matrices...")
    # Load Source-Receptor (SR) matrix data
    SOA = sr['SOA'].get_orthogonal_selection(([0], ppl, slice(None)))
    print("- SOA data loaded")
    pNO3 = sr['pNO3'].get_orthogonal_selection(([0], ppl, slice(None)))
    print("- pNO3 data loaded")
    pNH4 = sr['pNH4'].get_orthogonal_selection(([0], ppl, slice(None)))
    print("- pNH4 data loaded")
    pSO4 = sr['pSO4'].get_orthogonal_selection(([0], ppl, slice(None)))
    print("- pSO4 data loaded")
    PM25 = sr['PrimaryPM25'].get_orthogonal_selection(([0], ppl, slice(None)))
    print("- PrimaryPM25 data loaded")

    # Initialize output data arrays
    SOA_data, pNO3_data, pNH4_data, pSO4_data, PM25_data = 0.0, 0.0, 0.0, 0.0, 0.0

    print("Calculating pollution impact...")
    # Calculate pollution data using emissions
    for i in range(len(index)):
        loc_idx = dictionary[index[i]]  # Get correct index
        SOA_data += SOA[0, loc_idx, :] * emis.VOC.iloc[i]
        pNO3_data += pNO3[0, loc_idx, :] * emis.NOx.iloc[i]
        pNH4_data += pNH4[0, loc_idx, :] * emis.NH3.iloc[i]
        pSO4_data += pSO4[0, loc_idx, :] * emis.SOx.iloc[i]
        PM25_data += PM25[0, loc_idx, :] * emis.PM2_5.iloc[i]

    data = SOA_data + pNO3_data + pNH4_data + pSO4_data + PM25_data

    # Apply emission unit conversion factor
    fact = 28766.639 if emis_units == "tons/year" else 1

    print("Computing health impacts...")
    # Compute final pollution metrics
    TotalPM25 = fact * data
    TotalPop = sr['TotalPop'][0:52411]
    MortalityRate = sr['MortalityRate'][0:52411]
    deathsK = (np.exp(np.log(1.06)/10 * TotalPM25) - 1) * TotalPop * 1.04658 * MortalityRate / 100000 * 1.02523
    deathsL = (np.exp(np.log(1.14)/10 * TotalPM25) - 1) * TotalPop * 1.04658 * MortalityRate / 100000 * 1.02523

    # Create output GeoDataFrame
    ret = gpd.GeoDataFrame(pd.DataFrame({
        'SOA': fact * SOA_data,
        'pNO3': fact * pNO3_data,
        'pNH4': fact * pNH4_data,
        'pSO4': fact * pSO4_data,
        'PrimaryPM25': fact * PM25_data,
        'TotalPM25': TotalPM25,
        'deathsK': deathsK,
        'deathsL': deathsL
    }), geometry=p[:52411], crs=gdf.crs)

    print(f"SR model run complete in {time.time() - start:.0f} seconds")
    return ret

# ================================
# Functions for single plant analysis
# ================================

def load_data(nei_file_path, counties_shapefile_path):
    """Load NEI data and county boundaries."""
    print("Loading data...")
    
    # Load NEI facility data
    try:
        df = pd.read_csv(nei_file_path, sep=',', low_memory=False)
        print(f"NEI data loaded with shape: {df.shape}")
    except Exception as e:
        try:
            df = pd.read_csv(nei_file_path, low_memory=False)
            print(f"NEI data loaded with shape: {df.shape}")
        except Exception as e2:
            print(f"Error loading NEI data: {e2}")
            return None, None
    
    # Process NEI data as in your original script
    # Convert emissions to metric tonnes
    def convert_to_tonnes(row):
        if row['emissions uom'] == 'LB':
            return float(row['total emissions']) * 0.000453592  # Convert pounds to metric tonnes
        elif row['emissions uom'] == 'TON':
            return float(row['total emissions']) * 0.90718474  # Convert short tons to metric tonnes
        return float(row['total emissions'])  # Already in metric tonnes

    df['emissions_tonnes'] = df.apply(convert_to_tonnes, axis=1)

    # Categorize pollutants
    def categorize_pollutant(row):
        pollutant = str(row['pollutant code']).upper()
        pollutant_desc = str(row['pollutant desc']).upper()

        if pollutant == 'VOC' or 'VOLATILE ORGANIC' in pollutant_desc:
            return 'VOC'
        elif pollutant in ['NOX', 'NO', 'NO2'] or ('NITROGEN' in pollutant_desc and 'OXIDE' in pollutant_desc):
            return 'NOx'
        elif pollutant == 'NH3' or 'AMMONIA' in pollutant_desc:
            return 'NH3'
        elif pollutant in ['SO2', 'SO4'] or 'SULFUR' in pollutant_desc:
            return 'SOx'
        elif 'PM25' in pollutant or 'PM2.5' in pollutant_desc or 'PM2_5' in pollutant:
            return 'PM2_5'
        return 'Other'

    df['pollutant_category'] = df.apply(categorize_pollutant, axis=1)

    # Filter out Alaska & Hawaii
    df = df[~df['state'].isin(['AK', 'HI'])]

    # Aggregate data by facility
    facility_emissions = df.groupby([
        'eis facility id', 'site name', 'state', 'site latitude', 'site longitude', 
        'primary naics code', 'primary naics description', 'pollutant_category'
    ])['emissions_tonnes'].sum().reset_index()

    # Convert to wide format with pollutants as columns
    facility_wide = facility_emissions.pivot_table(
        index=['eis facility id', 'site name', 'state', 'site latitude', 'site longitude', 
               'primary naics code', 'primary naics description'],
        columns='pollutant_category', 
        values='emissions_tonnes',
        fill_value=0
    ).reset_index()

    # Ensure all required pollutant columns exist
    for cat in ['VOC', 'NOx', 'NH3', 'SOx', 'PM2_5']:
        if cat not in facility_wide.columns:
            facility_wide[cat] = 0

    # Create GeoDataFrame
    facility_wide['geometry'] = facility_wide.apply(lambda row: Point(row['site longitude'], row['site latitude']), axis=1)
    gdf = gpd.GeoDataFrame(facility_wide, geometry='geometry', crs='epsg:4269')

    # Filter for power plants (EGUs) using NAICS codes
    egu_naics = ['2211', '221111', '221112', '221113', '221114', '221115', '221116', '221117', '221118']
    egu_gdf = gdf[gdf['primary naics code'].astype(str).str.startswith(tuple(egu_naics))]
    print(f"Found {len(egu_gdf)} power plant facilities")

    # Load county boundaries
    try:
        us_counties = gpd.read_file(counties_shapefile_path)
        print(f"County boundaries loaded with {len(us_counties)} counties")
    except Exception as e:
        print(f"Error loading county boundaries: {e}")
        return egu_gdf, None
    
    return egu_gdf, us_counties

def list_facilities(egu_gdf, n=20):
    """Display a list of facilities for selection."""
    sample = egu_gdf[['eis facility id', 'site name', 'state', 'NOx', 'SOx', 'PM2_5']].drop_duplicates()
    sample = sample.sort_values('NOx', ascending=False).head(n)
    
    # Format the emissions columns
    for col in ['NOx', 'SOx', 'PM2_5']:
        sample[col] = sample[col].map(lambda x: f"{x:.1f}")
    
    sample.columns = ['Facility ID', 'Facility Name', 'State', 'NOx (tonnes)', 'SOx (tonnes)', 'PM2.5 (tonnes)']
    return sample

def analyze_single_plant(egu_gdf, facility_id):
    """Analyze health impacts from a single facility."""
    # Filter for just one facility
    single_plant = egu_gdf[egu_gdf['eis facility id'] == facility_id].copy()
    
    if len(single_plant) == 0:
        print(f"Facility ID {facility_id} not found.")
        return None
        
    print(f"Analyzing facility: {single_plant['site name'].iloc[0]}")
    print(f"Location: {single_plant['state'].iloc[0]}")
    print(f"Emissions (tonnes): NOx={single_plant['NOx'].iloc[0]:.2f}, SOx={single_plant['SOx'].iloc[0]:.2f}, PM2.5={single_plant['PM2_5'].iloc[0]:.2f}")
    
    # Run the SR model on just this facility
    results = run_sr(single_plant, model="isrm", emis_units="tons/year")
    
    return results, single_plant

def calculate_plant_county_impacts(results, us_counties):
    """Calculate county-level health impacts for a single plant."""
    # Convert counties to match results CRS if needed
    us_counties = us_counties.to_crs(results.crs)
    
    # Exclude Alaska, Hawaii, and Puerto Rico using STATEFP codes
    us_counties = us_counties[~us_counties['STATEFP'].isin(["02", "15", "72"])]
    
    # Perform spatial join to assign each grid cell to a county
    results_county = results.sjoin(us_counties, how="left", predicate="intersects")
    
    # Aggregate health impacts by county and state
    county_summary = results_county.groupby(["STATEFP", "NAME"]).agg({
        "TotalPM25": "mean",  # Average PM2.5 concentration
        "deathsK": "sum",     # Total premature deaths
    }).reset_index()
    
    # Calculate health damages using VSL
    VSL = 13.2e6  # Value of a Statistical Life in dollars
    county_summary['HealthDamages'] = county_summary['deathsK'] * VSL
    
    # Add state names
    state_fips = {
        '01': 'Alabama', '04': 'Arizona', '05': 'Arkansas', '06': 'California', 
        '08': 'Colorado', '09': 'Connecticut', '10': 'Delaware', '11': 'District of Columbia',
        '12': 'Florida', '13': 'Georgia', '16': 'Idaho', '17': 'Illinois', '18': 'Indiana',
        '19': 'Iowa', '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana', '23': 'Maine',
        '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota',
        '28': 'Mississippi', '29': 'Missouri', '30': 'Montana', '31': 'Nebraska',
        '32': 'Nevada', '33': 'New Hampshire', '34': 'New Jersey', '35': 'New Mexico',
        '36': 'New York', '37': 'North Carolina', '38': 'North Dakota', '39': 'Ohio',
        '40': 'Oklahoma', '41': 'Oregon', '42': 'Pennsylvania', '44': 'Rhode Island',
        '45': 'South Carolina', '46': 'South Dakota', '47': 'Tennessee', '48': 'Texas',
        '49': 'Utah', '50': 'Vermont', '51': 'Virginia', '53': 'Washington',
        '54': 'West Virginia', '55': 'Wisconsin', '56': 'Wyoming'
    }
    
    county_summary['State'] = county_summary['STATEFP'].map(state_fips)
    
    # Format the county name with state
    county_summary['County'] = county_summary['NAME'] + ', ' + county_summary['State']
    
    return county_summary

def plot_single_plant_impacts(county_summary, us_counties, facility_name, facility_state, plant_info):
    """Create a map visualization of single plant impacts with plant location marker."""
    # Merge summary with county shapefile
    us_counties = us_counties.to_crs('epsg:4269')  # Ensure consistent CRS
    counties_with_impacts = us_counties.merge(
        county_summary[['NAME', 'STATEFP', 'HealthDamages']], 
        on=['NAME', 'STATEFP'], 
        how='left'
    )
    
    # Fill NaN values with 0
    counties_with_impacts['HealthDamages'] = counties_with_impacts['HealthDamages'].fillna(0)
    
    # Create bins and colors for the map
    # Adjust bins based on the actual damage values
    max_damage = counties_with_impacts['HealthDamages'].max()
    
    if max_damage < 1e6:  # Less than $1M max
        bins = [0, 1e3, 5e3, 1e4, 5e4, 1e5, 2e5, 5e5, max_damage * 0.9, float("inf")]
        legend_labels = [
            "$0 - $1K", "$1K - $5K", "$5K - $10K", "$10K - $50K", 
            "$50K - $100K", "$100K - $200K", "$200K - $500K", 
            f"$500K - ${int(max_damage * 0.9/1000)}K", f"${int(max_damage * 0.9/1000)}K+"
        ]
    else:  # More than $1M max
        bins = [0, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, max_damage * 0.9, float("inf")]
        legend_labels = [
            "$0 - $10K", "$10K - $50K", "$50K - $100K", "$100K - $500K", 
            "$500K - $1M", "$1M - $5M", "$5M - $10M", 
            f"$10M - ${int(max_damage * 0.9/1000000)}M", f"${int(max_damage * 0.9/1000000)}M+"
        ]
    
    colors = ['#ffedea', '#ffcec5', '#ffad9f', '#ff7f66', '#ff4d33', 
              '#ff1a00', '#cc1600', '#990f00', '#660a00', '#400600']
    
    cmap = mcolors.ListedColormap(colors)
    counties_with_impacts['HealthDamages_Binned'] = pd.cut(
        counties_with_impacts['HealthDamages'], bins=bins, labels=False, include_lowest=True)
    
    # Create the map
    fig, ax = plt.subplots(figsize=(12, 8))
    counties_with_impacts.plot(column='HealthDamages_Binned', cmap=cmap, 
                               linewidth=0.3, edgecolor="black", ax=ax, legend=False)
    
    # Add plant location marker
    # Convert plant location to the same CRS as the map
    plant_point = gpd.GeoDataFrame(
        {'name': [facility_name]}, 
        geometry=[Point(plant_info['site longitude'].iloc[0], plant_info['site latitude'].iloc[0])],
        crs='epsg:4269'
    )
    
    # Plot plant location with a red star marker
    plant_point.plot(ax=ax, color='red', marker='*', markersize=150, zorder=5)
    
    # Add text label for the plant
    plt.text(
        x=plant_info['site longitude'].iloc[0] + 0.5,  # Offset for readability
        y=plant_info['site latitude'].iloc[0],
        s=facility_name,
        fontsize=10,
        ha='left',
        va='center',
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'),
        zorder=6
    )
    
    # Add title and formatting
    ax.set_title(f"Health Damages from {facility_name} ({facility_state})", fontsize=14)
    ax.axis('off')
    ax.set_aspect(1.3)
    
    # Set limits to properly zoom into the contiguous U.S.
    ax.set_xlim(-130, -60)  # Longitude limits
    ax.set_ylim(20, 55)     # Latitude limits
    
    # Create legend
    legend_patches = [mpatches.Patch(color=colors[i], label=legend_labels[i]) 
                     for i in range(len(legend_labels))]
    
    ax.legend(handles=legend_patches, title="Health Damages ($)", loc="lower right")
    plt.tight_layout()
    
    return fig, counties_with_impacts

def calculate_national_totals(results):
    """Calculate national total impacts."""
    total_pm25 = results['TotalPM25'].sum()
    total_deaths = results['deathsK'].sum()
    total_damages = total_deaths * 13.2e6  # VSL in dollars
    
    return {
        'total_pm25': total_pm25,
        'total_deaths': total_deaths,
        'total_damages': total_damages
    }

# This function has been removed as we're focusing on single plant analysis

# ================================
# Main function to run the analysis for a single plant
# ================================

def analyze_plant_by_id(nei_file_path, counties_shapefile_path, facility_id):
    """Analyze health impacts from a specific power plant by ID."""
    # Load data
    egu_gdf, us_counties = load_data(nei_file_path, counties_shapefile_path)
    
    if egu_gdf is None:
        print("Error loading data. Exiting.")
        return None, None, None
    
    # Check if the facility exists
    if facility_id not in egu_gdf['eis facility id'].values:
        print(f"Facility ID {facility_id} not found. Available facilities include:")
        sample_facilities = list_facilities(egu_gdf, n=10)
        print(sample_facilities)
        return None, None, None
    
    # Run analysis for the specific plant
    print(f"Analyzing facility ID: {facility_id}")
    results, plant_info = analyze_single_plant(egu_gdf, facility_id)
    
    if results is None:
        return None, None, None
    
    # Calculate county impacts
    county_impacts = calculate_plant_county_impacts(results, us_counties)
    
    # Display top impacted counties
    print("\nTop 10 counties with highest health damages:")
    top_counties = county_impacts.sort_values('HealthDamages', ascending=False).head(10)
    top_counties['HealthDamages_Millions'] = top_counties['HealthDamages'] / 1e6
    print(top_counties[['County', 'HealthDamages_Millions']].rename(
        columns={'HealthDamages_Millions': 'Health Damages ($ millions)'}))
    
    # Calculate national totals
    national_totals = calculate_national_totals(results)
    print("\nNational Totals:")
    print(f"Total premature deaths: {national_totals['total_deaths']:.2f}")
    print(f"Total health damages: ${national_totals['total_damages']/1e6:.2f} million")
    
    # Create visualization
    facility_name = plant_info['site name'].iloc[0]
    facility_state = plant_info['state'].iloc[0]
    fig, counties_with_data = plot_single_plant_impacts(county_impacts, us_counties, facility_name, facility_state, plant_info)
    
    # Save results
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(f"{output_dir}/health_impacts_{facility_id}.png", dpi=300, bbox_inches='tight')
    county_impacts.to_csv(f"{output_dir}/county_impacts_{facility_id}.csv", index=False)
    
    print(f"\nResults saved to {output_dir}/")
    
    # Show plot
    plt.show()
    
    return results, county_impacts, national_totals

# ================================
# Example usage
# ================================

if __name__ == "__main__":
    # Paths to your data files
    nei_file_path = "../data/raw/2021_NEI_Facility_summary.csv"
    counties_shapefile_path = "../data/raw/cb_2018_us_county_500k/cb_2018_us_county_500k.shp"
    
    # Load data first to list available facilities
    egu_gdf, us_counties = load_data(nei_file_path, counties_shapefile_path)
    
    if egu_gdf is not None:
        # Display sample facilities for selection
        print("\nTop facilities by NOx emissions:")
        sample_facilities = list_facilities(egu_gdf, n=20)
        print(sample_facilities)
        
        # Get a facility ID from the list (in a real scenario, this could be user input)
        # Here we just use the first one in the list as an example
        facility_id = sample_facilities['Facility ID'].iloc[0]
        
        # Analyze a specific facility
        results, county_impacts, national_totals = analyze_plant_by_id(
            nei_file_path, counties_shapefile_path, facility_id)
    else:
        print("Could not load data. Please check file paths.")