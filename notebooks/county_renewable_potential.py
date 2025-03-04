"""
County Renewable Potential Map Generator

This script:
1. Calculates county-level wind and solar capacity factors using raster data
2. Creates a 3x3 matrix visualization of renewable potential by county
3. Optionally overlays power plant transition data 
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import rasterstats

def load_county_shapefile(shapefile_path):
    """Load and prepare US county shapefile"""
    # Load counties
    counties = gpd.read_file(shapefile_path)
    
    # Filter for continental US counties only - excluding AK (02) and HI (15)
    conus_states = [str(i).zfill(2) for i in range(1, 57) if i not in [2, 15]]
    counties = counties[counties['STATEFP'].isin(conus_states)]
    
    # Create a 5-digit FIPS code for easier matching
    counties['FIPS'] = counties['STATEFP'] + counties['COUNTYFP']
    
    # Project to appropriate CRS for US
    counties = counties.to_crs("EPSG:5070")  # NAD83 / Conus Albers
    
    return counties

def calculate_county_renewable_potential(counties, wind_tif_path, solar_tif_path):
    """Calculate average wind and solar capacity factors for each county"""
    print("Calculating county renewable potential...")
    
    # Initialize columns
    counties['wind_cf'] = np.nan
    counties['solar_cf'] = np.nan
    
    # Calculate zonal statistics for wind
    print("Processing wind capacity factors...")
    wind_stats = rasterstats.zonal_stats(
        counties.geometry,
        wind_tif_path,
        stats=['mean'],
        nodata=np.nan,
        all_touched=False
    )
    counties['wind_cf'] = [stat['mean'] for stat in wind_stats]
    
    # Calculate zonal statistics for solar
    print("Processing solar capacity factors...")
    solar_stats = rasterstats.zonal_stats(
        counties.geometry,
        solar_tif_path,
        stats=['mean'],
        nodata=np.nan,
        all_touched=False
    )
    counties['solar_cf'] = [stat['mean'] if 'mean' in stat else np.nan for stat in solar_stats]
    
    # Handle missing values with state and national averages
    state_wind_avg = counties.groupby('STATEFP')['wind_cf'].transform('mean')
    state_solar_avg = counties.groupby('STATEFP')['solar_cf'].transform('mean')
    
    national_wind_avg = counties['wind_cf'].mean()
    national_solar_avg = counties['solar_cf'].mean()
    
    # Fill missing values
    counties['wind_cf'] = counties['wind_cf'].fillna(state_wind_avg)
    counties['solar_cf'] = counties['solar_cf'].fillna(state_solar_avg)
    
    counties['wind_cf'] = counties['wind_cf'].fillna(national_wind_avg)
    counties['solar_cf'] = counties['solar_cf'].fillna(national_solar_avg)
    
    print(f"Wind capacity factor range: {counties['wind_cf'].min():.3f} to {counties['wind_cf'].max():.3f}")
    print(f"Solar capacity factor range: {counties['solar_cf'].min():.3f} to {counties['solar_cf'].max():.3f}")
    
    return counties

def create_resource_categories(counties):
    """Create 3x3 matrix resource categories based on wind and solar potential"""
    # Normalize values using quantiles to handle outliers
    wind_min, wind_max = counties['wind_cf'].quantile([0.05, 0.95])
    solar_min, solar_max = counties['solar_cf'].quantile([0.05, 0.95])
    
    # Normalize to 0-1 range
    counties['wind_cf_norm'] = (counties['wind_cf'] - wind_min) / (wind_max - wind_min)
    counties['wind_cf_norm'] = counties['wind_cf_norm'].clip(0, 1)
    
    counties['solar_cf_norm'] = (counties['solar_cf'] - solar_min) / (solar_max - solar_min)
    counties['solar_cf_norm'] = counties['solar_cf_norm'].clip(0, 1)
    
    # Create the 3x3 matrix category (0-8)
    def get_resource_category(row):
        wind_cat = 0 if row['wind_cf_norm'] < 0.33 else (1 if row['wind_cf_norm'] < 0.66 else 2)
        solar_cat = 0 if row['solar_cf_norm'] < 0.33 else (1 if row['solar_cf_norm'] < 0.66 else 2)
        return wind_cat * 3 + solar_cat
    
    counties['resource_category'] = counties.apply(get_resource_category, axis=1)
    
    # Add string descriptions for easier interpretation
    resource_labels = {
        0: 'Low Wind, Low Solar',
        1: 'Low Wind, Med Solar',
        2: 'Low Wind, High Solar',
        3: 'Med Wind, Low Solar',
        4: 'Med Wind, Med Solar',
        5: 'Med Wind, High Solar',
        6: 'High Wind, Low Solar',
        7: 'High Wind, Med Solar',
        8: 'High Wind, High Solar'
    }
    
    counties['resource_label'] = counties['resource_category'].map(resource_labels)
    
    # Count counties in each category
    category_counts = counties['resource_category'].value_counts().sort_index()
    print("\nCounty distribution by resource category:")
    for cat, count in category_counts.items():
        print(f"{resource_labels[cat]}: {count} counties")
    
    return counties

def plot_county_renewable_potential(counties, output_path=None, title=None):
    """Create a map of county-level renewable potential"""
    # Create custom colormap for the 3x3 matrix
    resource_cmap = LinearSegmentedColormap.from_list(
        'resource_cmap', 
        ['#FFFFFF', '#F5F5DC', '#D3D3A4', 
         '#B0E0E6', '#ADD8E6', '#87CEEB', 
         '#6495ED', '#4682B4', '#000080']
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot counties colored by resource category
    counties.plot(
        column='resource_category',
        cmap=resource_cmap,
        linewidth=0.2,
        edgecolor='white',
        alpha=0.8,
        ax=ax
    )
    
    # Set up the legend
    resource_labels = [
        'Low Wind, Low Solar', 'Low Wind, Med Solar', 'Low Wind, High Solar',
        'Med Wind, Low Solar', 'Med Wind, Med Solar', 'Med Wind, High Solar',
        'High Wind, Low Solar', 'High Wind, Med Solar', 'High Wind, High Solar'
    ]
    
    # Create legend patches
    legend_patches = []
    for i in range(9):
        color = resource_cmap(i/8)
        patch = mpatches.Patch(color=color, label=resource_labels[i])
        legend_patches.append(patch)
    
    # Add legend
    ax.legend(
        handles=legend_patches,
        title="County Renewable Potential",
        fontsize=10,
        title_fontsize=12,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5)
    )
    
    # Set title
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title('County-level Renewable Energy Potential', fontsize=16)
    
    # Remove axis
    ax.set_axis_off()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Map saved to {output_path}")
    
    return fig, ax

def overlay_transitions(ax, counties, merged_df, ccs_scenario=True):
    """Overlay power plant transition data on the county map"""
    # Set up colors for different transitions based on scenario
    if ccs_scenario:
        color_dict = {
            'Coal to Wind': '#87CEEB',
            'Coal to Solar': '#FFD700',
            'Coal to NGCC-CCS': '#98FB98',
            'Gas to Wind': '#4682B4',
            'Gas to Solar': '#DAA520',
            'Gas to NGCC-CCS': '#3CB371',
            'Oil to Wind': '#483D8B',
            'Oil to Solar': '#CD853F',
            'Oil to NGCC-CCS': '#2E8B57',
            'Other to Wind': '#808080',
            'Other to Solar': '#A9A9A9',
            'Other to NGCC-CCS': '#696969'
        }
        tech_column = 'NGCC-CCS'
    else:
        color_dict = {
            'Coal to Wind': '#87CEEB',
            'Coal to Solar': '#FFD700',
            'Coal to NGCC': '#98FB98',
            'Gas to Wind': '#4682B4',
            'Gas to Solar': '#DAA520',
            'Gas to NGCC': '#3CB371',
            'Oil to Wind': '#483D8B',
            'Oil to Solar': '#CD853F',
            'Oil to NGCC': '#2E8B57',
            'Other to Wind': '#808080',
            'Other to Solar': '#A9A9A9',
            'Other to NGCC': '#696969'
        }
        tech_column = 'NGCC'
    
    # Helper functions
    def get_lowest_macc_option(row):
        maccs = {
            'Wind': row[f'MACC - Wind ($/tonne CO2)'],
            'Solar': row[f'MACC - Solar ($/tonne CO2)'],
            f'{tech_column}': row[f'MACC - {tech_column} ($/tonne CO2)']
        }
        valid_maccs = {k: v for k, v in maccs.items() if pd.notnull(v) and not np.isinf(v)}
        if not valid_maccs:
            return None, None
        best_option = min(valid_maccs.items(), key=lambda x: x[1])
        return best_option
    
    def get_fuel_category(tech):
        if pd.isna(tech):
            return 'Other'
        tech = str(tech)  # Convert to string to handle float values
        if 'Coal' in tech:
            return 'Coal'
        elif 'Gas' in tech:
            return 'Gas'
        elif 'Petroleum' in tech:
            return 'Oil'
        return 'Other'
    
    # Filter and prepare plant data
    plant_map_df = merged_df[['State', 'Latitude', 'Longitude', 'Technology', 
                            'Old Emissions (tonnes CO2)', f'MACC - Wind ($/tonne CO2)',
                            f'MACC - Solar ($/tonne CO2)', f'MACC - {tech_column} ($/tonne CO2)']].copy()
    
    # Filter for continental US coordinates only
    plant_map_df = plant_map_df[
        (plant_map_df['Longitude'] >= -125) & 
        (plant_map_df['Longitude'] <= -65) & 
        (plant_map_df['Latitude'] >= 25) & 
        (plant_map_df['Latitude'] <= 50)
    ]
    
    # Map the best technology option and derive transition names
    plant_map_df['Best Technology'], plant_map_df['Best MACC'] = zip(*plant_map_df.apply(get_lowest_macc_option, axis=1))
    plant_map_df['Fuel Category'] = plant_map_df['Technology'].apply(get_fuel_category)
    plant_map_df['Transition'] = plant_map_df['Fuel Category'] + " to " + plant_map_df['Best Technology']
    
    # Create ordered list of transitions
    ordered_transitions = []
    if ccs_scenario:
        technologies = ['Solar', 'Wind', 'NGCC-CCS']
    else:
        technologies = ['Solar', 'Wind', 'NGCC']
        
    for tech in technologies:
        for fuel in ['Coal', 'Gas', 'Oil', 'Other']:
            transition = f"{fuel} to {tech}"
            if transition in plant_map_df['Transition'].unique():
                ordered_transitions.append(transition)
    
    # Ensure valid transitions
    plant_map_df = plant_map_df[plant_map_df['Transition'].isin(ordered_transitions)]
    
    # Organize transitions by target technology
    solar_transitions = [t for t in ordered_transitions if 'to Solar' in t]
    wind_transitions = [t for t in ordered_transitions if 'to Wind' in t]
    if ccs_scenario:
        gas_transitions = [t for t in ordered_transitions if 'to NGCC-CCS' in t]
    else:
        gas_transitions = [t for t in ordered_transitions if 'to NGCC' in t and 'CCS' not in t]
    
    # Reorganize transitions
    final_transitions = solar_transitions + wind_transitions + gas_transitions
    
    # Create scatter plots and collect handles and labels for legend
    handles = []
    labels = []
    for transition in final_transitions:
        subset = plant_map_df[plant_map_df['Transition'] == transition]
        if subset.empty:
            continue
            
        sizes = np.sqrt(subset['Old Emissions (tonnes CO2)']) / 8 
        scatter = ax.scatter(subset['Longitude'], subset['Latitude'], 
                          s=sizes, c=color_dict.get(transition, '#808080'), label=transition, 
                          alpha=0.95, edgecolors='black', linewidth=0.6)
        
        # Create a proxy artist for the legend with fixed size
        proxy = mpatches.Patch(color=color_dict.get(transition, '#808080'), label=transition)
        handles.append(proxy)
        labels.append(transition)
    
    # Create size legend handles
    size_legend_values = [1, 5, 10]  # In million tonnes
    size_legend_handles = []
    for value in size_legend_values:
        size = np.sqrt(value * 1e6) / 8  
        handle = mlines.Line2D([], [], color='black', marker='o',
                            markersize=np.sqrt(size)/2,
                            label=f'{value} Mt CO₂',
                            linewidth=0, markerfacecolor='grey',
                            markeredgecolor='black', alpha=0.95)
        size_legend_handles.append(handle)
    
    # Add plant transition legend
    plt.gca().get_figure().legend(
        handles=handles,
        title="Technology Transition",
        fontsize=10,
        title_fontsize=12,
        bbox_to_anchor=(1.02, 0.75),
        loc='center left'
    )
    
    # Add size legend
    plt.gca().get_figure().legend(
        handles=size_legend_handles,
        title="Annual Emissions",
        fontsize=10,
        title_fontsize=12,
        bbox_to_anchor=(1.02, 0.25),
        loc='center left'
    )
    
    # Adjust layout for legends
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    return ax

def save_county_potential_data(counties, output_path):
    """Save the county renewable potential data for later use"""
    # Select relevant columns
    output_data = counties[['STATEFP', 'COUNTYFP', 'NAME', 'FIPS', 
                           'wind_cf', 'solar_cf', 'resource_category', 'resource_label']]
    
    # Save to CSV
    output_data.to_csv(output_path, index=False)
    print(f"County renewable potential data saved to {output_path}")

def main():
    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'output', 'renewable_potential')
    os.makedirs(output_dir, exist_ok=True)
    
    # Define input paths
    county_shapefile = os.path.join(data_dir, 'raw/cb_2018_us_county_500k/cb_2018_us_county_500k.shp')
    wind_tif = os.path.join(data_dir, 'USA_capacity-factor_IEC1.tif')
    solar_tif = os.path.join(data_dir, 'PVOUT_2.tif')
    
    # Load county data
    print("Loading county shapefile...")
    counties = load_county_shapefile(county_shapefile)
    print(f"Loaded {len(counties)} counties")
    
    # Calculate renewable potential
    counties = calculate_county_renewable_potential(counties, wind_tif, solar_tif)
    
    # Create resource categories
    counties = create_resource_categories(counties)
    
    # Plot the map
    print("Creating map visualization...")
    fig, ax = plot_county_renewable_potential(
        counties, 
        output_path=os.path.join(output_dir, 'county_renewable_potential.png'),
        title='County-level Renewable Energy Potential (3×3 Matrix)'
    )
    
    # Save the data
    save_county_potential_data(
        counties,
        output_path=os.path.join(output_dir, 'county_renewable_potential.csv')
    )
    
    print("Done!")
    
if __name__ == "__main__":
    main()