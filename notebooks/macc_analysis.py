"""
MACC Analysis Module - U.S. Power Grid Transition Analysis

This module implements a Marginal Abatement Cost Curve (MACC) framework to evaluate 
the cost-effectiveness of transitioning existing fossil fuel power plants to wind, 
solar, and natural gas with or without carbon capture and storage.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import rowcol
from matplotlib.ticker import FuncFormatter
import logging
import geopandas as gpd
import matplotlib.lines as mlines

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('macc_analysis')

# Default configuration
DEFAULT_CONFIG = {
    'discount_rate': 0.06,
    'gas_price': 3.5,  # $/MMBtu
    'analysis_year': 2024,
    'plant_lifetime': 30,  # years
    'ngcc_cf': 0.59,  # capacity factor for NGCC
    'battery_storage_ratio': 25/150,  # MWh per MW of installed capacity
    'macc_filter_min': -1000,  # $/tonne CO2
    'macc_filter_max': 1000,   # $/tonne CO2
    'data_dir': 'data',
    'output_dir': 'output',
    'save_intermediates': True
}

def fix_paths_for_notebook(config):
    """
    Adjust paths in config to work when running from notebooks directory
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Updated configuration with absolute paths
    """
    import os
    
    # Make a copy to avoid modifying the original
    updated_config = config.copy()
    
    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set project root to one level up if we're in notebooks directory
    if os.path.basename(module_dir) == 'notebooks':
        project_root = os.path.dirname(module_dir)
    else:
        project_root = module_dir
    
    # If data_dir is a relative path, make it absolute
    if 'data_dir' in config and not os.path.isabs(config['data_dir']):
        updated_config['data_dir'] = os.path.join(project_root, config['data_dir'])
    
    # Same for output_dir
    if 'output_dir' in config and not os.path.isabs(config['output_dir']):
        updated_config['output_dir'] = os.path.join(project_root, config['output_dir'])
    
    return updated_config

# Update load_config to use fix_paths_for_notebook
def load_config(config_file=None):
    """
    Load configuration from file or use defaults
    
    Args:
        config_file: Path to configuration file (optional)
        
    Returns:
        Dictionary of configuration parameters
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_file and os.path.exists(config_file):
        try:
            custom_config = pd.read_json(config_file, typ='series').to_dict()
            config.update(custom_config)
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.warning(f"Error loading config file: {e}")
            logger.warning("Using default configuration")
    
    # Fix paths for notebook environment
    config = fix_paths_for_notebook(config)
    
    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)
    
    return config

def load_emissions_data(config):
    """
    Load and preprocess annual emissions data
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame of processed emissions data and set of facility IDs
    """
    file_path = os.path.join(config['data_dir'], '2023_annual_emissions_CEMS.csv')
    
    try:
        annual_emissions = pd.read_csv(file_path)
        logger.info(f"Loaded emissions data with {len(annual_emissions)} records")
        
        # Drop unnecessary columns and remove rows with NaN values
        annual_emissions.drop(columns=['Facility Name', 'Year', 'Steam Load (1000 lb)'], 
                            inplace=True, errors='ignore')
        annual_emissions.dropna(inplace=True)
        
        # Get unique plant IDs
        plants = set(annual_emissions['Facility ID'].unique())
        logger.info(f"Found {len(plants)} unique facilities in emissions data")
        
        if config['save_intermediates']:
            annual_emissions.to_csv(os.path.join(config['output_dir'], 'processed_emissions.csv'), index=False)
            
        return annual_emissions, plants
    
    except FileNotFoundError:
        logger.error(f"Emissions data file not found at {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error processing emissions data: {e}")
        raise

def load_plant_data(config, plants):
    """
    Load and process plant data
    
    Args:
        config: Configuration dictionary
        plants: Set of facility IDs to filter
        
    Returns:
        DataFrame of processed plant data
    """
    file_path = os.path.join(config['data_dir'], '3_1_Generator_Y2023.xlsx')
    
    try:
        plant_df = pd.read_excel(file_path, skiprows=1)
        logger.info(f"Loaded plant data with {len(plant_df)} records")
        
        selected_columns = [
            'Plant Code', 'State', 'County', 'Generator ID', 'Technology', 
            'Prime Mover', 'Nameplate Capacity (MW)', 'Status', 'Operating Year', 
            'Planned Retirement Year', 'Energy Source 1'
        ]
        
        # Filter relevant plants
        plant_df_filtered = plant_df.loc[plant_df['Plant Code'].isin(plants), selected_columns]
        logger.info(f"Filtered to {len(plant_df_filtered)} plant records")
        
        # Rename 'Energy Source 1' to 'Energy Source'
        plant_df_filtered.rename(columns={'Energy Source 1': 'Energy Source'}, inplace=True)
        
        # Define fuel mapping
        fuel_mapping = {
            'NG': 'Gas', 'BIT': 'Coal', 'DFO': 'Oil', 'SUB': 'Coal', 'MWH': 'Other', 
            'SUN': 'Other', 'KER': 'Oil', 'RFO': 'Oil', 'WAT': 'Other', 'JF': 'Other', 
            'WDS': 'Other', 'NUC': 'Other', 'SGC': 'Gas', 'OG': 'Gas', 'PC': 'Coal', 
            'LFG': 'Gas', 'RC': 'Coal', 'LIG': 'Coal', 'WC': 'Other', 'BLQ': 'Other', 
            'OBG': 'Other', 'WND': 'Other', 'MSW': 'Other', 'BFG': 'Other'
        }
        
        plant_df_filtered['Fuel Category'] = plant_df_filtered['Energy Source'].map(fuel_mapping)
        
        # Ensure 'Fuel Category' is not missing
        plant_df_filtered.dropna(subset=['Fuel Category'], inplace=True)
        
        # Remove 'Other' category
        plant_df_filtered = plant_df_filtered[plant_df_filtered['Fuel Category'] != 'Other']
        
        if config['save_intermediates']:
            plant_df_filtered.to_csv(os.path.join(config['output_dir'], 'filtered_plants.csv'), index=False)
            
        return plant_df_filtered
    
    except FileNotFoundError:
        logger.error(f"Plant data file not found at {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error processing plant data: {e}")
        raise

# Update the function signature to accept config as a parameter
def aggregate_facility_data(plant_df_filtered, config):
    """
    Aggregate plant data to facility level
    
    Args:
        plant_df_filtered: Filtered plant DataFrame
        config: Configuration dictionary with analysis year
        
    Returns:
        DataFrame of facility-level attributes and unit-level attributes
    """
    # Facility-level aggregation function
    def assign_facility_attributes(group):
        # Find the largest unit in the group
        largest_unit = group.loc[group['Nameplate Capacity (MW)'].idxmax()]
        
        return pd.Series({
            'Technology': largest_unit['Technology'],
            'Age': config['analysis_year'] - group['Operating Year'].min(),
            'Energy Source': largest_unit['Energy Source'],
            'Total Nameplate Capacity (MW)': group['Nameplate Capacity (MW)'].sum(),
            'Fuel Category': largest_unit['Fuel Category']
        })
    
    # Aggregate facility-level attributes
    aggregated_facility_attributes = plant_df_filtered.groupby('Plant Code').apply(assign_facility_attributes).reset_index()
    aggregated_facility_attributes.rename(columns={'Plant Code': 'Facility ID'}, inplace=True)
    
    # Extract unit-level attributes
    unit_level_attributes = plant_df_filtered[['Plant Code', 'Generator ID', 'Technology', 
                                             'Operating Year', 'Nameplate Capacity (MW)', 
                                             'Fuel Category', 'Energy Source']]
    unit_level_attributes.rename(columns={'Plant Code': 'Facility ID'}, inplace=True)
    
    return aggregated_facility_attributes, unit_level_attributes

def load_capital_cost_data(config):
    """
    Load and process capital cost data
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame of processed capital cost data
    """
    file_path = os.path.join(config['data_dir'], 'Power plant capital cost.xlsx')
    
    try:
        capital_cost_df = pd.read_excel(file_path)
        logger.info(f"Loaded capital cost data with {len(capital_cost_df)} records")
        
        # Inflation adjustments (precomputed CPI data)
        inflation_data = {year: index for year, index in zip(
            range(1985, 2025), 
            [323.2, 329.4, 341.4, 355.4, 372.5, 392.6, 409.3, 421.7, 434.1, 445.4, 
             457.9, 471.3, 482.4, 489.8, 500.6, 517.5, 532.1, 540.5, 552.8, 567.6, 
             586.9, 605.8, 623.1, 647.0, 644.7, 655.3, 676.0, 689.9, 700.0, 711.4, 
             712.3, 721.2, 736.6, 754.6, 768.3, 777.7, 814.3, 879.4, 915.6, 944.9]
        )}
        
        # Adjust costs to current year dollars
        analysis_year = config['analysis_year']
        capital_cost_df[f'Cost ({analysis_year}$/kW)'] = capital_cost_df.apply(
            lambda row: row['Cost ($/kW)'] * (inflation_data[analysis_year] / inflation_data.get(row['$ Year'], 1)), 
            axis=1
        )
        
        return capital_cost_df
    
    except FileNotFoundError:
        logger.error(f"Capital cost data file not found at {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error processing capital cost data: {e}")
        raise

def calculate_capital_costs(unit_level_attributes, capital_cost_df, config):
    """
    Calculate capital costs for each unit
    
    Args:
        unit_level_attributes: DataFrame of unit-level attributes
        capital_cost_df: DataFrame of capital cost data
        config: Configuration dictionary
        
    Returns:
        DataFrame with capital costs added
    """
    # Define technology mapping
    mapping_rules = {
        'Natural Gas Fired Combustion Turbine': 'Combustion turbine',
        'Conventional Steam Coal': 'Coal',
        'Natural Gas Fired Combined Cycle': 'Combined cycle',
        'Natural Gas Steam Turbine': 'Oil/Gas steam',
        'Petroleum Liquids': 'Combustion turbine - aeroderivative',
        'Natural Gas Internal Combustion Engine': 'Internal combustion engine',
        'Petroleum Coke': 'Combustion turbine - aeroderivative'
    }
    
    # Find the closest capital cost
    def find_closest_capital_cost(row):
        mapped_technology = mapping_rules.get(row['Technology'])
        if mapped_technology:
            costs = capital_cost_df[capital_cost_df['Technology'] == mapped_technology]
            if not costs.empty:
                closest_year = costs.iloc[(costs['Year'] - row['Operating Year']).abs().argsort()[:1]]
                return closest_year[f'Cost ({config["analysis_year"]}$/kW)'].values[0]
        return None
    
    # Add capital costs and age
    unit_level_attributes['Capital Cost (2024$/kW)'] = unit_level_attributes.apply(find_closest_capital_cost, axis=1)
    unit_level_attributes['Age'] = config['analysis_year'] - unit_level_attributes['Operating Year']
    
    # Facility-level aggregation with Fuel Category preservation
    def get_facility_attributes(group):
        largest_unit = group.loc[group['Nameplate Capacity (MW)'].idxmax()]
        return pd.Series({
            'Technology': largest_unit['Technology'],
            'Technology_alt': mapping_rules.get(largest_unit['Technology'], 'Unknown'),
            'Operating Year': largest_unit['Operating Year'],
            'Age': largest_unit['Age'],
            'Total Capital Cost (2024$/kW)': group['Capital Cost (2024$/kW)'].sum(),
            'Total Nameplate Capacity (MW)': group['Nameplate Capacity (MW)'].sum(),
            'Fuel Category': largest_unit['Fuel Category'],
            'Energy Source': largest_unit['Energy Source']
        })
    
    # Apply function to group by Facility ID
    facility_level_df = unit_level_attributes.groupby('Facility ID').apply(get_facility_attributes).reset_index()
    
    if config['save_intermediates']:
        facility_level_df.to_csv(os.path.join(config['output_dir'], 'facility_level_data.csv'), index=False)
    
    return facility_level_df

def load_technology_data():
    """
    Load technology data for different power generation technologies
    
    Returns:
        DataFrame of technology data
    """
    data = {
        "Technology": [
            "Ultra-supercritical coal (USC)",
            "USC with 30% carbon capture and sequestration (CCS)",
            "USC with 90% CCS",
            "Combined-cycle—single-shaft",
            "Combined-cycle—multi-shaft",
            "Combined-cycle with 90% CCS",
            "Internal combustion engine",
            "Combustion turbine-aeroderivative",
            "Combustion turbine—industrial frame",
            "Wind",
            "Solar",
            "Oil/Gas steam"
        ],
        "Size (MW)": [
            650, 650, 650, 418, 1083, 377, 21, 105, 237, 200, 150, 418
        ],
        "Lead time (years)": [
            4, 4, 4, 3, 3, 3, 2, 2, 2, 3, 2, 3
        ],
        "Base overnight cost (2022$/kW)": [
            4507, 5577, 7176, 1330, 1176, 3019, 2240, 1428, 867, 2098, 1448, 1330
        ],
        "Total overnight cost (2022$/kW)": [
            4507, 5633, 7319, 1330, 1176, 3140, 2240, 1428, 867, 2098, 1448, 1330
        ],
        "Variable O&M (2022$/MWh)": [
            5.06, 7.97, 12.35, 2.87, 2.10, 6.57, 6.40, 5.29, 5.06, 0.00, 0.00, 2.87
        ],
        "Fixed O&M (2022$/kW)": [
            45.68, 61.11, 67.02, 15.87, 13.73, 31.06, 39.57, 18.35, 7.88, 29.64, 17.16, 15.87
        ],
        "Heat rate (Btu/kWh)": [
            8638, 9751, 12507, 6431, 6370, 7124, 8295, 9124, 9905, None, None, 6431
        ],
        "Fuel Cost ($/MMBtu)": [
            2.25, 2.25, 2.25, 4.25, 4.25, 4.25, 20.00, 4.25, 4.25, 0.00, 0.00, 12.50
        ],
        "Fuel Cost ($/MWh)": [
            19.43, 21.94, 28.14, 27.31, 27.10, 30.53, 165.90, 38.53, 42.12, 0.00, 0.00, 80.39
        ]
    }
    
    # Create the DataFrame
    technology_df = pd.DataFrame(data)
    
    # Inverse of the technology mapping
    inverse_mapping = {
        "Combustion turbine-aeroderivative": "Natural Gas Fired Combustion Turbine",
        "Ultra-supercritical coal (USC)": "Conventional Steam Coal",
        "Combined-cycle—multi-shaft": "Natural Gas Fired Combined Cycle",
        "Combustion turbine—industrial frame": "Petroleum Liquids",
        "Internal combustion engine": "Natural Gas Internal Combustion Engine"
    }
    
    technology_df["Technology"] = technology_df["Technology"].replace(inverse_mapping)
    
    # Apply the mapping rules
    mapping_rules = {
        "Natural Gas Fired Combustion Turbine": "Combustion turbine",
        "Conventional Steam Coal": "Coal",
        "Natural Gas Fired Combined Cycle": "Combined cycle",
        "Natural Gas Steam Turbine": "Oil/Gas steam",
        "Petroleum Liquids": "Combustion turbine - aeroderivative",
        "Natural Gas Internal Combustion Engine": "Internal combustion engine",
        "Petroleum Coke": "Combustion turbine - aeroderivative"
    }
    
    technology_df["Technology"] = technology_df["Technology"].replace(mapping_rules)
    
    # Preserved technologies
    preserved_technologies = [
        "Wind", "Solar", "Combined-cycle with 90% CCS", "USC with 90% CCS", "Oil/Gas steam"
    ]
    
    # Filter technologies
    technology_df = technology_df[
        technology_df["Technology"].isin(mapping_rules.values()) | 
        technology_df["Technology"].isin(preserved_technologies)
    ]
    
    return technology_df

def load_plant_location_data(config, plants):
    """
    Load and process plant location data
    
    Args:
        config: Configuration dictionary
        plants: Set of facility IDs to filter
        
    Returns:
        DataFrame of plant locations
    """
    file_path = os.path.join(config['data_dir'], '2___Plant_Y2023.xlsx')
    
    try:
        plant_location = pd.read_excel(file_path, skiprows=1)
        plant_location = plant_location[['Plant Code', 'Latitude', 'Longitude']]
        plant_location.rename(columns={'Plant Code': 'Facility ID'}, inplace=True)
        
        # Filter for plants in the list
        plant_location = plant_location[plant_location['Facility ID'].isin(plants)]
        logger.info(f"Loaded location data for {len(plant_location)} plants")
        
        return plant_location
    
    except FileNotFoundError:
        logger.error(f"Plant location data file not found at {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error processing plant location data: {e}")
        raise

def get_renewable_resources(plant_location, config):
    """
    Get renewable resource data (wind capacity factors and solar PVOUT) for each plant location
    
    Args:
        plant_location: DataFrame of plant locations
        config: Configuration dictionary
        
    Returns:
        DataFrame with added renewable resource data
    """
    # Function to find the closest valid wind resource value
    def get_closest_wind_value(lat, lon, wind_raster, transform, max_search_radius=5):
        row, col = rowcol(transform, lon, lat)
        try:
            if 0 <= row < wind_raster.shape[0] and 0 <= col < wind_raster.shape[1]:
                wind_value = wind_raster[row, col]
                if not np.isnan(wind_value):
                    return wind_value
            for radius in range(1, max_search_radius + 1):
                for drow in range(-radius, radius + 1):
                    for dcol in range(-radius, radius + 1):
                        new_row, new_col = row + drow, col + dcol
                        if 0 <= new_row < wind_raster.shape[0] and 0 <= new_col < wind_raster.shape[1]:
                            wind_value = wind_raster[new_row, new_col]
                            if not np.isnan(wind_value):
                                return wind_value
            return None
        except IndexError:
            return None
    
    # Function to find the closest valid solar resource value
    def get_closest_solar_value(lat, lon, solar_raster, transform, max_search_radius=5):
        row, col = rowcol(transform, lon, lat)
        try:
            if 0 <= row < solar_raster.shape[0] and 0 <= col < solar_raster.shape[1]:
                solar_value = solar_raster[row, col]
                if not np.isnan(solar_value):
                    return solar_value
            for radius in range(1, max_search_radius + 1):
                for drow in range(-radius, radius + 1):
                    for dcol in range(-radius, radius + 1):
                        new_row, new_col = row + drow, col + dcol
                        if 0 <= new_row < solar_raster.shape[0] and 0 <= new_col < solar_raster.shape[1]:
                            solar_value = solar_raster[new_row, new_col]
                            if not np.isnan(solar_value):
                                return solar_value
            return None
        except IndexError:
            return None
    
    # Load wind capacity factor data
    wind_tif_path = os.path.join(config['data_dir'], "USA_capacity-factor_IEC1.tif")
    try:
        with rasterio.open(wind_tif_path) as src:
            wind_data = src.read(1)
            transform = src.transform
        
        plant_location['wind_cf'] = plant_location.apply(
            lambda row: get_closest_wind_value(row['Latitude'], row['Longitude'], wind_data, transform),
            axis=1
        )
        logger.info(f"Added wind capacity factors to {len(plant_location)} plants")
    except FileNotFoundError:
        logger.error(f"Wind resource data file not found at {wind_tif_path}")
        # Continue without wind data
        plant_location['wind_cf'] = np.nan
    
    # Load solar PVOUT data
    solar_tif_path = os.path.join(config['data_dir'], "PVOUT_2.tif")
    try:
        with rasterio.open(solar_tif_path) as src_solar:
            solar_data = src_solar.read(1)
            solar_transform = src_solar.transform
        
        plant_location['pvout'] = plant_location.apply(
            lambda row: get_closest_solar_value(row['Latitude'], row['Longitude'], solar_data, solar_transform),
            axis=1
        )
        logger.info(f"Added solar PVOUT data to {len(plant_location)} plants")
    except FileNotFoundError:
        logger.error(f"Solar resource data file not found at {solar_tif_path}")
        # Continue without solar data
        plant_location['pvout'] = np.nan
    
    if config['save_intermediates']:
        plant_location.to_csv(os.path.join(config['output_dir'], 'plant_locations_with_resources.csv'), index=False)
    
    return plant_location

def calculate_macc(annual_emissions, facility_level_df, plant_location, technology_df, config):
    """
    Calculate Marginal Abatement Cost Curve (MACC) for power plant transitions
    
    Args:
        annual_emissions: DataFrame of annual emissions data
        facility_level_df: DataFrame of facility-level attributes
        plant_location: DataFrame of plant locations with renewable resources
        technology_df: DataFrame of technology data
        config: Configuration dictionary
        
    Returns:
        DataFrame of merged data with MACC calculations
    """
    # Merge datasets to get required information
    merged_df = annual_emissions.merge(facility_level_df, on="Facility ID", how="left")
    merged_df = merged_df.merge(plant_location, on="Facility ID", how="left")
    
    # Check for missing values after merge
    missing_wind = merged_df['wind_cf'].isna().sum()
    missing_solar = merged_df['pvout'].isna().sum()
    if missing_wind > 0 or missing_solar > 0:
        logger.warning(f"Missing wind CF for {missing_wind} plants, missing solar PVOUT for {missing_solar} plants")
    
    # Fill missing values with reasonable defaults (national averages)
    if missing_wind > 0:
        merged_df['wind_cf'].fillna(0.35, inplace=True)
    if missing_solar > 0:
        merged_df['pvout'].fillna(1300, inplace=True)
    
    # Calculate required sizes for Wind, Solar, and NGCC plants
    merged_df["Wind Size (MW)"] = merged_df["Gross Load (MWh)"] / (merged_df["wind_cf"] * 8760)
    merged_df["Solar Size (MW)"] = merged_df.apply(
        lambda row: (row["Gross Load (MWh)"] * 1000 / row["pvout"]) / 1000 if row["pvout"] > 0 else None, 
        axis=1
    )
    merged_df["NGCC Size (MW)"] = merged_df["Gross Load (MWh)"] / (config['ngcc_cf'] * 8760)
    
    # Get technology costs and parameters
    ngcc_data = technology_df[technology_df["Technology"] == "Combined cycle"]
    ngcc_cost_per_kw = ngcc_data["Total overnight cost (2022$/kW)"].values[0]
    ngcc_heat_rate = ngcc_data["Heat rate (Btu/kWh)"].values[0]
    ngcc_fixed_om = ngcc_data["Fixed O&M (2022$/kW)"].values[0]
    ngcc_var_om = ngcc_data["Variable O&M (2022$/MWh)"].values[0]
    
    wind_data = technology_df[technology_df["Technology"] == "Wind"]
    wind_cost_per_kw = wind_data["Total overnight cost (2022$/kW)"].values[0]
    wind_fixed_om = wind_data["Fixed O&M (2022$/kW)"].values[0]
    wind_var_om = wind_data["Variable O&M (2022$/MWh)"].values[0]
    
    solar_data = technology_df[technology_df["Technology"] == "Solar"]
    solar_cost_per_kw = solar_data["Total overnight cost (2022$/kW)"].values[0]
    solar_fixed_om = solar_data["Fixed O&M (2022$/kW)"].values[0]
    solar_var_om = solar_data["Variable O&M (2022$/MWh)"].values[0]
    
    # Get existing plant parameters based on technology
    def get_om_costs(technology):
        tech_data = technology_df[technology_df["Technology"] == technology]
        if len(tech_data) > 0:
            return (
                tech_data["Fixed O&M (2022$/kW)"].values[0],
                tech_data["Variable O&M (2022$/MWh)"].values[0],
                tech_data["Fuel Cost ($/MWh)"].values[0]
            )
        return 0, 0, 0
    
    # Calculate capital costs
    merged_df["Wind Capital Cost ($)"] = merged_df["Wind Size (MW)"] * 1000 * wind_cost_per_kw
    merged_df["Solar Capital Cost ($)"] = merged_df["Solar Size (MW)"] * 1000 * solar_cost_per_kw
    merged_df["NGCC Capital Cost ($)"] = merged_df["NGCC Size (MW)"] * 1000 * ngcc_cost_per_kw
    
    storage_cost_per_kw = 1270  # $/kW from AEO2023 report
    storage_size_mw_wind = merged_df["Wind Size (MW)"] * (60 / 100)
    storage_size_mw_solar = merged_df["Wind Size (MW)"] * (60 / 100)

    merged_df["Wind Capital Cost ($)"] = merged_df["Wind Size (MW)"] * 1000 * wind_cost_per_kw + storage_size_mw_wind * 1000 * storage_cost_per_kw
    merged_df["Solar Capital Cost ($)"] = merged_df["Solar Size (MW)"] * 1000 * solar_cost_per_kw + storage_size_mw_solar * 1000 * storage_cost_per_kw
    
    # Calculate remaining capital cost for plants less than 30 years old
    merged_df["Existing Capital Cost ($)"] = merged_df["Total Capital Cost (2024$/kW)"] * merged_df["Total Nameplate Capacity (MW)"] * 1000
    merged_df["Remaining Capital Cost ($)"] = merged_df.apply(
        lambda row: row["Existing Capital Cost ($)"] * ((1 + config['discount_rate']) ** (config['plant_lifetime'] - row["Age"])) 
        if row["Age"] < config['plant_lifetime'] else 0, 
        axis=1
    )
    
    # Calculate O&M costs for new technologies
    merged_df["Fixed O&M - Wind ($/year)"] = merged_df["Wind Size (MW)"] * 1000 * wind_fixed_om
    merged_df["Fixed O&M - Solar ($/year)"] = merged_df["Solar Size (MW)"] * 1000 * solar_fixed_om
    merged_df["Fixed O&M - NGCC ($/year)"] = merged_df["NGCC Size (MW)"] * 1000 * ngcc_fixed_om
    
    merged_df["Variable O&M - Wind ($/year)"] = merged_df["Gross Load (MWh)"] * wind_var_om
    merged_df["Variable O&M - Solar ($/year)"] = merged_df["Gross Load (MWh)"] * solar_var_om
    merged_df["Variable O&M - NGCC ($/year)"] = merged_df["Gross Load (MWh)"] * ngcc_var_om
    
    # Calculate O&M costs for existing plant
    merged_df["Fixed O&M - Existing ($/year)"] = merged_df.apply(
        lambda row: row["Total Nameplate Capacity (MW)"] * 1000 * get_om_costs(row["Technology"])[0], 
        axis=1
    )
    merged_df["Variable O&M - Existing ($/year)"] = merged_df.apply(
        lambda row: row["Gross Load (MWh)"] * get_om_costs(row["Technology"])[1],
        axis=1
    )
    merged_df["Fuel costs ($/MWh)"] = merged_df.apply(
        lambda row: row["Gross Load (MWh)"] * get_om_costs(row["Technology"])[2],
        axis=1
    )
    
    # Calculate fuel costs
    gas_price = config['gas_price']  # $/MMBtu
    merged_df["NGCC Fuel Cost ($/year)"] = (merged_df["Gross Load (MWh)"] * ngcc_heat_rate * 1000 / 1e6) * gas_price
    merged_df["Existing Fuel Cost ($/year)"] = merged_df["Fuel costs ($/MWh)"] * merged_df["Gross Load (MWh)"]
    
    # Annualize costs
    discount_rate = config['discount_rate']
    plant_lifetime = config['plant_lifetime']
    annuity_factor = (discount_rate * (1 + discount_rate) ** plant_lifetime) / ((1 + discount_rate) ** plant_lifetime - 1)
    
    merged_df["Annualized Capital - Wind ($)"] = merged_df["Wind Capital Cost ($)"] * annuity_factor
    merged_df["Annualized Capital - Solar ($)"] = merged_df["Solar Capital Cost ($)"] * annuity_factor
    merged_df["Annualized Capital - NGCC ($)"] = merged_df["NGCC Capital Cost ($)"] * annuity_factor
    merged_df["Annualized Capital - Existing ($)"] = merged_df["Remaining Capital Cost ($)"] * annuity_factor
    
    # Convert old emissions from short tons to metric tons
    short_to_tonnes = 0.9071847
    merged_df["Old Emissions (tonnes CO2)"] = (merged_df["CO2 Mass (short tons)"]) * short_to_tonnes
    merged_df["New Emissions (tonnes CO2)"] = (merged_df["Gross Load (MWh)"] * ngcc_heat_rate * 1000 * 0.05291 / 1e6)
    merged_df["Delta Emissions (tonnes CO2)"] = merged_df["Old Emissions (tonnes CO2)"] - merged_df["New Emissions (tonnes CO2)"]
    
    # Calculate net costs for each component
    # Wind
    merged_df["Net Capital Cost - Wind ($/year)"] = merged_df["Annualized Capital - Wind ($)"] - merged_df["Annualized Capital - Existing ($)"]
    merged_df["Net Fixed O&M - Wind ($/year)"] = merged_df["Fixed O&M - Wind ($/year)"] - merged_df["Fixed O&M - Existing ($/year)"]
    merged_df["Net Variable O&M - Wind ($/year)"] = merged_df["Variable O&M - Wind ($/year)"] - merged_df["Variable O&M - Existing ($/year)"]
    merged_df["Net Fuel Cost - Wind ($/year)"] = 0 - merged_df["Existing Fuel Cost ($/year)"]
    
    # Solar
    merged_df["Net Capital Cost - Solar ($/year)"] = merged_df["Annualized Capital - Solar ($)"] - merged_df["Annualized Capital - Existing ($)"]
    merged_df["Net Fixed O&M - Solar ($/year)"] = merged_df["Fixed O&M - Solar ($/year)"] - merged_df["Fixed O&M - Existing ($/year)"]
    merged_df["Net Variable O&M - Solar ($/year)"] = merged_df["Variable O&M - Solar ($/year)"] - merged_df["Variable O&M - Existing ($/year)"]
    merged_df["Net Fuel Cost - Solar ($/year)"] = 0 - merged_df["Existing Fuel Cost ($/year)"]
    
    # NGCC
    merged_df["Net Capital Cost - NGCC ($/year)"] = merged_df["Annualized Capital - NGCC ($)"] - merged_df["Annualized Capital - Existing ($)"]
    merged_df["Net Fixed O&M - NGCC ($/year)"] = merged_df["Fixed O&M - NGCC ($/year)"] - merged_df["Fixed O&M - Existing ($/year)"]
    merged_df["Net Variable O&M - NGCC ($/year)"] = merged_df["Variable O&M - NGCC ($/year)"] - merged_df["Variable O&M - Existing ($/year)"]
    merged_df["Net Fuel Cost - NGCC ($/year)"] = merged_df["NGCC Fuel Cost ($/year)"] - merged_df["Existing Fuel Cost ($/year)"]
    
    # Calculate total cost per year for each technology
    merged_df["Total Cost - Wind ($/year)"] = (
        merged_df["Net Capital Cost - Wind ($/year)"] +
        merged_df["Net Fixed O&M - Wind ($/year)"] +
        merged_df["Net Variable O&M - Wind ($/year)"] +
        merged_df["Net Fuel Cost - Wind ($/year)"]
    )
    
    merged_df["Total Cost - Solar ($/year)"] = (
        merged_df["Net Capital Cost - Solar ($/year)"] +
        merged_df["Net Fixed O&M - Solar ($/year)"] +
        merged_df["Net Variable O&M - Solar ($/year)"] +
        merged_df["Net Fuel Cost - Solar ($/year)"]
    )
    
    merged_df["Total Cost - NGCC ($/year)"] = (
        merged_df["Net Capital Cost - NGCC ($/year)"] +
        merged_df["Net Fixed O&M - NGCC ($/year)"] +
        merged_df["Net Variable O&M - NGCC ($/year)"] +
        merged_df["Net Fuel Cost - NGCC ($/year)"]
    )
    
    # Calculate MACC for each technology
    merged_df["MACC - Wind ($/tonne CO2)"] = merged_df["Total Cost - Wind ($/year)"] / merged_df["Old Emissions (tonnes CO2)"]
    merged_df["MACC - Solar ($/tonne CO2)"] = merged_df["Total Cost - Solar ($/year)"] / merged_df["Old Emissions (tonnes CO2)"]
    merged_df["MACC - NGCC ($/tonne CO2)"] = merged_df["Total Cost - NGCC ($/year)"] / merged_df["Delta Emissions (tonnes CO2)"]
    
    if config['save_intermediates']:
        merged_df.to_csv(os.path.join(config['output_dir'], 'macc_calculations.csv'), index=False)
    
    return merged_df

def create_macc_curve(merged_df, config):
    """
    Create Marginal Abatement Cost Curve (MACC) data
    
    Args:
        merged_df: DataFrame with MACC calculations
        config: Configuration dictionary
        
    Returns:
        DataFrame with MACC curve data
    """
    # Process data to find lowest MACC for each facility
    def get_lowest_macc_option(row):
        maccs = {
            'Wind': row['MACC - Wind ($/tonne CO2)'],
            'Solar': row['MACC - Solar ($/tonne CO2)'],
            'NGCC': row['MACC - NGCC ($/tonne CO2)']
        }
        valid_maccs = {k: v for k, v in maccs.items() if pd.notnull(v) and not np.isinf(v)}
        if not valid_maccs:
            return None, None
        best_option = min(valid_maccs.items(), key=lambda x: x[1])
        return best_option[0], best_option[1]
    
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
    
    # Create transition data
    transitions_data = []
    for _, row in merged_df.iterrows():
        best_tech, macc = get_lowest_macc_option(row)
        if best_tech and config['macc_filter_min'] <= macc <= config['macc_filter_max']:
            from_fuel = get_fuel_category(row['Technology'])
            emissions = row['Old Emissions (tonnes CO2)'] if best_tech in ['Wind', 'Solar'] else row['Delta Emissions (tonnes CO2)']
            
            transition_name = f"{from_fuel} to {best_tech}"
            transitions_data.append({
                'facility_id': row['Facility ID'],
                'macc': macc,
                'emissions': abs(emissions),
                'transition': transition_name,
                'from_fuel': from_fuel,
                'to_tech': best_tech,
                'nameplate_capacity': row['Total Nameplate Capacity (MW)'],
                'annual_generation': row['Gross Load (MWh)']
            })
    
    # Convert to DataFrame and sort by MACC
    df = pd.DataFrame(transitions_data)
    if df.empty:
        logger.warning("No valid transitions found within MACC filter range")
        return pd.DataFrame()
        
    df = df.sort_values('macc')
    
    # Calculate cumulative emissions
    df['cumulative_end'] = df['emissions'].cumsum()
    df['cumulative_start'] = df['cumulative_end'] - df['emissions']
    
    if config['save_intermediates']:
        df.to_csv(os.path.join(config['output_dir'], 'macc_curve_data.csv'), index=False)
    
    return df

def plot_macc_curve(macc_data, config):
    """
    Plot Marginal Abatement Cost Curve (MACC)
    
    Args:
        macc_data: DataFrame with MACC curve data
        config: Configuration dictionary
        
    Returns:
        Matplotlib figure
    """
    if macc_data.empty:
        logger.error("Cannot create MACC plot: No valid data")
        return None
    
    # Set up colors for different transitions
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
    
    # Create MACC Plot
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot bars
    used_labels = set()
    for _, row in macc_data.iterrows():
        label = row['transition'] if row['transition'] not in used_labels else None
        ax.bar(x=(row['cumulative_start'] + row['cumulative_end']) / (2*1e9),
               height=row['macc'],
               width=row['emissions']/1e9,
               color=color_dict.get(row['transition'], '#808080'),
               label=label,
               linewidth=0.6, alpha=0.9)
        if label:
            used_labels.add(row['transition'])
    
    # Customize the scientific-style plot
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Cumulative CO$_2$ displaced (Gigatonnes)', fontsize=14)
    ax.set_ylabel('Abatement cost per tonne CO$_2$ ($/tonne)', fontsize=14)
    ax.set_title('Marginal abatement cost curve (MACC)', fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add faint gridlines on the y-axis
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Fine-tuning ticks and labels
    ax.tick_params(axis='both', labelsize=12)
    
    # Improve legend positioning
    ax.legend(title="Technology Transition", fontsize=12, title_fontsize=12, 
              loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if requested
    if config.get('save_figures', True):
        fig_path = os.path.join(config['output_dir'], 'macc_curve.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"MACC curve saved to {fig_path}")
    
    return fig

def create_transition_map(merged_df, config, output_path=None, color_dict=None):
    """
    Create a map visualization of power plant transitions across the US
    
    Args:
        merged_df: DataFrame with plant data and MACC calculations
        shapefile_path: Path to the US states shapefile
        output_path: Path to save the figure (optional)
        color_dict: Dictionary mapping transitions to colors (optional)
        
    Returns:
        matplotlib figure object
    """
    import pandas as pd
    import numpy as np
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import os

    shapefile_path = os.path.join(config['data_dir'], 'raw/ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp')
    
    # Default color dictionary if not provided
    if color_dict is None:
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
    
    # Helper functions for categorizing plants
    def get_lowest_macc_option(row):
        maccs = {
            'Wind': row['MACC - Wind ($/tonne CO2)'],
            'Solar': row['MACC - Solar ($/tonne CO2)'],
            'NGCC': row['MACC - NGCC ($/tonne CO2)']
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
    
    # Step 1: Load US States Shapefile
    us_states = gpd.read_file(shapefile_path)
    
    # Filter for continental US states only
    continental_us = us_states[(us_states['admin'] == 'United States of America') & 
                              (~us_states['iso_3166_2'].isin(['AK', 'HI']))]
    
    # Step 2: Prepare and filter the data from merged_df
    plant_map_df = merged_df[['State', 'Latitude', 'Longitude', 'Technology', 
                              'Old Emissions (tonnes CO2)', 'MACC - Wind ($/tonne CO2)',
                              'MACC - Solar ($/tonne CO2)', 'MACC - NGCC ($/tonne CO2)']].copy()
    
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
    for tech in ['Solar', 'Wind', 'NGCC']:
        for fuel in ['Coal', 'Gas', 'Oil', 'Other']:
            transition = f"{fuel} to {tech}"
            if transition in plant_map_df['Transition'].unique():
                ordered_transitions.append(transition)
    
    # Step 3: Ensure valid transitions
    plant_map_df = plant_map_df[plant_map_df['Transition'].isin(ordered_transitions)]
    
    # Create figure 
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])  # [left, bottom, width, height]
    
    # Set the map extent to continental US
    ax.set_xlim([-125, -65])
    ax.set_ylim([25, 50])
    
    # Remove box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Plot continental US States as a base layer
    continental_us.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.6, alpha=0.4)
    
    # Organize transitions by target technology
    solar_transitions = [t for t in ordered_transitions if 'to Solar' in t]
    wind_transitions = [t for t in ordered_transitions if 'to Wind' in t]
    ngcc_transitions = [t for t in ordered_transitions if 'to NGCC' in t]
    
    # Reorganize transitions
    final_transitions = solar_transitions + wind_transitions + ngcc_transitions
    
    # Create scatter plots and collect handles and labels for legend
    handles = []
    labels = []
    for transition in final_transitions:
        subset = plant_map_df[plant_map_df['Transition'] == transition]
        sizes = np.sqrt(subset['Old Emissions (tonnes CO2)']) / 8 
        scatter = ax.scatter(subset['Longitude'], subset['Latitude'], 
                            s=sizes, c=color_dict.get(transition, '#808080'), label=transition, 
                            alpha=0.95, edgecolors='black', linewidth=0.6)
        # Create a proxy artist for the legend with fixed size
        proxy = plt.scatter([], [], c=color_dict.get(transition, '#808080'), 
                           s=100, label=transition,  # Fixed size for legend
                           alpha=0.95, edgecolors='black', linewidth=0.6)
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
    
    # Create main transitions legend
    legend1 = fig.legend(handles, labels,
                        title="Technology Transition", 
                        fontsize=12,
                        title_fontsize=12,
                        bbox_to_anchor=(0.5, 0.2),
                        loc='center',
                        ncol=3)
    
    # Create size legend
    legend2 = ax.legend(handles=size_legend_handles,
                       title="Annual Emissions",
                       fontsize=12,
                       title_fontsize=12,
                       loc='lower right')
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title
    plt.suptitle('U.S. Power Plant Transition Opportunities', fontsize=16, y=0.95)
    plt.figtext(0.5, 0.9, 'Cost-optimal technology replacements by plant location and size', 
                ha='center', fontsize=14)
    
    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def summarize_results(macc_data, config):
    """
    Generate summary statistics of MACC results
    
    Args:
        macc_data: DataFrame with MACC curve data
        config: Configuration dictionary
        
    Returns:
        Dictionary of summary statistics
    """
    if macc_data.empty:
        return {}
    
    # Calculate summary statistics
    total_emissions = macc_data['emissions'].sum() / 1e9  # Convert to Gt
    negative_cost_df = macc_data[macc_data['macc'] <= 0]
    negative_cost_emissions = negative_cost_df['emissions'].sum() / 1e9  # Convert to Gt
    
    # Transitions by type
    transitions = macc_data.groupby('transition').agg({
        'emissions': 'sum',
        'macc': 'mean',
        'facility_id': 'count'
    }).reset_index()
    
    transitions['emissions'] = transitions['emissions'] / 1e9  # Convert to Gt
    transitions = transitions.sort_values('emissions', ascending=False)
    
    # Transitions by source fuel
    by_source = macc_data.groupby('from_fuel').agg({
        'emissions': 'sum',
        'facility_id': 'count'
    }).reset_index()
    
    by_source['emissions'] = by_source['emissions'] / 1e9  # Convert to Gt
    by_source['percentage'] = by_source['emissions'] / total_emissions * 100
    
    # Transitions by target technology
    by_target = macc_data.groupby('to_tech').agg({
        'emissions': 'sum',
        'facility_id': 'count'
    }).reset_index()
    
    by_target['emissions'] = by_target['emissions'] / 1e9  # Convert to Gt
    by_target['percentage'] = by_target['emissions'] / total_emissions * 100
    
    # Cost-effectiveness bands
    cost_bands = [
        ('Negative cost', -float('inf'), 0),
        ('0-20 $/tonne', 0, 20),
        ('20-50 $/tonne', 20, 50),
        ('50-100 $/tonne', 50, 100),
        ('> 100 $/tonne', 100, float('inf'))
    ]
    
    cost_distribution = []
    for band_name, lower, upper in cost_bands:
        band_df = macc_data[(macc_data['macc'] > lower) & (macc_data['macc'] <= upper)]
        band_emissions = band_df['emissions'].sum() / 1e9
        band_percentage = band_emissions / total_emissions * 100 if total_emissions > 0 else 0
        cost_distribution.append({
            'cost_band': band_name,
            'emissions_gt': band_emissions,
            'percentage': band_percentage,
            'facility_count': len(band_df)
        })
    
    # Create summary dictionary
    summary = {
        'total_emissions_gt': total_emissions,
        'negative_cost_emissions_gt': negative_cost_emissions,
        'negative_cost_percentage': negative_cost_emissions / total_emissions * 100 if total_emissions > 0 else 0,
        'facility_count': len(macc_data['facility_id'].unique()),
        'transitions': transitions.to_dict('records'),
        'by_source': by_source.to_dict('records'),
        'by_target': by_target.to_dict('records'),
        'cost_distribution': cost_distribution
    }
    
    # Save summary to file
    if config.get('save_summary', True):
        import json
        summary_path = os.path.join(config['output_dir'], 'summary_results.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary results saved to {summary_path}")
    
    return summary

# Update the call to aggregate_facility_data in run_scenario
def run_scenario(config=None, scenario_name=None):
    """
    Run a complete MACC analysis scenario
    
    Args:
        config: Configuration dictionary or path to config file
        scenario_name: Name for this scenario (used in output files)
        
    Returns:
        Tuple of (merged_df, macc_data, summary)
    """
    # Load configuration
    if isinstance(config, str):
        scenario_config = load_config(config)
    elif isinstance(config, dict):
        scenario_config = fix_paths_for_notebook(config) if 'fix_paths_for_notebook' in globals() else config
    else:
        scenario_config = load_config()
    
    # Customize output directory for this scenario
    if scenario_name:
        scenario_config['output_dir'] = os.path.join(scenario_config['output_dir'], scenario_name)
        os.makedirs(scenario_config['output_dir'], exist_ok=True)
    
    logger.info(f"Running MACC analysis scenario: {scenario_name or 'default'}")
    
    # Load data
    annual_emissions, plants = load_emissions_data(scenario_config)
    plant_df_filtered = load_plant_data(scenario_config, plants)
    aggregated_facility_attributes, unit_level_attributes = aggregate_facility_data(plant_df_filtered, scenario_config)
    capital_cost_df = load_capital_cost_data(scenario_config)
    facility_level_df = calculate_capital_costs(unit_level_attributes, capital_cost_df, scenario_config)
    technology_df = load_technology_data()
    plant_location = load_plant_location_data(scenario_config, plants)
    plant_location = get_renewable_resources(plant_location, scenario_config)
    
    # Calculate MACC
    merged_df = calculate_macc(annual_emissions, facility_level_df, plant_location, technology_df, scenario_config)
    macc_data = create_macc_curve(merged_df, scenario_config)
    
    # Plot and summarize
    fig = plot_macc_curve(macc_data, scenario_config)
    summary = summarize_results(macc_data, scenario_config)
    
    logger.info(f"Completed MACC analysis scenario: {scenario_name or 'default'}")
    return merged_df, macc_data, summary

def run_sensitivity_analysis(base_config, sensitivity_params, scenario_prefix="sensitivity"):
    """
    Run sensitivity analysis on key parameters
    
    Args:
        base_config: Base configuration dictionary
        sensitivity_params: Dictionary of parameters to vary, each with a list of values
        scenario_prefix: Prefix for scenario names
        
    Returns:
        Dictionary of scenario names to summary results
    """
    # Load base configuration
    if isinstance(base_config, str):
        config = load_config(base_config)
    elif isinstance(base_config, dict):
        config = base_config
    else:
        config = load_config()
    
    results = {}
    
    # Generate parameter combinations (one at a time)
    for param, values in sensitivity_params.items():
        for value in values:
            # Create a variation of the base config
            scenario_config = config.copy()
            scenario_config[param] = value
            
            # Create scenario name
            scenario_name = f"{scenario_prefix}_{param}_{value}"
            
            # Run the scenario
            logger.info(f"Running sensitivity scenario: {scenario_name}")
            _, macc_data, summary = run_scenario(scenario_config, scenario_name)
            
            # Store results
            results[scenario_name] = summary
    
    # Compare results
    comparison_path = os.path.join(config['output_dir'], f"{scenario_prefix}_comparison.json")
    with open(comparison_path, 'w') as f:
        import json
        json.dump(results, f, indent=2)
    
    return results

def calculate_macc_with_ccs(annual_emissions, facility_level_df, plant_location, technology_df, config):
    """
    Calculate Marginal Abatement Cost Curve (MACC) for power plant transitions with CCS options
    
    This variant replaces natural gas with natural gas + CCS as an option
    
    Args:
        annual_emissions: DataFrame of annual emissions data
        facility_level_df: DataFrame of facility-level attributes
        plant_location: DataFrame of plant locations with renewable resources
        technology_df: DataFrame of technology data
        config: Configuration dictionary
        
    Returns:
        DataFrame of merged data with MACC calculations
    """
    # Merge datasets to get required information
    merged_df = annual_emissions.merge(facility_level_df, on="Facility ID", how="left")
    merged_df = merged_df.merge(plant_location, on="Facility ID", how="left")
    
    # Check for missing values after merge
    missing_wind = merged_df['wind_cf'].isna().sum()
    missing_solar = merged_df['pvout'].isna().sum()
    if missing_wind > 0 or missing_solar > 0:
        logger.warning(f"Missing wind CF for {missing_wind} plants, missing solar PVOUT for {missing_solar} plants")
    
    # Fill missing values with reasonable defaults (national averages)
    merged_df['wind_cf'] = merged_df['wind_cf'].fillna(0.35)
    merged_df['pvout'] = merged_df['pvout'].fillna(1300)
    
    # Calculate required sizes for Wind, Solar, and NGCC-CCS plants
    merged_df["Wind Size (MW)"] = merged_df["Gross Load (MWh)"] / (merged_df["wind_cf"] * 8760)
    merged_df["Solar Size (MW)"] = merged_df.apply(
        lambda row: (row["Gross Load (MWh)"] * 1000 / row["pvout"]) / 1000 if row["pvout"] > 0 else None, 
        axis=1
    )
    merged_df["NGCC-CCS Size (MW)"] = merged_df["Gross Load (MWh)"] / (config.get('ngcc_ccs_cf', 0.55) * 8760)
    
    # Get technology costs and parameters for wind and solar (same as before)
    wind_data = technology_df[technology_df["Technology"] == "Wind"]
    wind_cost_per_kw = wind_data["Total overnight cost (2022$/kW)"].values[0]
    wind_fixed_om = wind_data["Fixed O&M (2022$/kW)"].values[0]
    wind_var_om = wind_data["Variable O&M (2022$/MWh)"].values[0]
    
    solar_data = technology_df[technology_df["Technology"] == "Solar"]
    solar_cost_per_kw = solar_data["Total overnight cost (2022$/kW)"].values[0]
    solar_fixed_om = solar_data["Fixed O&M (2022$/kW)"].values[0]
    solar_var_om = solar_data["Variable O&M (2022$/MWh)"].values[0]
    
    # Get technology costs and parameters for NGCC with CCS
    ccs_data = technology_df[technology_df["Technology"] == "Combined-cycle with 90% CCS"]
    ccs_cost_per_kw = ccs_data["Total overnight cost (2022$/kW)"].values[0]
    ccs_heat_rate = ccs_data["Heat rate (Btu/kWh)"].values[0]
    ccs_fixed_om = ccs_data["Fixed O&M (2022$/kW)"].values[0]
    ccs_var_om = ccs_data["Variable O&M (2022$/MWh)"].values[0]
    
    # Get existing plant parameters based on technology
    def get_om_costs(technology):
        tech_data = technology_df[technology_df["Technology"] == technology]
        if len(tech_data) > 0:
            return (
                tech_data["Fixed O&M (2022$/kW)"].values[0],
                tech_data["Variable O&M (2022$/MWh)"].values[0],
                tech_data["Fuel Cost ($/MWh)"].values[0]
            )
        return 0, 0, 0
    
    # Calculate capital costs
    merged_df["Wind Capital Cost ($)"] = merged_df["Wind Size (MW)"] * 1000 * wind_cost_per_kw
    merged_df["Solar Capital Cost ($)"] = merged_df["Solar Size (MW)"] * 1000 * solar_cost_per_kw
    merged_df["NGCC-CCS Capital Cost ($)"] = merged_df["NGCC-CCS Size (MW)"] * 1000 * ccs_cost_per_kw
    
    # Calculate remaining capital cost for plants less than 30 years old
    merged_df["Existing Capital Cost ($)"] = merged_df["Total Capital Cost (2024$/kW)"] * merged_df["Total Nameplate Capacity (MW)"] * 1000
    merged_df["Remaining Capital Cost ($)"] = merged_df.apply(
        lambda row: row["Existing Capital Cost ($)"] * ((1 + config['discount_rate']) ** (config['plant_lifetime'] - row["Age"])) 
        if row["Age"] < config['plant_lifetime'] else 0, 
        axis=1
    )
    
    # Calculate O&M costs for new technologies
    merged_df["Fixed O&M - Wind ($/year)"] = merged_df["Wind Size (MW)"] * 1000 * wind_fixed_om
    merged_df["Fixed O&M - Solar ($/year)"] = merged_df["Solar Size (MW)"] * 1000 * solar_fixed_om
    merged_df["Fixed O&M - NGCC-CCS ($/year)"] = merged_df["NGCC-CCS Size (MW)"] * 1000 * ccs_fixed_om
    
    merged_df["Variable O&M - Wind ($/year)"] = merged_df["Gross Load (MWh)"] * wind_var_om
    merged_df["Variable O&M - Solar ($/year)"] = merged_df["Gross Load (MWh)"] * solar_var_om
    merged_df["Variable O&M - NGCC-CCS ($/year)"] = merged_df["Gross Load (MWh)"] * ccs_var_om
    
    # Calculate O&M costs for existing plant
    merged_df["Fixed O&M - Existing ($/year)"] = merged_df.apply(
        lambda row: row["Total Nameplate Capacity (MW)"] * 1000 * get_om_costs(row["Technology"])[0], 
        axis=1
    )
    merged_df["Variable O&M - Existing ($/year)"] = merged_df.apply(
        lambda row: row["Gross Load (MWh)"] * get_om_costs(row["Technology"])[1],
        axis=1
    )
    merged_df["Fuel costs ($/MWh)"] = merged_df.apply(
        lambda row: row["Gross Load (MWh)"] * get_om_costs(row["Technology"])[2],
        axis=1
    )
    
    # Calculate fuel costs
    gas_price = config['gas_price']  # $/MMBtu
    merged_df["NGCC-CCS Fuel Cost ($/year)"] = (merged_df["Gross Load (MWh)"] * ccs_heat_rate * 1000 / 1e6) * gas_price
    merged_df["Existing Fuel Cost ($/year)"] = merged_df["Fuel costs ($/MWh)"] * merged_df["Gross Load (MWh)"]
    
    # Annualize costs
    discount_rate = config['discount_rate']
    plant_lifetime = config['plant_lifetime']
    annuity_factor = (discount_rate * (1 + discount_rate) ** plant_lifetime) / ((1 + discount_rate) ** plant_lifetime - 1)
    
    merged_df["Annualized Capital - Wind ($)"] = merged_df["Wind Capital Cost ($)"] * annuity_factor
    merged_df["Annualized Capital - Solar ($)"] = merged_df["Solar Capital Cost ($)"] * annuity_factor
    merged_df["Annualized Capital - NGCC-CCS ($)"] = merged_df["NGCC-CCS Capital Cost ($)"] * annuity_factor
    merged_df["Annualized Capital - Existing ($)"] = merged_df["Remaining Capital Cost ($)"] * annuity_factor
    
    # Convert old emissions from short tons to metric tons
    short_to_tonnes = 0.9071847
    merged_df["Old Emissions (tonnes CO2)"] = (merged_df["CO2 Mass (short tons)"]) * short_to_tonnes
    
    # Calculate new emissions (with CCS capturing 90% of CO2)
    ccs_capture_rate = 0.90  # 90% capture
    merged_df["New Emissions (tonnes CO2)"] = (merged_df["Gross Load (MWh)"] * ccs_heat_rate * 1000 * 0.05291 / 1e6) * (1 - ccs_capture_rate)
    merged_df["Delta Emissions (tonnes CO2)"] = merged_df["Old Emissions (tonnes CO2)"] - merged_df["New Emissions (tonnes CO2)"]
    
    # Calculate net costs for each component
    # Wind
    merged_df["Net Capital Cost - Wind ($/year)"] = merged_df["Annualized Capital - Wind ($)"] - merged_df["Annualized Capital - Existing ($)"]
    merged_df["Net Fixed O&M - Wind ($/year)"] = merged_df["Fixed O&M - Wind ($/year)"] - merged_df["Fixed O&M - Existing ($/year)"]
    merged_df["Net Variable O&M - Wind ($/year)"] = merged_df["Variable O&M - Wind ($/year)"] - merged_df["Variable O&M - Existing ($/year)"]
    merged_df["Net Fuel Cost - Wind ($/year)"] = 0 - merged_df["Existing Fuel Cost ($/year)"]
    
    # Solar
    merged_df["Net Capital Cost - Solar ($/year)"] = merged_df["Annualized Capital - Solar ($)"] - merged_df["Annualized Capital - Existing ($)"]
    merged_df["Net Fixed O&M - Solar ($/year)"] = merged_df["Fixed O&M - Solar ($/year)"] - merged_df["Fixed O&M - Existing ($/year)"]
    merged_df["Net Variable O&M - Solar ($/year)"] = merged_df["Variable O&M - Solar ($/year)"] - merged_df["Variable O&M - Existing ($/year)"]
    merged_df["Net Fuel Cost - Solar ($/year)"] = 0 - merged_df["Existing Fuel Cost ($/year)"]
    
    # NGCC-CCS
    merged_df["Net Capital Cost - NGCC-CCS ($/year)"] = merged_df["Annualized Capital - NGCC-CCS ($)"] - merged_df["Annualized Capital - Existing ($)"]
    merged_df["Net Fixed O&M - NGCC-CCS ($/year)"] = merged_df["Fixed O&M - NGCC-CCS ($/year)"] - merged_df["Fixed O&M - Existing ($/year)"]
    merged_df["Net Variable O&M - NGCC-CCS ($/year)"] = merged_df["Variable O&M - NGCC-CCS ($/year)"] - merged_df["Variable O&M - Existing ($/year)"]
    merged_df["Net Fuel Cost - NGCC-CCS ($/year)"] = merged_df["NGCC-CCS Fuel Cost ($/year)"] - merged_df["Existing Fuel Cost ($/year)"]
    
    # Calculate total cost per year for each technology
    merged_df["Total Cost - Wind ($/year)"] = (
        merged_df["Net Capital Cost - Wind ($/year)"] +
        merged_df["Net Fixed O&M - Wind ($/year)"] +
        merged_df["Net Variable O&M - Wind ($/year)"] +
        merged_df["Net Fuel Cost - Wind ($/year)"]
    )
    
    merged_df["Total Cost - Solar ($/year)"] = (
        merged_df["Net Capital Cost - Solar ($/year)"] +
        merged_df["Net Fixed O&M - Solar ($/year)"] +
        merged_df["Net Variable O&M - Solar ($/year)"] +
        merged_df["Net Fuel Cost - Solar ($/year)"]
    )
    
    merged_df["Total Cost - NGCC-CCS ($/year)"] = (
        merged_df["Net Capital Cost - NGCC-CCS ($/year)"] +
        merged_df["Net Fixed O&M - NGCC-CCS ($/year)"] +
        merged_df["Net Variable O&M - NGCC-CCS ($/year)"] +
        merged_df["Net Fuel Cost - NGCC-CCS ($/year)"]
    )
    
    # Calculate MACC for each technology
    merged_df["MACC - Wind ($/tonne CO2)"] = merged_df["Total Cost - Wind ($/year)"] / merged_df["Old Emissions (tonnes CO2)"]
    merged_df["MACC - Solar ($/tonne CO2)"] = merged_df["Total Cost - Solar ($/year)"] / merged_df["Old Emissions (tonnes CO2)"]
    merged_df["MACC - NGCC-CCS ($/tonne CO2)"] = merged_df["Total Cost - NGCC-CCS ($/year)"] / merged_df["Delta Emissions (tonnes CO2)"]
    
    if config.get('save_intermediates', False):
        output_dir = config.get('output_dir', 'output')
        merged_df.to_csv(os.path.join(output_dir, 'macc_calculations_ccs.csv'), index=False)
    
    return merged_df

def create_macc_curve_with_ccs(merged_df, config):
    """
    Create Marginal Abatement Cost Curve (MACC) data for CCS scenario
    
    Args:
        merged_df: DataFrame with MACC calculations
        config: Configuration dictionary
        
    Returns:
        DataFrame with MACC curve data
    """
    # Process data to find lowest MACC for each facility
    def get_lowest_macc_option(row):
        maccs = {
            'Wind': row['MACC - Wind ($/tonne CO2)'],
            'Solar': row['MACC - Solar ($/tonne CO2)'],
            'NGCC-CCS': row['MACC - NGCC-CCS ($/tonne CO2)']
        }
        valid_maccs = {k: v for k, v in maccs.items() if pd.notnull(v) and not np.isinf(v)}
        if not valid_maccs:
            return None, None
        best_option = min(valid_maccs.items(), key=lambda x: x[1])
        return best_option[0], best_option[1]
    
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
    
    # Create transition data
    transitions_data = []
    for _, row in merged_df.iterrows():
        best_tech, macc = get_lowest_macc_option(row)
        if best_tech and config['macc_filter_min'] <= macc <= config['macc_filter_max']:
            from_fuel = get_fuel_category(row['Technology'])
            emissions = row['Old Emissions (tonnes CO2)'] if best_tech in ['Wind', 'Solar'] else row['Delta Emissions (tonnes CO2)']
            
            transition_name = f"{from_fuel} to {best_tech}"
            transitions_data.append({
                'facility_id': row['Facility ID'],
                'macc': macc,
                'emissions': abs(emissions),
                'transition': transition_name,
                'from_fuel': from_fuel,
                'to_tech': best_tech,
                'nameplate_capacity': row['Total Nameplate Capacity (MW)'],
                'annual_generation': row['Gross Load (MWh)']
            })
    
    # Convert to DataFrame and sort by MACC
    df = pd.DataFrame(transitions_data)
    if df.empty:
        logger.warning("No valid transitions found within MACC filter range for CCS scenario")
        return pd.DataFrame()
        
    df = df.sort_values('macc')
    
    # Calculate cumulative emissions
    df['cumulative_end'] = df['emissions'].cumsum()
    df['cumulative_start'] = df['cumulative_end'] - df['emissions']
    
    if config.get('save_intermediates', False):
        output_dir = config.get('output_dir', 'output')
        df.to_csv(os.path.join(output_dir, 'macc_curve_data_ccs.csv'), index=False)
    
    return df

def plot_macc_curve_with_ccs(macc_data, config):
    """
    Plot Marginal Abatement Cost Curve (MACC) for CCS scenario
    
    Args:
        macc_data: DataFrame with MACC curve data
        config: Configuration dictionary
        
    Returns:
        Matplotlib figure
    """
    if macc_data.empty:
        logger.error("Cannot create MACC plot: No valid data")
        return None
    
    # Set up colors for different transitions
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
    
    # Create MACC Plot
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot bars
    used_labels = set()
    for _, row in macc_data.iterrows():
        label = row['transition'] if row['transition'] not in used_labels else None
        ax.bar(x=(row['cumulative_start'] + row['cumulative_end']) / (2*1e9),
               height=row['macc'],
               width=row['emissions']/1e9,
               color=color_dict.get(row['transition'], '#808080'),
               label=label,
               linewidth=0.6, alpha=0.9)
        if label:
            used_labels.add(row['transition'])
    
    # Customize the scientific-style plot
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Cumulative CO$_2$ displaced (Gigatonnes)', fontsize=14)
    ax.set_ylabel('Abatement cost per tonne CO$_2$ ($/tonne)', fontsize=14)
    ax.set_title('Marginal abatement cost curve with CCS scenario', fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add faint gridlines on the y-axis
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Fine-tuning ticks and labels
    ax.tick_params(axis='both', labelsize=12)
    
    # Improve legend positioning
    ax.legend(title="Technology Transition", fontsize=12, title_fontsize=12, 
              loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if requested
    if config.get('save_figures', True):
        output_dir = config.get('output_dir', 'output')
        fig_path = os.path.join(output_dir, 'macc_curve_ccs.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"MACC curve for CCS scenario saved to {fig_path}")
    
    return fig

def run_ccs_scenario(config=None, scenario_name=None):
    """
    Run a MACC analysis scenario with CCS options
    
    Args:
        config: Configuration dictionary or path to config file
        scenario_name: Name for this scenario (used in output files)
        
    Returns:
        Tuple of (merged_df, macc_data, summary)
    """
    # Load configuration
    if isinstance(config, str):
        scenario_config = load_config(config)
    elif isinstance(config, dict):
        scenario_config = fix_paths_for_notebook(config) if 'fix_paths_for_notebook' in globals() else config
    else:
        scenario_config = load_config()
    
    # Set scenario name if not provided
    if scenario_name is None:
        scenario_name = 'ccs_scenario'
    
    # Customize output directory for this scenario
    scenario_config['output_dir'] = os.path.join(scenario_config.get('output_dir', 'output'), scenario_name)
    os.makedirs(scenario_config['output_dir'], exist_ok=True)
    
    logger.info(f"Running MACC analysis with CCS scenario: {scenario_name}")
    
    # Load data
    annual_emissions, plants = load_emissions_data(scenario_config)
    plant_df_filtered = load_plant_data(scenario_config, plants)
    aggregated_facility_attributes, unit_level_attributes = aggregate_facility_data(plant_df_filtered, scenario_config)
    capital_cost_df = load_capital_cost_data(scenario_config)
    facility_level_df = calculate_capital_costs(unit_level_attributes, capital_cost_df, scenario_config)
    technology_df = load_technology_data()
    plant_location = load_plant_location_data(scenario_config, plants)
    plant_location = get_renewable_resources(plant_location, scenario_config)
    
    # Calculate MACC with CCS options
    merged_df = calculate_macc_with_ccs(annual_emissions, facility_level_df, plant_location, technology_df, scenario_config)
    macc_data = create_macc_curve_with_ccs(merged_df, scenario_config)
    
    # Plot and summarize
    fig = plot_macc_curve_with_ccs(macc_data, scenario_config)
    summary = summarize_results(macc_data, scenario_config)
    
    logger.info(f"Completed MACC analysis with CCS scenario: {scenario_name}")
    return merged_df, macc_data, summary

def create_transition_map_with_ccs(merged_df, config, output_path=None, title=None):
    """
    Create a map visualization of power plant transitions across the US with CCS scenario
    
    Args:
        merged_df: DataFrame with plant data and MACC calculations from CCS scenario
        config: Configuration dictionary with data paths
        output_path: Path to save the figure (optional)
        title: Custom title for the map (optional)
        
    Returns:
        matplotlib figure object
    """
    import pandas as pd
    import numpy as np
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import os
    
    # Set up colors for different transitions with CCS
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
    
    # Helper functions for categorizing plants
    def get_lowest_macc_option(row):
        maccs = {
            'Wind': row['MACC - Wind ($/tonne CO2)'],
            'Solar': row['MACC - Solar ($/tonne CO2)'],
            'NGCC-CCS': row['MACC - NGCC-CCS ($/tonne CO2)']
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
    
    # Get shapefile path from config
    shapefile_path = os.path.join(
        config['data_dir'], 
        'raw/ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp'
    )
    
    # Step 1: Load US States Shapefile
    us_states = gpd.read_file(shapefile_path)
    
    # Filter for continental US states only
    continental_us = us_states[(us_states['admin'] == 'United States of America') & 
                              (~us_states['iso_3166_2'].isin(['AK', 'HI']))]
    
    # Step 2: Prepare and filter the data from merged_df
    plant_map_df = merged_df[['State', 'Latitude', 'Longitude', 'Technology', 
                              'Old Emissions (tonnes CO2)', 'MACC - Wind ($/tonne CO2)',
                              'MACC - Solar ($/tonne CO2)', 'MACC - NGCC-CCS ($/tonne CO2)']].copy()
    
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
    for tech in ['Solar', 'Wind', 'NGCC-CCS']:
        for fuel in ['Coal', 'Gas', 'Oil', 'Other']:
            transition = f"{fuel} to {tech}"
            if transition in plant_map_df['Transition'].unique():
                ordered_transitions.append(transition)
    
    # Step 3: Ensure valid transitions
    plant_map_df = plant_map_df[plant_map_df['Transition'].isin(ordered_transitions)]
    
    # Create figure 
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])  # [left, bottom, width, height]
    
    # Set the map extent to continental US
    ax.set_xlim([-125, -65])
    ax.set_ylim([25, 50])
    
    # Remove box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Plot continental US States as a base layer
    continental_us.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.6, alpha=0.4)
    
    # Organize transitions by target technology
    solar_transitions = [t for t in ordered_transitions if 'to Solar' in t]
    wind_transitions = [t for t in ordered_transitions if 'to Wind' in t]
    ccs_transitions = [t for t in ordered_transitions if 'to NGCC-CCS' in t]
    
    # Reorganize transitions
    final_transitions = solar_transitions + wind_transitions + ccs_transitions
    
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
        proxy = plt.scatter([], [], c=color_dict.get(transition, '#808080'), 
                           s=100, label=transition,  # Fixed size for legend
                           alpha=0.95, edgecolors='black', linewidth=0.6)
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
    
    # Create main transitions legend
    legend1 = fig.legend(handles, labels,
                        title="Technology Transition", 
                        fontsize=12,
                        title_fontsize=12,
                        bbox_to_anchor=(0.5, 0.2),
                        loc='center',
                        ncol=3)
    
    # Create size legend
    legend2 = ax.legend(handles=size_legend_handles,
                       title="Annual Emissions",
                       fontsize=12,
                       title_fontsize=12,
                       loc='lower right')
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title
    if title:
        plt.suptitle(title, fontsize=16, y=0.95)
    else:
        plt.suptitle('U.S. Power Plant Transition with CCS Scenario', fontsize=16, y=0.95)
    
    plt.figtext(0.5, 0.9, 'Cost-optimal technology replacements including wind, solar, and NGCC with CCS', 
               ha='center', fontsize=14)
    
    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Map visualization saved to {output_path}")
    
    return fig

def create_enhanced_transition_map(merged_df, config, ccs_scenario=True, output_path=None):
    """
    Create an enhanced map visualization of power plant transitions overlaid on county renewable capacity factors
    
    Args:
        merged_df: DataFrame with plant data and MACC calculations
        config: Configuration dictionary with data paths
        ccs_scenario: Whether this is the CCS scenario (True) or baseline (False)
        output_path: Path to save the figure (optional)
        
    Returns:
        matplotlib figure object
    """
    import pandas as pd
    import numpy as np
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches
    import os
    from matplotlib.gridspec import GridSpec
    
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
    
    # Helper functions for categorizing plants
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
    
    # Get shapefile paths from config
    county_shapefile_path = os.path.join(
        config['data_dir'], 
        'raw/cb_2018_us_county_500k/cb_2018_us_county_500k.shp'
    )
    
    # Step 1: Load US Counties Shapefile
    us_counties = gpd.read_file(county_shapefile_path)
    
    # Filter for continental US counties only - assuming CONUS states have STATEFP <= 56 and not including AK (02) or HI (15)
    conus_states = [str(i).zfill(2) for i in range(1, 57) if i not in [2, 15]]
    continental_counties = us_counties[us_counties['STATEFP'].isin(conus_states)]
    
    # Step 2: Create a county-level renewable capacity factor dataset
    # This would ideally come from actual data, but we'll simulate it here by using the plant-level data
    
    # First, extract wind and solar capacity factors for each county
    # Group plants by county and calculate average capacity factors
    plant_map_df = merged_df[['State', 'Latitude', 'Longitude', 'Technology', 
                              'Old Emissions (tonnes CO2)', f'MACC - Wind ($/tonne CO2)',
                              f'MACC - Solar ($/tonne CO2)', f'MACC - {tech_column} ($/tonne CO2)',
                              'wind_cf', 'pvout']].copy()
    
    # Filter for continental US coordinates only
    plant_map_df = plant_map_df[
        (plant_map_df['Longitude'] >= -125) & 
        (plant_map_df['Longitude'] <= -65) & 
        (plant_map_df['Latitude'] >= 25) & 
        (plant_map_df['Latitude'] <= 50)
    ]
    
    # Convert plant locations to GeoDataFrame
    plants_gdf = gpd.GeoDataFrame(
        plant_map_df, 
        geometry=gpd.points_from_xy(plant_map_df.Longitude, plant_map_df.Latitude),
        crs="EPSG:4326"
    )
    
    # Spatial join to get county FIPS for each plant
    plants_with_county = gpd.sjoin(plants_gdf, continental_counties, how="left", predicate="within")
    
    # Group by county and calculate average capacity factors
    county_cf = plants_with_county.groupby('GEOID').agg({
        'wind_cf': 'mean',
        'pvout': 'mean'
    }).reset_index()
    
    # Merge back to counties
    county_renewable_potential = continental_counties.merge(county_cf, left_on='GEOID', right_on='GEOID', how='left')
    
    # Fill NaN values with regional averages
    region_cf = plants_with_county.groupby('STATEFP').agg({
        'wind_cf': 'mean',
        'pvout': 'mean'
    }).reset_index()
    
    # For counties with no plants, use state average
    for idx, county in county_renewable_potential.iterrows():
        if pd.isna(county['wind_cf']) or pd.isna(county['pvout']):
            state_fips = county['STATEFP']
            state_avg = region_cf[region_cf['STATEFP'] == state_fips]
            if not state_avg.empty:
                if pd.isna(county['wind_cf']):
                    county_renewable_potential.at[idx, 'wind_cf'] = state_avg['wind_cf'].values[0]
                if pd.isna(county['pvout']):
                    county_renewable_potential.at[idx, 'pvout'] = state_avg['pvout'].values[0]
    
    # Fill any remaining NaNs with national average
    national_avg_wind = plants_with_county['wind_cf'].mean()
    national_avg_solar = plants_with_county['pvout'].mean()
    county_renewable_potential['wind_cf'].fillna(national_avg_wind, inplace=True)
    county_renewable_potential['pvout'].fillna(national_avg_solar, inplace=True)
    
    # Normalize capacity factors for coloring
    wind_min, wind_max = county_renewable_potential['wind_cf'].quantile([0.05, 0.95])
    solar_min, solar_max = county_renewable_potential['pvout'].quantile([0.05, 0.95])
    
    county_renewable_potential['wind_cf_norm'] = (county_renewable_potential['wind_cf'] - wind_min) / (wind_max - wind_min)
    county_renewable_potential['wind_cf_norm'] = county_renewable_potential['wind_cf_norm'].clip(0, 1)
    
    county_renewable_potential['pvout_norm'] = (county_renewable_potential['pvout'] - solar_min) / (solar_max - solar_min)
    county_renewable_potential['pvout_norm'] = county_renewable_potential['pvout_norm'].clip(0, 1)
    
    # Create a 3x3 matrix category for each county based on wind and solar potential
    def get_resource_category(row):
        wind_cat = 0 if row['wind_cf_norm'] < 0.33 else (1 if row['wind_cf_norm'] < 0.66 else 2)
        solar_cat = 0 if row['pvout_norm'] < 0.33 else (1 if row['pvout_norm'] < 0.66 else 2)
        return wind_cat * 3 + solar_cat
    
    county_renewable_potential['resource_category'] = county_renewable_potential.apply(get_resource_category, axis=1)
    
    # Create a custom colormap for the 3x3 matrix
    # Low wind, low solar -> light color
    # High wind, high solar -> dark color
    resource_cmap = LinearSegmentedColormap.from_list(
        'resource_cmap', 
        ['#FFFFFF', '#F5F5DC', '#D3D3A4', 
         '#B0E0E6', '#ADD8E6', '#87CEEB', 
         '#6495ED', '#4682B4', '#000080']
    )
    
    # Step 3: Map the best technology option and derive transition names for plants
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
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Set up layout with GridSpec for flexibility
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    
    # Set the map extent to continental US
    ax.set_xlim([-125, -65])
    ax.set_ylim([25, 50])
    
    # Remove box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Plot county renewable potential as background
    county_renewable_potential.plot(
        ax=ax, 
        column='resource_category',
        cmap=resource_cmap,
        edgecolor='white',
        linewidth=0.2,
        alpha=0.8
    )
    
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
    
    # Create main transitions legend
    legend1 = fig.legend(handles, labels,
                        title="Technology Transition", 
                        fontsize=12,
                        title_fontsize=12,
                        bbox_to_anchor=(0.9, 0.5),
                        loc='center right')
    
    # Create size legend
    legend2 = fig.legend(handles=size_legend_handles,
                       title="Annual Emissions",
                       fontsize=12,
                       title_fontsize=12,
                       bbox_to_anchor=(0.9, 0.3),
                       loc='center right')
    
    # Create resource potential matrix legend
    resource_labels = [
        'Low Wind, Low Solar', 'Low Wind, Med Solar', 'Low Wind, High Solar',
        'Med Wind, Low Solar', 'Med Wind, Med Solar', 'Med Wind, High Solar',
        'High Wind, Low Solar', 'High Wind, Med Solar', 'High Wind, High Solar'
    ]
    
    # Matrix legend (3x3)
    matrix_legend_handles = []
    for i in range(9):
        color = resource_cmap(i/8)  # Normalize to 0-1 range
        patch = mpatches.Patch(color=color, label=resource_labels[i])
        matrix_legend_handles.append(patch)
    
    # Create the matrix legend
    matrix_legend = fig.legend(
        handles=matrix_legend_handles,
        title="County Renewable Potential",
        fontsize=10,
        title_fontsize=12,
        bbox_to_anchor=(0.9, 0.7),
        loc='center right'
    )
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title
    if ccs_scenario:
        title = 'Geographic Distribution of Low-Carbon Power Transitions with CCS'
        subtitle = 'Cost-optimal replacements with wind, solar, and natural gas with carbon capture'
    else:
        title = 'Geographic Distribution of Power Plant Transitions'
        subtitle = 'Cost-optimal replacements with wind, solar, and natural gas'
    
    plt.suptitle(title, fontsize=16, y=0.95)
    plt.figtext(0.5, 0.92, subtitle, ha='center', fontsize=14)
    
    # Add additional context
    plt.figtext(0.5, 0.05, 
                'Background color indicates county-level renewable energy potential (3×3 matrix of wind and solar resources)',
                ha='center', fontsize=12, style='italic')
    
    # Adjust layout to make room for legends
    plt.subplots_adjust(right=0.8)
    
    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced map visualization saved to {output_path}")
    
    return fig

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Marginal Abatement Cost Curve Analysis for Power Plants")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--scenario", type=str, default="default", help="Scenario name")
    parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis")
    
    args = parser.parse_args()
    
    if args.sensitivity:
        # Run sensitivity analysis on key parameters
        base_config = load_config(args.config)
        sensitivity_params = {
            'discount_rate': [0.03, 0.06, 0.09],
            'gas_price': [2.5, 3.5, 4.5]
        }
        run_sensitivity_analysis(base_config, sensitivity_params)
    else:
        # Run single scenario
        run_scenario(args.config, args.scenario)