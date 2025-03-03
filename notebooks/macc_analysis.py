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

def aggregate_facility_data(plant_df_filtered):
    """
    Aggregate plant data to facility level
    
    Args:
        plant_df_filtered: Filtered plant DataFrame
        
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
    
    # Map technologies
    mapping_rules = {
        "Natural Gas Fired Combustion Turbine": "Combustion turbine",
        "Conventional Steam Coal": "Coal",
        "Natural Gas Fired Combined Cycle": "Combined cycle",
        "Natural Gas Steam Turbine": "Oil/Gas steam",
        "Petroleum Liquids": "Combustion turbine - aeroderivative",
        "Natural Gas Internal Combustion Engine": "Internal combustion engine",
        "Petroleum Coke": "Combustion turbine - aeroderivative"
    }
    
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
        scenario_config = config
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
    aggregated_facility_attributes, unit_level_attributes = aggregate_facility_data(plant_df_filtered)
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