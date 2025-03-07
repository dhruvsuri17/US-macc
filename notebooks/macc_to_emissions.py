import pandas as pd
import geopandas as gpd
import numpy as np
import os
from pathlib import Path

def process_macc_results_for_emissions(macc_data, merged_df, output_dir):
    """
    Process MACC results to create a modified emissions dataset for air pollution modeling
    
    Args:
        macc_data: DataFrame with MACC curve data
        merged_df: DataFrame with original emissions and plant data 
        output_dir: Directory to save outputs
        
    Returns:
        GeoDataFrame with modified emissions data for air pollution modeling
    """
    print("Processing MACC results for air pollution modeling...")
    
    # Extract plants that would be converted to NGCC
    ngcc_transitions = macc_data[macc_data['to_tech'] == 'NGCC']
    
    if ngcc_transitions.empty:
        print("No plants identified for NGCC conversion in the MACC results.")
        return None
    
    print(f"Found {len(ngcc_transitions)} plants targeted for NGCC conversion")
    
    # Create a mapping of facility IDs to their transitions
    facility_transitions = ngcc_transitions[['facility_id', 'transition', 'from_fuel', 
                                             'macc', 'annual_generation']].set_index('facility_id')
    
    # Extract relevant information from merged_df for these facilities
    ngcc_plants = merged_df[merged_df['Facility ID'].isin(ngcc_transitions['facility_id'])]
    
    # Define NGCC emission factors (tonnes per MWh)
    ngcc_emission_factors = {
        'NOx': 0.00011,  # tonnes/MWh
        'SOx': 0.00001,  # tonnes/MWh
        'PM2_5': 0.00001,  # tonnes/MWh
        'VOC': 0.00002,  # tonnes/MWh
        'NH3': 0.00001,  # tonnes/MWh
        'CO2': 0.37      # tonnes/MWh
    }
    
    # Create a new GeoDataFrame with modified emissions
    modified_data = []
    
    for _, plant in ngcc_plants.iterrows():
        facility_id = plant['Facility ID']
        
        # Skip if we don't have location data
        if pd.isna(plant['Latitude']) or pd.isna(plant['Longitude']):
            continue
            
        # Get the annual generation for this facility
        if facility_id in facility_transitions.index:
            transition = facility_transitions.loc[facility_id]
            annual_generation = transition['annual_generation']
            
            # Calculate new emissions based on NGCC emission factors
            emissions = {
                'NOx': annual_generation * ngcc_emission_factors['NOx'],
                'SOx': annual_generation * ngcc_emission_factors['SOx'],
                'PM2_5': annual_generation * ngcc_emission_factors['PM2_5'],
                'VOC': annual_generation * ngcc_emission_factors['VOC'],
                'NH3': annual_generation * ngcc_emission_factors['NH3'],
                'CO2': annual_generation * ngcc_emission_factors['CO2']
            }
            
            # Add to the modified data
            modified_data.append({
                'facility_id': facility_id,
                'site_name': plant.get('site name', f"Plant {facility_id}"),
                'state': plant.get('State', ''),
                'latitude': plant['Latitude'],
                'longitude': plant['Longitude'],
                'annual_generation_mwh': annual_generation,
                'original_fuel': transition['from_fuel'],
                'new_technology': 'NGCC',
                'abatement_cost': transition['macc'],
                'NOx': emissions['NOx'],
                'SOx': emissions['SOx'],
                'PM2_5': emissions['PM2_5'],
                'VOC': emissions['VOC'],
                'NH3': emissions['NH3'],
                'CO2': emissions['CO2'],
                'transition': transition['transition']
            })
    
    # Create DataFrame
    modified_df = pd.DataFrame(modified_data)
    
    if modified_df.empty:
        print("No valid plants with location data found for NGCC conversion.")
        return None
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        modified_df, 
        geometry=gpd.points_from_xy(modified_df['longitude'], modified_df['latitude']),
        crs='epsg:4269'  # NAD83
    )
    
    # Add stack parameters (typical for NGCC plants)
    gdf['height'] = 50.0      # meters
    gdf['diam'] = 5.0         # meters
    gdf['temp'] = 400.0       # Kelvin
    gdf['velocity'] = 20.0    # m/s
    
    # Save the modified emissions data
    output_path = os.path.join(output_dir, 'ngcc_converted_plants_emissions.gpkg')
    gdf.to_file(output_path, driver="GPKG")
    print(f"Modified emissions data saved to {output_path}")
    
    # Create a CSV version as well
    csv_path = os.path.join(output_dir, 'ngcc_converted_plants_emissions.csv')
    gdf.drop(columns=['geometry']).to_csv(csv_path, index=False)
    print(f"CSV version saved to {csv_path}")
    
    # Create a summary by state
    state_summary = gdf.groupby('state').agg({
        'facility_id': 'count',
        'annual_generation_mwh': 'sum',
        'NOx': 'sum',
        'SOx': 'sum',
        'PM2_5': 'sum',
        'CO2': 'sum'
    }).reset_index()
    
    state_summary.rename(columns={'facility_id': 'plant_count'}, inplace=True)
    state_summary_path = os.path.join(output_dir, 'ngcc_conversion_by_state.csv')
    state_summary.to_csv(state_summary_path, index=False)
    print(f"State summary saved to {state_summary_path}")
    
    return gdf

def prepare_inmap_input(gdf, output_dir):
    """
    Prepare the input file for InMAP air pollution modeling
    
    Args:
        gdf: GeoDataFrame with modified emissions
        output_dir: Directory to save outputs
        
    Returns:
        Path to the InMAP input file
    """
    # Create a copy of the data with only the required columns for InMAP
    inmap_columns = ["NOx", "SOx", "PM2_5", "NH3", "VOC", "height", "diam", "temp", "velocity", "geometry"]
    inmap_gdf = gdf[inmap_columns].copy()
    
    # Save as GeoPackage for InMAP
    inmap_path = os.path.join(output_dir, 'ngcc_converted_for_inmap.gpkg')
    inmap_gdf.to_file(inmap_path, driver="GPKG")
    print(f"InMAP input file saved to {inmap_path}")
    
    return inmap_path

def main(macc_data, merged_df, output_dir):
    """Main function to process MACC results and prepare for air pollution modeling"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process MACC results to create modified emissions dataset
    gdf = process_macc_results_for_emissions(macc_data, merged_df, output_dir)
    
    if gdf is not None:
        # Prepare InMAP input file
        inmap_path = prepare_inmap_input(gdf, output_dir)
        
        print("\nProcessing complete!")
        print(f"The InMAP input file is ready at: {inmap_path}")
        print("This file can be used with the InMAP model to assess air pollution impacts.")
    else:
        print("Processing could not be completed due to missing data.")

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Process MACC results for air pollution modeling")
    parser.add_argument("--macc_data", required=True, help="Path to MACC curve data CSV")
    parser.add_argument("--merged_df", required=True, help="Path to merged emissions data CSV")
    parser.add_argument("--output_dir", default="output/air_pollution", help="Directory for outputs")
    
    args = parser.parse_args()
    
    # Load data
    macc_data = pd.read_csv(args.macc_data)
    merged_df = pd.read_csv(args.merged_df)
    
    main(macc_data, merged_df, args.output_dir)