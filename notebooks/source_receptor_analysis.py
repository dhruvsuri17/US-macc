import time
import numpy as np
import pandas as pd
import geopandas as gpd
import s3fs
import zarr
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

def rect(i, w, s, e, n):
    """
    Create rectangle coordinates for a grid cell
    
    Args:
        i (int): Index of the grid cell
        w (array): West coordinates
        s (array): South coordinates
        e (array): East coordinates
        n (array): North coordinates
    
    Returns:
        tuple: x and y coordinates of the rectangle
    """
    x = [w[i], e[i], e[i], w[i], w[i]]
    y = [s[i], s[i], n[i], n[i], s[i]]
    return x, y

def poly(sr):
    """
    Create polygon geometries for source-receptor grid
    
    Args:
        sr (zarr.hierarchy.Group): Source-receptor matrix Zarr group
    
    Returns:
        list: List of Shapely Polygon objects
    """
    ret = []
    w, s, e, n = sr["W"][:], sr["S"][:], sr["E"][:], sr["N"][:]
    for i in range(len(w)):
        x, y = rect(i, w, s, e, n)
        ret.append(Polygon([[x[0], y[0]], [x[1], y[1]], [x[2], y[2]],
                            [x[3], y[3]], [x[4], y[4]]]))
    return ret

def run_sr(emis, model="isrm", emis_units="tons/year"):
    """
    Run source-receptor modeling
    
    Args:
        emis (GeoDataFrame): Emissions data
        model (str, optional): Source-receptor model to use
        emis_units (str, optional): Units of emissions
    
    Returns:
        GeoDataFrame: Source-receptor modeling results
    """
    start = time.time()
    
    # S3 access to source-receptor matrix
    url = 's3://inmap-model/isrm_v1.2.1.zarr/'
    fs = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name='us-east-2'))
    
    # Alternative Zarr opening method
    try:
        # Try using store directly
        store = fs.get_mapper(url)
        sr = zarr.open_group(store, mode='r')
    except Exception as e:
        print(f"Error opening Zarr store: {e}")
        
        # Fallback method
        try:
            sr = zarr.open_group(url, mode='r', storage_options={'anon': True, 'client_kwargs': {'region_name': 'us-east-2'}})
        except Exception as e:
            print(f"Fallback Zarr opening failed: {e}")
            raise
    
    # Verify store contents
    print("Available keys in Zarr store:", list(sr.keys()))
    
    # Build geometry
    p = poly(sr)
    print("Making polygons as geometry.")
    
    # Create initial grid GeoDataFrame
    df = pd.DataFrame({'Location': range(52411)})
    gdf = gpd.GeoDataFrame(df, geometry=p)
    
    # Reproject geometries for spatial joining
    emis.crs = "+proj=longlat"
    gdf.crs = "+proj=lcc +lat_1=33.000000 +lat_2=45.000000 +lat_0=40.000000 +lon_0=-97.000000 +x_0=0 +y_0=0 +a=6370997.000000 +b=6370997.000000 +to_meter=1"
    emis = emis.to_crs(gdf.crs)
    
    # Spatial join
    join_right_df = gdf.sjoin(emis, how="right")
    print("Finished joining the dataframes.")
    
    # Prepare indices for matrix selection
    index = join_right_df.Location.tolist()
    ppl = np.unique(join_right_df.Location.tolist())
    num = range(0, len(ppl))
    dictionary = dict(zip(ppl, num))
    
    # Allocate SR matrix data
    SOA = sr['SOA'].get_orthogonal_selection(([0], ppl, slice(None)))
    print("SOA data is allocated.")
    pNO3 = sr['pNO3'].get_orthogonal_selection(([0], ppl, slice(None)))
    print("pNO3 data is allocated.")
    pNH4 = sr['pNH4'].get_orthogonal_selection(([0], ppl, slice(None)))
    print("pNH4 data is allocated.")
    pSO4 = sr['pSO4'].get_orthogonal_selection(([0], ppl, slice(None)))
    print("pSO4 data is allocated.")
    PM25 = sr['PrimaryPM25'].get_orthogonal_selection(([0], ppl, slice(None)))
    print("PrimaryPM25 data is allocated.")
    
    # Calculate pollutant contributions
    SOA_data, pNO3_data, pNH4_data, pSO4_data, PM25_data = 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(len(index)):
        SOA_data += SOA[0, dictionary[index[i]], :] * emis.VOC[i]
        pNO3_data += pNO3[0, dictionary[index[i]], :] * emis.NOx[i]
        pNH4_data += pNH4[0, dictionary[index[i]], :] * emis.NH3[i]
        pSO4_data += pSO4[0, dictionary[index[i]], :] * emis.SOx[i]
        PM25_data += PM25[0, dictionary[index[i]], :] * emis.PM2_5[i]
    
    data = SOA_data + pNO3_data + pNH4_data + pSO4_data + PM25_data
    print("Accessing the data.")
    
    # Apply unit conversion and calculate health impacts
    if emis_units == "tons/year":
        fact = 28766.639
        TotalPM25 = fact * data
        TotalPop = sr['TotalPop'][0:52411]
        MortalityRate = sr['MortalityRate'][0:52411]
        
        # Calculate deaths using two methodologies
        deathsK = (np.exp(np.log(1.06)/10 * TotalPM25) - 1) * TotalPop * 1.0465819687408728 * MortalityRate / 100000 * 1.025229357798165
        deathsL = (np.exp(np.log(1.14)/10 * TotalPM25) - 1) * TotalPop * 1.0465819687408728 * MortalityRate / 100000 * 1.025229357798165
        
        # Create results GeoDataFrame
        ret = gpd.GeoDataFrame(pd.DataFrame({
            'SOA': fact * SOA_data, 
            'pNO3': fact * pNO3_data, 
            'pNH4': fact * pNH4_data, 
            'pSO4': fact * pSO4_data, 
            'PrimaryPM25': fact * PM25_data, 
            'TotalPM25': TotalPM25, 
            'deathsK': deathsK, 
            'deathsL': deathsL
        }), geometry=p[0:52411])
    
    print(f"Finished (%.0f seconds) " % (time.time()-start))
    return ret

def analyze_results(results):
    """
    Analyze source-receptor modeling results
    
    Args:
        results (GeoDataFrame): Source-receptor modeling results
    
    Returns:
        dict: Summary of health and environmental impacts
    """
    # Calculate total impacts
    summary = {
        'Total PM2.5': results['TotalPM25'].sum(),
        'Krewski Deaths': results['deathsK'].sum(),
        'LePeule Deaths': results['deathsL'].sum(),
        'SOA Contribution': results['SOA'].sum(),
        'Nitrate Contribution': results['pNO3'].sum(),
        'Ammonium Contribution': results['pNH4'].sum(),
        'Sulfate Contribution': results['pSO4'].sum(),
        'Primary PM2.5 Contribution': results['PrimaryPM25'].sum()
    }
    
    return summary

def visualize_results(results, output_file=None):
    """
    Visualize source-receptor modeling results
    
    Args:
        results (GeoDataFrame): Source-receptor modeling results
        output_file (str, optional): Path to save visualization
    """
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Source-Receptor Modeling Results', fontsize=16)
    
    # Total PM2.5 Concentration
    results.plot(column='TotalPM25', ax=axs[0, 0], 
                 legend=True, cmap='YlOrRd', 
                 legend_kwds={'label': 'Total PM2.5 Concentration'})
    axs[0, 0].set_title('Total PM2.5 Concentration')
    
    # Deaths (Krewski Method)
    results.plot(column='deathsK', ax=axs[0, 1], 
                 legend=True, cmap='Reds', 
                 legend_kwds={'label': 'Krewski Deaths'})
    axs[0, 1].set_title('Health Impacts (Krewski Method)')
    
    # Pollutant Contributions Pie Chart
    contributions = [
        results['SOA'].sum(),
        results['pNO3'].sum(),
        results['pNH4'].sum(),
        results['pSO4'].sum(),
        results['PrimaryPM25'].sum()
    ]
    labels = ['SOA', 'Nitrate', 'Ammonium', 'Sulfate', 'Primary PM2.5']
    axs[1, 0].pie(contributions, labels=labels, autopct='%1.1f%%')
    axs[1, 0].set_title('Pollutant Contributions')
    
    # Bar plot of key metrics
    metrics = [
        'Total PM2.5', 
        'Krewski Deaths', 
        'LePeule Deaths'
    ]
    values = [
        results['TotalPM25'].sum(),
        results['deathsK'].sum(),
        results['deathsL'].sum()
    ]
    axs[1, 1].bar(metrics, values)
    axs[1, 1].set_title('Key Metrics')
    axs[1, 1].set_ylabel('Value')
    
    # Adjust layout and save/show
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()

def main(egu_gdf):
    """
    Main function to run source-receptor analysis
    
    Args:
        egu_gdf (GeoDataFrame): Emissions data for power plants
    
    Returns:
        dict: Comprehensive analysis results
    """
    # Run source-receptor modeling
    results = run_sr(egu_gdf)
    
    # Analyze results
    summary = analyze_results(results)
    
    # Visualize results
    visualize_results(results, output_file='sr_results_visualization.png')
    
    # Print summary
    print("\nSource-Receptor Modeling Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return {
        'results': results,
        'summary': summary
    }