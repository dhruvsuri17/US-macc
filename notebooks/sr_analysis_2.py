# Ensure compatibility between python 2 and python 3
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import requests
import platform
import os
import stat
import tempfile
import json
import time
import subprocess
import geopandas as gpd
import pandas as pd
import shutil
import matplotlib.pyplot as plt

class SourceReceptorAnalysis:
    def __init__(self, 
                 model="isrm", 
                 emis_units="tons/year", 
                 vsl=9.0e6,  # Value of Statistical Life
                 output_dir=None):
        """
        Initialize Source-Receptor Analysis

        Args:
            model (str): Source-receptor matrix model to use
            emis_units (str): Units of emissions (default: 'tons/year')
            vsl (float): Value of Statistical Life (default: $9M)
            output_dir (str, optional): Directory to save output files
        """
        self.model = model
        self.emis_units = emis_units
        self.vsl = vsl
        
        # Model paths (can be extended or overridden)
        self.model_paths = {
            "isrm": "/data/isrmv121/isrm_v1.2.1.ncf",
            "apsca_q0": "/data/apsca/apsca_sr_Q0_v1.2.1.ncf",
            "apsca_q1": "/data/apsca/apsca_sr_Q1_v1.2.1.ncf",
            "apsca_q2": "/data/apsca/apsca_sr_Q2_v1.2.1.ncf",
            "apsca_q3": "/data/apsca/apsca_sr_Q3_v1.2.1.ncf",
            "apsca_q4": "/data/apsca/apsca_sr_Q4_v1.2.1.ncf",
        }
        
        # Temporary directory for processing
        self._tmpdir = tempfile.TemporaryDirectory()
        self._inmap_exe = None
        
        # Output directory
        self.output_dir = output_dir or os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)

    def _download_executable(self):
        """Download InMAP executable for the current system"""
        version = "1.9.0"
        arch = platform.machine()
        if arch == "x86_64": 
            arch = "amd64"
        
        ost = platform.system()
        print(f"Downloading InMAP executable for {ost}", end='\r')
        
        executable_urls = {
            "Windows": f"https://github.com/spatialmodel/inmap/releases/download/v{version}/inmap-v{version}-windows-{arch}.exe",
            "Darwin": f"https://github.com/spatialmodel/inmap/releases/download/v{version}/inmap-v{version}-darwin-{arch}",
            "Linux": f"https://github.com/spatialmodel/inmap/releases/download/v{version}/inmap-v{version}-linux-{arch}"
        }
        
        if ost not in executable_urls:
            raise OSError(f"Unsupported operating system: {ost}")
        
        exe_path = os.path.join(self._tmpdir.name, f"inmap_{version}" + (".exe" if ost == "Windows" else ""))
        
        # Download executable
        with open(exe_path, "wb") as file:
            response = requests.get(executable_urls[ost])
            if not response.ok:
                raise Exception(f"Downloading file from {executable_urls[ost]} failed")
            file.write(response.content)
        
        # Make executable
        os.chmod(exe_path, stat.S_IXUSR|stat.S_IRUSR|stat.S_IWUSR)
        
        return exe_path

    def run_sr(self, 
           emis, 
           output_variables=None, 
           memory_gb=8,  # Increased default memory 
           max_runtime_hours=24,  # Allow up to 24 hours
           verbose=True):
        """
        Run source-receptor prediction with extended runtime and enhanced logging

        Args:
            emis (GeoDataFrame): Emissions data
            output_variables (list): Variables to output
            memory_gb (int): Memory allocation for cloud job
            max_runtime_hours (int): Maximum job runtime in hours
            verbose (bool): Print detailed information

        Returns:
            GeoDataFrame: Source-receptor model results
        """
        # Validate model
        if self.model not in self.model_paths:
            models = ', '.join(self.model_paths.keys())
            raise ValueError(f'Model must be one of {models}, but is `{self.model}`')
        
        # Default output variables if not specified
        if output_variables is None:
            output_variables = [
                "TotalPM25", 
                "deathsK", 
                "deathsL", 
                "SOA", 
                "pNO3", 
                "pNH4", 
                "pSO4", 
                "PrimaryPM25"
            ]
        
        # Ensure executable is available
        if self._inmap_exe is None:
            self._inmap_exe = self._download_executable()
        
        # Start timing
        start = time.time()
        job_name = f"run_aqm_{start}"
        
        # Prepare emissions file
        emis_file = os.path.join(self._tmpdir.name, f"{job_name}.shp")
        
        # Additional logging for emissions data
        print("\nEmissions Data Summary:")
        print(f"Total number of facilities: {len(emis)}")
        print("Emission Columns:", list(emis.columns))
        
        # Check and log emissions statistics
        emission_columns = ['VOC', 'NOx', 'NH3', 'SOx', 'PM2_5']
        for col in emission_columns:
            print(f"\n{col} Emissions:")
            print(f"  Total: {emis[col].sum()}")
            print(f"  Mean: {emis[col].mean()}")
            print(f"  Max: {emis[col].max()}")
            print(f"  Zero Values: {(emis[col] == 0).sum()}")
        
        # Save emissions file
        emis.to_file(emis_file)
        print(f"\nEmissions file saved to: {emis_file}")
        
        # Prepare cloud run command with extended parameters
        cloud_cmd = [
            self._inmap_exe, "cloud", "start",
            "--cmds=srpredict",
            f"--version=v1.9.0",
            f"--job_name={job_name}",
            f"--memory_gb={memory_gb}",
            f"--EmissionUnits={self.emis_units}",
            f"--EmissionsShapefiles={emis_file}",
            f"--OutputVariables={json.dumps(output_variables)}",
            f"--SR.OutputFile={self.model_paths[self.model]}"
        ]
        
        # Add max runtime in minutes
        max_runtime_minutes = int(max_runtime_hours * 60)
        cloud_cmd.append(f"--max-runtime={max_runtime_minutes}")
        
        # Run cloud prediction
        try:
            subprocess.check_output(cloud_cmd)
        except subprocess.CalledProcessError as e:
            print(f"Initial job submission failed: {e}")
            raise
        
        # Extended job monitoring with more detailed logging
        total_wait_time = 0
        check_interval = 60  # Check every minute
        while True:
            try:
                # Get job status
                status = subprocess.check_output([
                    self._inmap_exe, "cloud", "status", 
                    f"--job_name={job_name}"
                ]).decode("utf-8").strip()
                
                # Calculate elapsed time
                elapsed_time = time.time() - start
                total_wait_time += check_interval
                
                if verbose:
                    print(f"Simulation {status} (Elapsed: {elapsed_time/60:.2f} mins, Total Wait: {total_wait_time/60:.2f} mins)")
                
                # Check job status
                if status == "Complete":
                    break
                elif status == "Failed":
                    # Attempt to retrieve logs
                    try:
                        logs = subprocess.check_output([
                            self._inmap_exe, "cloud", "logs", 
                            f"--job_name={job_name}"
                        ]).decode("utf-8")
                        print("\nJob Failure Logs:")
                        print(logs)
                    except Exception as log_err:
                        print(f"Could not retrieve job logs: {log_err}")
                    
                    raise ValueError(f"Job failed after {elapsed_time/60:.2f} minutes")
                elif status != "Running":
                    raise ValueError(f"Unexpected job status: {status}")
                
                # Check for timeout
                if elapsed_time > max_runtime_hours * 3600:
                    raise TimeoutError(f"Job exceeded maximum runtime of {max_runtime_hours} hours")
                
                # Wait before next status check
                time.sleep(check_interval)
            
            except subprocess.CalledProcessError as err:
                print(f"Status check error: {err}")
                raise
        
        # Retrieve output
        subprocess.check_call([
            self._inmap_exe, "cloud", "output", 
            f"--job_name={job_name}"
        ])
        
        # Read output
        output = gpd.read_file(f"{job_name}/OutputFile.shp")
        
        # Clean up
        try:
            shutil.rmtree(job_name)
            subprocess.check_call([
                self._inmap_exe, "cloud", "delete", 
                f"--job_name={job_name}"
            ])
        except Exception as cleanup_err:
            print(f"Warning: Job cleanup failed: {cleanup_err}")
        
        # Final timing and logging
        total_runtime = time.time() - start
        print(f"\nFinished source-receptor modeling in {total_runtime/60:.2f} minutes")
        
        return output

    def analyze_results(self, results):
        """
        Analyze source-receptor results and calculate health/economic impacts

        Args:
            results (GeoDataFrame): Source-receptor model results

        Returns:
            dict: Analysis results including deaths and economic damages
        """
        # Compute deaths
        deaths = pd.DataFrame({
            "Model": [self.model],
            "Krewski Deaths": [results["deathsK"].sum()],
            "LePeule Deaths": [results["deathsL"].sum()]
        })
        
        # Calculate economic damages
        damages = pd.DataFrame({
            "Model": [self.model],
            "Krewski Damages": deaths["Krewski Deaths"] * self.vsl,
            "LePeule Damages": deaths["LePeule Deaths"] * self.vsl
        })
        
        return {
            "deaths": deaths,
            "damages": damages
        }

    def visualize_pm25(self, results, save_path=None):
        """
        Visualize PM2.5 concentrations

        Args:
            results (GeoDataFrame): Source-receptor model results
            save_path (str, optional): Path to save the visualization
        """
        # Cut off at 98.5 percentile for better visualization
        cut = results["TotalPM25"].quantile(0.985)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        results.plot(
            column="TotalPM25", 
            cmap="GnBu", 
            legend=True, 
            vmin=0, 
            vmax=cut, 
            ax=ax,
            legend_kwds={'label': 'PM2.5 Concentration'}
        )
        plt.title(f"PM2.5 Concentrations - {self.model.upper()} Model")
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def save_results(self, results, filename=None):
        """
        Save results to a shapefile

        Args:
            results (GeoDataFrame): Source-receptor model results
            filename (str, optional): Output filename
        """
        if filename is None:
            filename = os.path.join(
                self.output_dir, 
                f"{self.model}_sr_results.shp"
            )
        
        results.to_file(filename)
        print(f"Results saved to {filename}")

# Example usage function
def run_source_receptor_analysis(egu_gdf, model="isrm"):
    """
    Convenience function to run full source-receptor analysis

    Args:
        egu_gdf (GeoDataFrame): Emissions data
        model (str, optional): Source-receptor model to use

    Returns:
        dict: Comprehensive analysis results
    """
    # Initialize analysis
    sr_analysis = SourceReceptorAnalysis(model=model)
    
    # Run source-receptor model
    results = sr_analysis.run_sr(egu_gdf)
    
    # Analyze results
    analysis = sr_analysis.analyze_results(results)
    
    # Visualize PM2.5
    sr_analysis.visualize_pm25(
        results, 
        save_path=os.path.join(sr_analysis.output_dir, f"{model}_pm25_map.png")
    )
    
    # Save results
    sr_analysis.save_results(results)
    
    # Print results
    print("\nTotal Health Impacts:")
    print(analysis['deaths'])
    print("\nEconomic Damage Estimates:")
    print(analysis['damages'])
    
    return {
        'results': results,
        'analysis': analysis
    }