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
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point

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

    def run_sr_local(self, 
                    emis, 
                    output_variables=None, 
                    model_path=None,
                    verbose=True):
        """
        Run source-receptor prediction locally
        """
        # Validate model
        if self.model not in self.model_paths:
            models = ', '.join(self.model_paths.keys())
            raise ValueError(f'Model must be one of {models}, but is `{self.model}`')
        
        # Use the local model path if provided, otherwise use default
        model_path = model_path or self.model_paths.get(self.model)
        
        if not model_path:
            raise ValueError(f"No model path found for {self.model}")
        
        # Verify model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Source-receptor matrix file not found: {model_path}")
        
        start = time.time()
        job_name = f"run_aqm_{start}"
        
        # Prepare emissions file
        emis_file = os.path.join(self._tmpdir.name, f"{job_name}.shp")
        emis.to_file(emis_file)
        
        # Prepare output variables
        output_vars = output_variables or [
            "TotalPM25", "deathsK", "deathsL", 
            "SOA", "pNO3", "pNH4", "pSO4", "PrimaryPM25"
        ]
        
        try:
            # Ensure executable is available
            if self._inmap_exe is None:
                self._inmap_exe = self._download_executable()
            
            # Verify executable exists and is executable
            if not os.path.exists(self._inmap_exe):
                raise FileNotFoundError(f"InMAP executable not found: {self._inmap_exe}")
            
            if not os.access(self._inmap_exe, os.X_OK):
                raise PermissionError(f"InMAP executable is not executable: {self._inmap_exe}")
            
            # Detailed logging of parameters
            print("InMAP Executable:", self._inmap_exe)
            print("Emissions File:", emis_file)
            print("Model Path:", model_path)
            print("Emission Units:", self.emis_units)
            print("Output Variables:", json.dumps(output_vars))
            
            # Local run command with more detailed error handling
            try:
                output = subprocess.run([
                    self._inmap_exe, "srpredict",
                    f"--EmissionUnits={self.emis_units}",
                    f"--EmissionsShapefiles={emis_file}",
                    f"--OutputVariables={json.dumps(output_vars)}",
                    f"--SR.OutputFile={model_path}"
                ], 
                capture_output=True, 
                text=True, 
                check=True)
                
                # Print command output for debugging
                print("Command Output (stdout):", output.stdout)
                print("Command Output (stderr):", output.stderr)
            
            except subprocess.CalledProcessError as e:
                print("Subprocess Error:")
                print("Return Code:", e.returncode)
                print("Command:", e.cmd)
                print("Standard Output:", e.stdout)
                print("Standard Error:", e.stderr)
                raise
            
            # Check if results file exists
            results_file = f"{job_name}/OutputFile.shp"
            if not os.path.exists(results_file):
                raise FileNotFoundError(f"Results file not generated: {results_file}")
            
            # Read and return results
            results = gpd.read_file(results_file)
            
            if verbose:
                print(f"Finished local SR modeling in {time.time() - start:.2f} seconds")
            
            return results
        
        except Exception as e:
            print(f"Comprehensive error in local SR run: {e}")
            import traceback
            traceback.print_exc()
            raise

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
    results = sr_analysis.run_sr_local(egu_gdf)
    
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