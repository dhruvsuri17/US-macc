import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from pathlib import Path

def create_figure4_comprehensive(config, baseline_scenario, ccs_scenario, health_damages_by_state=None):
    """
    Create a comprehensive four-panel figure showing power plant characteristics and cost distributions
    
    Args:
        config: Configuration dictionary
        baseline_scenario: Tuple of (merged_df, macc_data, summary) from baseline scenario
        ccs_scenario: Tuple of (merged_df, macc_data, summary) from CCS scenario
        health_damages_by_state: Optional DataFrame with health damages by state
        
    Returns:
        Matplotlib figure object
    """
    # Unpack scenario data
    baseline_merged_df, baseline_macc_data, baseline_summary = baseline_scenario
    ccs_merged_df, ccs_macc_data, ccs_summary = ccs_scenario
    
    # Create figure with 4 panels using GridSpec for more control
    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])
    
    # Panel A: Age distribution of existing EGUs
    ax1 = fig.add_subplot(gs[0, 0])
    plot_age_distribution(baseline_merged_df, ax1)
    
    # Panel B: MACC distribution for baseline (negative/zero cost)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_macc_distribution(baseline_macc_data, ax2, title="MACC Distribution for Transitions with Negative or Zero Cost", 
                          cost_range=(-1000, 0))
    
    # Panel C: MACC distribution for CCS scenario (negative/zero cost)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_macc_distribution(ccs_macc_data, ax3, title="MACC Distribution for Transitions with CCS (Negative or Zero Cost)", 
                          cost_range=(-1000, 0))
    
    # Panel D: Log-scale MACC by state with health damages
    ax4 = fig.add_subplot(gs[1, 1])
    plot_macc_by_state(baseline_macc_data, health_damages_by_state, ax4)
    
    # Add overall title
    fig.suptitle('Characteristics of Power Plants and Cost Distribution of Fuel Transitions', 
                y=0.98, fontsize=16, fontweight='bold')
    
    # Add figure caption as footer
    caption_text = (
        "Figure 4. Characteristics of existing power plants and the cost distribution of potential fuel transitions in the United States. "
        "(a) The age distribution of existing electricity generating units (EGUs) across different technologies. "
        "(b) The distribution of marginal abatement costs ($/ton CO₂) for transitions with negative or zero cost, by fuel type. "
        "(c) The distribution of marginal abatement costs ($/ton CO₂) for transitions limited to wind, solar, and natural gas with CCS. "
        "(d) A log-scale representation of marginal abatement costs across U.S. states, with health damages on secondary axis."
    )
    fig.text(0.5, 0.01, caption_text, ha='center', fontsize=10, wrap=True)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save the figure
    output_path = os.path.join(config.get('output_dir', 'output'), 'figures', 'figure4_comprehensive.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_age_distribution(merged_df, ax):
    """
    Plot the age distribution of existing electricity generating units
    
    Args:
        merged_df: DataFrame with plant data
        ax: Matplotlib axis to plot on
    """
    # Copy data to avoid modifying original
    df = merged_df.copy()
    
    # Calculate age if not already present
    if 'Age' not in df.columns and 'Operating Year' in df.columns:
        current_year = 2024  # Use config['analysis_year'] if available
        df['Age'] = current_year - df['Operating Year']
    
    # Get technology types
    if 'Technology' in df.columns:
        tech_column = 'Technology'
    elif 'Prime Mover' in df.columns:
        tech_column = 'Prime Mover'
    else:
        # Find a suitable column
        for col in df.columns:
            if 'tech' in col.lower() or 'type' in col.lower():
                tech_column = col
                break
        else:
            # If no suitable column, create a dummy one
            df['Technology'] = 'Unknown'
            tech_column = 'Technology'
    
    # Simplify technology types
    def simplify_tech(tech):
        tech = str(tech).lower()
        if 'coal' in tech:
            return 'Coal'
        elif 'gas' in tech or 'turbine' in tech or 'combined cycle' in tech:
            return 'Natural Gas'
        elif 'nuclear' in tech:
            return 'Nuclear'
        elif 'wind' in tech:
            return 'Wind'
        elif 'solar' in tech or 'pv' in tech:
            return 'Solar'
        elif 'hydro' in tech or 'water' in tech:
            return 'Hydro'
        elif 'oil' in tech or 'petroleum' in tech:
            return 'Oil'
        elif 'biomass' in tech or 'wood' in tech:
            return 'Biomass'
        else:
            return 'Other'
    
    # Apply technology simplification
    df['Technology_Simple'] = df[tech_column].apply(simplify_tech)
    
    # Create age bins
    df['Age_Bin'] = pd.cut(df['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 100], 
                          labels=['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '60+'])
    
    # Create violin plot
    sns.violinplot(x='Technology_Simple', y='Age', data=df, ax=ax, inner='quartile', 
                  palette='viridis', cut=0)
    
    # Customize plot
    ax.set_title('Age Distribution of Existing Power Plants by Technology', fontsize=12)
    ax.set_xlabel('Technology', fontsize=10)
    ax.set_ylabel('Age (years)', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add mean age as text
    for i, tech in enumerate(df['Technology_Simple'].unique()):
        mean_age = df[df['Technology_Simple'] == tech]['Age'].mean()
        if not np.isnan(mean_age):
            ax.text(i, df['Age'].max()-5, f'Mean: {mean_age:.1f}', 
                   ha='center', va='center', fontsize=8, color='red')

def plot_macc_distribution(macc_data, ax, title="MACC Distribution", cost_range=(-1000, 0)):
    """
    Plot the distribution of marginal abatement costs for transitions
    
    Args:
        macc_data: DataFrame with MACC data
        ax: Matplotlib axis to plot on
        title: Plot title
        cost_range: Tuple of (min, max) cost to include
    """
    # Copy data to avoid modifying original
    df = macc_data.copy()
    
    # Filter by cost range
    df = df[(df['macc'] >= cost_range[0]) & (df['macc'] <= cost_range[1])]
    
    if df.empty:
        ax.text(0.5, 0.5, "No data in the specified cost range", 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=12)
        return
    
    # Create cost bins - divide the range into 10 equal bins
    bin_width = (cost_range[1] - cost_range[0]) / 10
    bins = np.arange(cost_range[0], cost_range[1] + bin_width, bin_width)
    
    # Assign each transition to a bin
    df['Cost_Bin'] = pd.cut(df['macc'], bins=bins)
    
    # Group by transition type and cost bin
    if 'to_tech' in df.columns:
        transition_col = 'to_tech'
    elif 'transition' in df.columns:
        # Extract target technology from transition string
        df['target_tech'] = df['transition'].apply(lambda x: x.split(' to ')[1] if ' to ' in str(x) else x)
        transition_col = 'target_tech'
    else:
        # Create dummy column
        df['transition_type'] = 'Unknown'
        transition_col = 'transition_type'
    
    # Group data by bin and transition type
    grouped = df.groupby(['Cost_Bin', transition_col])['emissions'].sum().reset_index()
    grouped_pivot = grouped.pivot(index='Cost_Bin', columns=transition_col, values='emissions')
    
    # Fill NaN with 0
    grouped_pivot = grouped_pivot.fillna(0)
    
    # Plot stacked histogram
    grouped_pivot.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
    
    # Customize plot
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Marginal Abatement Cost ($/tonne CO₂)', fontsize=10)
    ax.set_ylabel('CO₂ Emissions Abated (tonnes)', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Format x-axis tick labels to display the range represented by each bin
    def format_bin_label(bin_obj):
        return f"{bin_obj.left:.0f} to {bin_obj.right:.0f}"
    
    ax.set_xticklabels([format_bin_label(bin_obj) for bin_obj in grouped_pivot.index])
    
    # Add legend
    ax.legend(title='Transition To', loc='upper right')

def plot_macc_by_state(macc_data, health_damages_by_state, ax):
    """
    Plot MACC by state with health damages on secondary axis
    
    Args:
        macc_data: DataFrame with MACC data
        health_damages_by_state: DataFrame with health damages by state
        ax: Matplotlib axis to plot on
    """
    # Copy data to avoid modifying original
    df = macc_data.copy()
    
    # Extract state from transition data if available
    if 'state' not in df.columns:
        # Try to find state information
        state_cols = [col for col in df.columns if 'state' in col.lower()]
        if state_cols:
            df['state'] = df[state_cols[0]]
        else:
            # If no state column, create dummy data
            df['state'] = 'Unknown'
    
    # Group by state and transition type, calculate total emissions abated
    if 'to_tech' in df.columns:
        transition_col = 'to_tech'
    elif 'transition' in df.columns:
        # Extract target technology from transition string
        df['target_tech'] = df['transition'].apply(lambda x: x.split(' to ')[1] if ' to ' in str(x) else x)
        transition_col = 'target_tech'
    else:
        # Create dummy column
        df['transition_type'] = 'Unknown'
        transition_col = 'transition_type'
    
    # Separate positive and negative MACC
    df_neg = df[df['macc'] <= 0].copy()
    df_pos = df[df['macc'] > 0].copy()
    
    # Group by state and transition type
    neg_by_state = df_neg.groupby(['state', transition_col])['emissions'].sum().reset_index()
    pos_by_state = df_pos.groupby(['state', transition_col])['emissions'].sum().reset_index()
    
    # Calculate total by state for sorting
    total_by_state = df.groupby('state')['emissions'].sum().reset_index()
    sorted_states = total_by_state.sort_values('emissions', ascending=False)['state'].tolist()
    
    # Convert to pivoted dataframes
    neg_pivot = neg_by_state.pivot(index='state', columns=transition_col, values='emissions').fillna(0)
    pos_pivot = pos_by_state.pivot(index='state', columns=transition_col, values='emissions').fillna(0)
    
    # Reorder states
    if not neg_pivot.empty:
        neg_pivot = neg_pivot.reindex(sorted_states)
    if not pos_pivot.empty:
        pos_pivot = pos_pivot.reindex(sorted_states)
    
    # Determine colors for transition types
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_map = {}
    all_techs = set()
    if not neg_pivot.empty:
        all_techs.update(neg_pivot.columns)
    if not pos_pivot.empty:
        all_techs.update(pos_pivot.columns)
    
    for i, tech in enumerate(all_techs):
        color_map[tech] = colors[i % len(colors)]
    
    # Create stacked horizontal bar chart
    bar_height = 0.35
    y_pos = np.arange(len(sorted_states))
    
    # Plot negative MACC (left side)
    left_edge = np.zeros(len(sorted_states))
    if not neg_pivot.empty:
        for tech in neg_pivot.columns:
            if tech in neg_pivot:
                # Use log scale for better visualization
                values = -np.log10(-neg_pivot[tech].fillna(0) + 1)  # Add 1 to avoid log(0)
                ax.barh(y_pos - bar_height/2, values, bar_height, left=left_edge, 
                      label=f"{tech} (Negative Cost)" if tech not in color_map else "", 
                      color=color_map.get(tech, 'gray'))
                left_edge = left_edge + values
    
    # Plot positive MACC (right side)
    right_edge = np.zeros(len(sorted_states))
    if not pos_pivot.empty:
        for tech in pos_pivot.columns:
            if tech in pos_pivot:
                # Use log scale for better visualization
                values = np.log10(pos_pivot[tech].fillna(0) + 1)  # Add 1 to avoid log(0)
                ax.barh(y_pos + bar_height/2, values, bar_height, left=right_edge, 
                       label=f"{tech} (Positive Cost)" if tech not in color_map else "", 
                       color=color_map.get(tech, 'gray'))
                right_edge = right_edge + values
    
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Technology", loc='upper right')
    
    # Add health damages on secondary y-axis if available
    if health_damages_by_state is not None:
        # Create secondary y-axis
        ax2 = ax.twinx()
        
        # Plot health damages as points
        health_data = []
        for state in sorted_states:
            if state in health_damages_by_state.index:
                health_data.append(health_damages_by_state.loc[state])
            else:
                health_data.append(0)
        
        # Plot health damages
        ax2.scatter(np.zeros(len(sorted_states)) - 10, y_pos, s=np.array(health_data)/1e6, 
                   color='red', alpha=0.6, label='Health Damages')
        
        # Add legend for health damages
        ax2.set_ylabel('Health Damages ($ millions)', fontsize=10)
        
        # Create custom legend for health damages sizes
        health_sizes = [10, 100, 1000]
        for size in health_sizes:
            ax2.scatter([], [], s=size/1e6, color='red', alpha=0.6, 
                       label=f'${size}M')
        
        # Add second legend for health damages
        ax2.legend(title="Health Damages", loc='lower right')
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_states)
    ax.set_title('Marginal Abatement Costs by State (Log Scale)', fontsize=12)
    ax.set_xlabel('Abatement Potential (Log Scale)', fontsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add a vertical line at x=0 to separate positive and negative costs
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    
    # Add annotations explaining the axes
    ax.text(-5, len(sorted_states) + 1, "← Negative Cost (Savings)", fontsize=10, ha='center')
    ax.text(5, len(sorted_states) + 1, "Positive Cost →", fontsize=10, ha='center')

def prepare_health_damages_by_state(health_damages_data):
    """
    Prepare health damages data by state for plotting
    
    Args:
        health_damages_data: GeoDataFrame or DataFrame with health damages data
        
    Returns:
        Series of health damages by state
    """
    # If health_damages_data is a GeoDataFrame, convert to DataFrame
    df = pd.DataFrame(health_damages_data.drop(columns=['geometry']) 
                     if 'geometry' in health_damages_data.columns else health_damages_data)
    
    # Find the state column
    state_col = None
    for col in df.columns:
        if 'state' in col.lower():
            state_col = col
            break
    
    if state_col is None:
        return pd.Series({})
    
    # Find the health damages column
    health_col = None
    for col in df.columns:
        if 'health' in col.lower() and 'damage' in col.lower():
            health_col = col
            break
    
    if health_col is None:
        return pd.Series({})
    
    # Aggregate health damages by state
    health_by_state = df.groupby(state_col)[health_col].sum()
    
    return health_by_state

def main(config, baseline_dir, ccs_dir):
    """
    Main function to create Figure 4
    
    Args:
        config: Configuration dictionary
        baseline_dir: Directory with baseline scenario results
        ccs_dir: Directory with CCS scenario results
    """
    # Load baseline scenario data
    baseline_merged_df = pd.read_csv(os.path.join(baseline_dir, 'macc_calculations.csv'))
    baseline_macc_data = pd.read_csv(os.path.join(baseline_dir, 'macc_curve_data.csv'))
    
    # Load CCS scenario data
    ccs_merged_df = pd.read_csv(os.path.join(ccs_dir, 'ccs_macc_calculations.csv'))
    ccs_macc_data = pd.read_csv(os.path.join(ccs_dir, 'ccs_macc_curve_data.csv'))
    
    # Load health damages by state if available
    health_damages_path = os.path.join(config['output_dir'], 'air_pollution', 'ngcc_conversion_by_state.csv')
    health_damages_by_state = None
    if os.path.exists(health_damages_path):
        health_damages_df = pd.read_csv(health_damages_path)
        health_damages_by_state = prepare_health_damages_by_state(health_damages_df)
    
    # Create figure
    baseline_summary = {}  # Placeholder
    ccs_summary = {}      # Placeholder
    fig = create_figure4_comprehensive(
        config,
        (baseline_merged_df, baseline_macc_data, baseline_summary),
        (ccs_merged_df, ccs_macc_data, ccs_summary),
        health_damages_by_state
    )
    
    return fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Figure 4 for MACC analysis")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--baseline", type=str, default="output/baseline", help="Baseline scenario directory")
    parser.add_argument("--ccs", type=str, default="output/ccs_scenario", help="CCS scenario directory")
    
    args = parser.parse_args()
    
    # Load configuration
    from macc_analysis import load_config
    config = load_config(args.config)
    
    # Create figure
    fig = main(config, args.baseline, args.ccs)
    plt.show()