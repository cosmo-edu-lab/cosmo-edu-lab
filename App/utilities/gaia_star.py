import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import cKDTree
from functools import lru_cache
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os

STAR_GAIA_PATH = r"C:\\Users\\elyon\\OneDrive\\Desktop\\Cosmo-Edu_Lab\\data\\GaiaSource.csv"
MIST_PATH = r"C:\\Users\\elyon\\OneDrive\\Desktop\\Cosmo-Edu_Lab\\iso_fe0.01"
OUTPUT_GIF_PATH = r"C:\\Users\\elyon\\OneDrive\\Desktop\\Cosmo-Edu_Lab\\images\\hr_diagram_evolution.gif"
TEMP_FOLDER = "temp_frames_hr"

def assign_phase(df):
    df = df.sort_values(['star_mass','log_Teff'], ascending=[True, False])
    df['phase'] = 'MS' 

    # White Dwarfs
 
    df.loc[
    (df['log_Teff'] > 4.25) &   #
    (df['log_L'] < 0.0),       
    'phase'
] = 'WD'


    # RGB
    df.loc[(df['log_L'] >= 1.0) & (df['log_Teff'] < 3.9), 'phase'] = 'RGB'

    # Helium Burning (HeB)
    df.loc[(df['log_L'] >= 1.0) & (df['log_L'] <= 3.5) & (df['log_Teff'] >= 3.7) & (df['log_Teff'] <= 4.1), 'phase'] = 'HeB'

    # AGB
    df.loc[(df['log_L'] > 3.0) & (df['log_Teff'] < 3.8), 'phase'] = 'AGB'

    # Subgiant Branch
    df.loc[(df['log_L'] > 0.2) & (df['log_L'] < 1.0) & (df['log_Teff'] < 3.9), 'phase'] = 'SGB'

    return df
def refine_phase_with_gaia(df):
    
    refined = df['stellar_phase'].copy()
    
    bp_rp = df['bp_rp']
    abs_g = df['abs_mag']

    
    is_wd = (abs_g > 10) & (abs_g > 2.5 * bp_rp + 8.5)
    refined[is_wd] = 'WD'

   
    is_sgb_region = (abs_g >= 3.0) & (abs_g <= 4.5) & (bp_rp > 0.6) & (bp_rp < 1.4)
    
    mask_force_sgb = is_sgb_region & refined.isin(['MS', 'SGB'])
    refined[mask_force_sgb] = 'SGB' 

   
    is_bright_red = (abs_g < 3.0) & (bp_rp > 0.7)
    
    mask_fix_giants = is_bright_red & (refined.isin(['MS', 'HeB', 'AGB']))
    refined[mask_fix_giants] = 'RGB' 


    is_faint = (abs_g > 5.0) & (~is_wd)
    
 
    mask_fix_ms = is_faint & (refined.isin(['RGB', 'AGB', 'HeB', 'SGB']))
    refined[mask_fix_ms] = 'MS'

    return refined

@lru_cache(maxsize=1)
def load_isochrones_cached(mist_iso_folder):
    iso_list = []
    for file in os.listdir(mist_iso_folder):
        if file.endswith(".txt"):
            iso = pd.read_csv(os.path.join(mist_iso_folder, file), sep=r"\s+")
            if {'Gaia_BP_MAWb','Gaia_RP_MAW','Gaia_G_MAW'}.issubset(iso.columns):
                iso = assign_phase(iso)
                iso['color'] = iso['Gaia_BP_MAWb'] - iso['Gaia_RP_MAW']
                iso['abs_mag'] = iso['Gaia_G_MAW']
                iso = iso[np.isfinite(iso['color']) & np.isfinite(iso['abs_mag'])]
                iso_list.append(iso[['color','abs_mag','phase','star_mass','log_L','log_Teff']])
    iso_df = pd.concat(iso_list, ignore_index=True)
    tree = cKDTree(iso_df[['color','abs_mag']].values)
    return iso_df, tree

ISO_DF, ISO_TREE = load_isochrones_cached(MIST_PATH)



@lru_cache(maxsize=1)
def load_gaia_cached(csv_path):
    df = pd.read_csv(csv_path, dtype={'source_id': str})
    df = df.dropna(subset=['bp_rp', 'phot_g_mean_mag', 'parallax', 'ra', 'dec'])
    df = df[df['parallax'] > 0]
    df['abs_mag'] = df['phot_g_mean_mag'] + 5*np.log10(df['parallax']/1000) + 5
    M_sun_G = 4.67  
    df['log_L'] = 0.4*(M_sun_G - df['abs_mag'])
    df['log_Teff'] = 3.7 - 0.25*(df['bp_rp'] - 0.5)  
    return df

GAIA_DF = load_gaia_cached(STAR_GAIA_PATH)


def plot_hr_diagram(sample_size=20000):
    df = GAIA_DF.copy()
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=1).reset_index(drop=True)

    distances, idxs = ISO_TREE.query(df[['bp_rp','abs_mag']].values)
    df['stellar_phase'] = ISO_DF['phase'].values[idxs]
    df['mass_est'] = ISO_DF['star_mass'].values[idxs]
   
   
    df['stellar_phase'] = refine_phase_with_gaia(df)
  

    phase_colors = {
        'MS':'blue', 'SGB':'green', 'RGB':'red', 
        'HeB':'orange', 'AGB':'magenta', 'WD':'grey', 'CHeB':'purple'
    }
    df['stellar_phase_color'] = df['stellar_phase'].map(phase_colors).fillna('blue')
    counts = df['stellar_phase'].value_counts()
    total_stars = len(df)

   
    label_map = {}      
    final_colors = {}   

    for phase in counts.index:
        count = counts[phase]
        perc = (count / total_stars) * 100
        
       
        new_label = f"{phase} ({perc:.1f}%)"
        label_map[phase] = new_label
        
       
        color = phase_colors.get(phase, 'black')
        final_colors[new_label] = color

   
    df['stellar_phase_label'] = df['stellar_phase'].map(label_map)


    phase_order = [label_map[p] for p in counts.index]
    fig = px.scatter(
        df,
        x='bp_rp',
        y='abs_mag',
        color='stellar_phase_label',
        hover_data=['source_id', 'bp_rp', 'abs_mag','mass_est'],
        title=f"Gaia H–R Diagram (N={total_stars})",
        color_discrete_map=final_colors,
        category_orders={'stellar_phase_label': phase_order},
        labels={'bp_rp':'BP−RP Color', 'abs_mag':'Absolute G Magnitude'},
        opacity=0.6
    )
    fig.update_yaxes(autorange='reversed')

   
    fig.add_trace(
        go.Scatter(
            x=df['bp_rp'],
            y=df['mass_est'],
            mode='markers',
            marker=dict(opacity=0),
            showlegend=False,
            hoverinfo='skip',
            yaxis='y2'
        )
    )
    title=f"Gaia H–R Diagram (N={total_stars})"
    fig.update_layout(
        title={
                    'text': f"<b>{title}</b>",   
                    'y': 0.95,                  
                    'x': 0.5,                    
                    'xanchor': 'center',      
                    'yanchor': 'top',
                    'font': {
                        'size': 24,              
                        'color': 'black'         
                    }
                },
        yaxis2=dict(
            title="Mass (M☉)",
            overlaying='y',
            side='right',
            type='log',
            exponentformat='power'
        ),
        margin=dict(l=60, r=80, t=60, b=60)
    )

    #fig.show()
    fig.write_image(r"C:\\Users\\elyon\\OneDrive\\Desktop\\Cosmo-Edu_Lab\\images\\hr_diagram.png", width=800, height=500)


#plot_hr_diagram()
df=GAIA_DF

if len(df) > 20000:
    df = df.sample(20000, random_state=42).reset_index(drop=True)


print("Assegnazione fasi stellari...")
if ISO_TREE:
    distances, idxs = ISO_TREE.query(df[['bp_rp','abs_mag']].values)
    df['stellar_phase'] = ISO_DF['phase'].values[idxs]
    df['mass_est'] = ISO_DF['star_mass'].values[idxs]
    df['stellar_phase'] = refine_phase_with_gaia(df)
else:
    print("Errore: Impossibile caricare isocrone. Assegno fase fittizia.")
    df['stellar_phase'] = 'MS'
    df['mass_est'] = 1.0


phase_priority = {'MS': 1, 'SGB': 2, 'RGB': 3, 'HeB': 4, 'AGB': 5, 'WD': 6}
df['phase_rank'] = df['stellar_phase'].map(phase_priority).fillna(99)


df = df.sort_values(['phase_rank', 'mass_est'], ascending=[True, False]).reset_index(drop=True)


os.makedirs(TEMP_FOLDER, exist_ok=True)
n_frames = 80
chunk_size = len(df) // n_frames
filenames = []

phase_colors = {
    'MS': 'blue', 'SGB': 'green', 'RGB': 'red', 
    'HeB': 'orange', 'AGB': 'magenta', 'WD': 'grey', 'CHeB': 'purple'
}


legend_order = ['MS', 'SGB', 'RGB', 'HeB', 'AGB', 'WD']


xlims = (df['bp_rp'].min() - 0.5, df['bp_rp'].max() + 0.5)
ylims = (df['abs_mag'].max() + 1, df['abs_mag'].min() - 1)
mass_min = df['mass_est'].min()
mass_max = df['mass_est'].max()
print(f"Generazione {n_frames} frames...")

for i in range(n_frames + 1):
    current_idx = min((i + 1) * chunk_size, len(df))
    subset = df.iloc[:current_idx]
    total_current = len(subset)
    if total_current > 0:
        avg_mass = subset['mass_est'].mean()
    else:
        avg_mass = 0.0
    counts = subset['stellar_phase'].value_counts() if total_current > 0 else {}

    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    
    for phase in legend_order:
        if phase not in counts: continue
            
        mask = subset['stellar_phase'] == phase
        
      
        count = counts.get(phase, 0)
        pct = (count / total_current * 100) if total_current > 0 else 0
        label_str = f"{phase} ({pct:.1f}%)"
        
        ax.scatter(
            subset.loc[mask, 'bp_rp'], 
            subset.loc[mask, 'abs_mag'], 
            c=phase_colors.get(phase, 'black'), 
            s=20, 
            alpha=0.7, 
            label=label_str,
            edgecolors='none'
        )

    ax.set_xlim(xlims)
    ax.set_ylim(ylims) 
    ax.set_title("Gaia H-R Diagram Stars Evolution ", fontsize=16, fontweight='bold')
    ax.set_xlabel("BP - RP Color", fontsize=12)
    ax.set_ylabel("Absolute Magnitude (G)", fontsize=12)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax2 = ax.twinx()  
    

    ax2.scatter(df['bp_rp'], df['mass_est'], alpha=0.0) 
    
    ax2.set_ylabel("Mass ($M_{\\odot}$)", fontsize=12, color='black')
    ax2.set_yscale('log') 

    ax2.set_ylim(mass_min, mass_max) 
    
  
    info_str = (
        f"Stars Observed: {total_current}\n"
        f"Avg Mass: {avg_mass:.2f} $M_\\odot$"
    )
    ax.text(0.02, 0.98, info_str, transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
   

  
    ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.98), fontsize=10, title="Phase (Live %)", frameon=True, framealpha=0.9, edgecolor='gray')
    plt.tight_layout()

    filename = f"{TEMP_FOLDER}/frame_{i:03d}.png"
    plt.savefig(filename, facecolor='white', edgecolor='none')
    plt.close()
    filenames.append(filename)
    
    if i % 10 == 0:
        print(f"Frame {i}/{n_frames} generato.")


print("Salvataggio GIF...")
with imageio.get_writer(OUTPUT_GIF_PATH, mode='I', duration=0.8, loop=1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

for filename in filenames:
    os.remove(filename)
os.rmdir(TEMP_FOLDER)

print(f"Fatto! GIF salvata in: {OUTPUT_GIF_PATH}")
