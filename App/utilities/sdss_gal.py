import pandas as pd
import plotly.express as px
from astropy.coordinates import SkyCoord
import astropy.units as u
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
import warnings
from astropy.coordinates.angles.formats import IllegalSecondWarning


warnings.filterwarnings('ignore', category=IllegalSecondWarning)
GAL_SDSS_PATH = r"C:\\Users\\elyon\\OneDrive\\Desktop\\Cosmo-Edu_Lab\\data\\galaxy_SDSS.csv"

OUTPUT_GIF_PATH = r"C:\\Users\\elyon\\OneDrive\\Desktop\\Cosmo-Edu_Lab\\images\\sdss_distribution_z.gif"
TEMP_FOLDER = "temp_frames_sdss"

def load_sdss(csv_path):
    df = pd.read_csv(csv_path, quotechar="'", skipinitialspace=True,low_memory=False)
    df.columns = df.columns.str.strip()
    return df

def load_morphology(path):
    colnames = [
        "OBJID","COL2","COL3",
        "umag","gmag","rmag","imag","zmag",
        "Extu","Extg","Extr","Exti","Extz",
        "R50","R90","R50_R90","z","DL",
        "uMAG","gMAG","rMAG","iMAG","zMAG",
        "u_r","g_i","r_z",
        "Kcorg","Kcori","Kcorr","Kcoru","Kcorz",
        "SVM","RF"
    ]

    df = pd.read_csv(path, sep=r"\s+", names=colnames,low_memory=False)
    return df


galaxy_df = load_sdss(GAL_SDSS_PATH)


galaxy_df = galaxy_df.dropna(subset=['ra','dec','z'])

galaxy_df['subclass'] = galaxy_df['subclass'].astype(str)


galaxy_df = galaxy_df[~galaxy_df['subclass'].str.lower().isin(['nan', 'none', ''])]
coords = SkyCoord(galaxy_df['ra'].values, galaxy_df['dec'].values, unit=(u.hourangle, u.deg), frame='icrs')
galaxy_df['ra_deg'] = coords.ra.degree
galaxy_df['dec_deg'] = coords.dec.degree
classification_column = 'subclass'


counts = galaxy_df[classification_column].value_counts()
total_galaxies = len(galaxy_df)


label_map = {}      
color_map = {}      
phase_order_labels = [] 


default_colors = px.colors.qualitative.Plotly


for i, (subclass, count) in enumerate(counts.items()):
    perc = (count / total_galaxies) * 100
    
   
    new_label = f"{subclass} ({perc:.1f}%)"
    label_map[subclass] = new_label
    
   
    color_map[new_label] = default_colors[i % len(default_colors)]
    
   
    phase_order_labels.append(new_label)


galaxy_df['subclass_label'] = galaxy_df[classification_column].map(label_map)
fig_ra_dec = px.scatter(
    galaxy_df,
    x='ra_deg',
    y='dec_deg',
    color='subclass_label', 
    hover_data=['specobj_id', 'z', 'ra_deg', 'dec_deg', 'subclass'],
    title=f'Galaxy Distribution (N={total_galaxies})',
    labels={'ra_deg': 'Right Ascension (deg)', 'dec_deg': 'Declination (deg)', 'subclass_label': 'Subclass'},
    color_discrete_map=color_map, 
    category_orders={'subclass_label': phase_order_labels} 
)
title0=f'Galaxy Distribution (N={total_galaxies})'
fig_ra_dec.update_layout(
        title={
                    'text': f"<b>{title0}</b>",   
                    'y': 0.95,                  
                    'x': 0.5,                    
                    'xanchor': 'center',      
                    'yanchor': 'top',
                    'font': {
                        'size': 24,              
                        'color': 'black'         
                    }
                },
       
        margin=dict(l=60, r=80, t=60, b=60)
    )
fig_3d = px.scatter_3d(
    galaxy_df,
    x="ra_deg",
    y="dec_deg",
    z="z",
    color="subclass_label",
    color_discrete_map=color_map,  
    hover_data=["specobj_id", "z"],
    title="3D Galaxy Distribution",
    opacity=0.6  
)


fig_3d.update_traces(marker=dict(size=2.5, line=dict(width=0))) 

title1 = "3D Galaxy Distribution"
fig_3d.update_layout(
    title={
        'text': f"<b>{title1}</b>",   
        'y': 0.95,                  
        'x': 0.5,                    
        'xanchor': 'center',      
        'yanchor': 'top',
        'font': {'size': 24, 'color': 'black'}
    },
   
    margin=dict(l=10, r=10, t=50, b=10),
    
  
    scene=dict(
        xaxis_title='RA (deg)',
        yaxis_title='Dec (deg)',
        zaxis_title='Redshift (z)',
       
        xaxis=dict(backgroundcolor="rgb(245, 245, 245)", gridcolor="white", showbackground=True),
        yaxis=dict(backgroundcolor="rgb(245, 245, 245)", gridcolor="white", showbackground=True),
        zaxis=dict(backgroundcolor="rgb(245, 245, 245)", gridcolor="white", showbackground=True),
        aspectmode='cube' 
    ),
    legend=dict(
        title="Galaxy Subclass",
        itemsizing="constant",
      
        yanchor="top", y=0.98,
        xanchor="right", x=0.98,
        bgcolor="rgba(255,255,255,0.6)"
    )
)
fig_ra_dec_z = px.scatter(
    galaxy_df,
    x="ra_deg",
    y="dec_deg",
    color="z",
    color_continuous_scale="Viridis",
    hover_data=["specobj_id", "z", "subclass"],
    title="SDSS RA–Dec - Redshift"
)
title2="SDSS RA–Dec - Redshift"
fig_ra_dec_z.update_layout(
        title={
                    'text': f"<b>{title2}</b>",   
                    'y': 0.95,                  
                    'x': 0.5,                    
                    'xanchor': 'center',      
                    'yanchor': 'top',
                    'font': {
                        'size': 24,              
                        'color': 'black'         
                    }
                },
       
        margin=dict(l=60, r=80, t=60, b=60)
    )
#fig_ra_dec_z.write_image(r"C:\\Users\\elyon\\OneDrive\\Desktop\\Cosmo-Edu_Lab\\images\\sdss_gal_z.png", width=800, height=500)

#fig_3d.write_image(r"C:\\Users\\elyon\\OneDrive\\Desktop\\Cosmo-Edu_Lab\\images\\sdss_gal_z3D.png", width=1200, height=1000)

#fig_ra_dec.show()
#fig_ra_dec.write_image(r"C:\\Users\\elyon\\OneDrive\\Desktop\\Cosmo-Edu_Lab\\images\\sdss_gal.png", width=800, height=500)

galaxy_df = galaxy_df.sort_values('z').reset_index(drop=True)

counts_for_ordering = galaxy_df['subclass'].value_counts()
subclasses_ordered = counts_for_ordering.index.tolist()


plotly_colors = px.colors.qualitative.Plotly

color_map = {}
for i, subclass in enumerate(subclasses_ordered):
    
    color_map[subclass] = plotly_colors[i % len(plotly_colors)]

print("Mappa colori assegnata (stile Plotly):", color_map)
os.makedirs(TEMP_FOLDER, exist_ok=True)
n_frames = 80  
chunk_size = len(galaxy_df) // n_frames
filenames = []


xlims = (galaxy_df['ra_deg'].min() - 5, galaxy_df['ra_deg'].max() + 5)
ylims = (galaxy_df['dec_deg'].min() - 5, galaxy_df['dec_deg'].max() + 5)

print(f"Inizio generazione {n_frames} frames...")

for i in range(n_frames + 1):
   
    current_idx = min((i + 1) * chunk_size, len(galaxy_df))
    subset = galaxy_df.iloc[:current_idx]
    
    current_z = subset['z'].max() if not subset.empty else 0
    total_current = len(subset)

  
    counts = subset['subclass'].value_counts() if total_current > 0 else {}

  
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for subclass in subclasses_ordered:
      
        if subclass not in counts:
            continue
            
        mask = subset['subclass'] == subclass
        
      
        count = counts.get(subclass, 0)
        pct = (count / total_current * 100) if total_current > 0 else 0
    
        label_str = f"{subclass} ({pct:.1f}%)"
        
        ax.scatter(
            subset.loc[mask, 'ra_deg'], 
            subset.loc[mask, 'dec_deg'], 
            c=color_map[subclass], 
            s=10,           
            alpha=0.7,      
            label=label_str, 
            edgecolors='none'
        )

 
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    #ax.invert_xaxis() 
    
    ax.set_title("SDSS Galaxy Distribution by Redshift", fontsize=16, fontweight='bold')
    ax.set_xlabel("Right Ascension (deg)", fontsize=12)
    ax.set_ylabel("Declination (deg)", fontsize=12)
    ax.grid(True, alpha=0.2, linestyle='--')

   
    info_str = (
        f"Redshift: z = {current_z:.3f}\n"
        f"Total Galaxies: {total_current}"
    )
  
    ax.text(0.02, 0.98, info_str, transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))


    ax.legend(loc='upper right', fontsize=10, title="Subclass ", frameon=True, framealpha=0.9, edgecolor='gray')

    plt.tight_layout()

    filename = f"{TEMP_FOLDER}/frame_{i:03d}.png"
    plt.savefig(filename)
    plt.close()
    filenames.append(filename)
    
    if i % 10 == 0 or i == n_frames:
        print(f"Frame {i}/{n_frames} generato (z={current_z:.3f})")


print("Creazione GIF in corso...")

with imageio.get_writer(OUTPUT_GIF_PATH, mode='I', duration=0.8, loop=1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)


for filename in filenames:
    os.remove(filename)
os.rmdir(TEMP_FOLDER)

print(f"Fatto! GIF salvata in: {OUTPUT_GIF_PATH}")