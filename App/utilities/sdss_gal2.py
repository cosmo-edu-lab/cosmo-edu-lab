import pandas as pd
import plotly.express as px
from astropy.coordinates import SkyCoord
import astropy.units as u
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os


SDSS_MORPHO_PATH = r"C:\\Users\\elyon\\OneDrive\\Desktop\\Cosmo-Edu_Lab\\data\\sdss_gal_morfo.txt"

def load_sdss(csv_path):
    df = pd.read_csv(csv_path, quotechar="'", skipinitialspace=True)
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
    df = pd.read_csv(path, sep=r"\s+", names=colnames)
    return df

morpho_df = load_morphology(SDSS_MORPHO_PATH)
morpho_df = morpho_df.dropna(subset=["uMAG","rMAG","R50","R90","R50_R90","z","DL","SVM"])
morpho_df["color_ur"] = morpho_df["uMAG"] - morpho_df["rMAG"]
morpho_df["concentration"] = morpho_df["R50"] / morpho_df["R90"]

def classify(row):
    if row['SVM'] == 0:
        return "Elliptical"
    else:
        if row['color_ur'] < 1.8 and row['concentration'] < 2.2:
            return "Irregular"
        elif row['color_ur'] < 2.2 and row['concentration'] < 2.6:
            return "Spiral"
        else:
            return "Lenticular"

morpho_df["morphology"] = morpho_df.apply(classify, axis=1)

counts = morpho_df["morphology"].value_counts()
total = len(morpho_df)
labels_with_pct = {m: f"{m} ({counts[m]/total*100:.1f}%)" for m in counts.index}
morpho_df["morphology_pct"] = morpho_df["morphology"].map(labels_with_pct)

color_map = {
    "Elliptical": "red",
    "Lenticular": "orange",
    "Spiral": "blue",
    "Irregular": "green"
}
color_discrete_map = {labels_with_pct[k]: v for k,v in color_map.items() if k in labels_with_pct}


fig_color_conc = px.scatter(
    morpho_df,
    x="color_ur",
    y="concentration",
    color="morphology_pct",
    color_discrete_map=color_discrete_map,
    hover_data={
        "OBJID": True,
        "morphology": True,
        "color_ur": ':.2f',
        "R50_R90": ':.2f',
        "z": True,
        "DL": ':.1f',
        "rMAG": True
    },
    title="SDSS Galaxy Morphology — Color–Concentration"
)
title3="SDSS Galaxy Morphology — Color–Concentration"
fig_color_conc.update_layout(
        title={
                    'text': f"<b>{title3}</b>",   
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
#fig_color_conc.show()
#fig_color_conc.write_image(r"C:\\Users\\elyon\\OneDrive\\Desktop\\Cosmo-Edu_Lab\\images\\col_gal.png", width=800, height=500)


fig_color_mag = px.scatter(
    morpho_df,
    x="color_ur",
    y="rMAG",
    color="morphology",
    size='z',
    size_max=15,
    color_discrete_map=color_map,
    hover_data={
        "OBJID": True,
        "morphology": True,
        "color_ur": ':.2f',
        "R50_R90": ':.2f',
        "z": ':.3f',
        "DL": ':.1f',
        "rMAG": True
    },
    title="SDSS Galaxy Morphology — Color–Magnitude"
)
title4="SDSS Galaxy Morphology — Color–Magnitude"
fig_color_mag.update_yaxes(autorange='reversed')
fig_color_mag.update_layout(
        title={
                    'text': f"<b>{title4}</b>",   
                    'y': 0.95,                  
                    'x': 0.5,                    
                    'xanchor': 'center',      
                    'yanchor': 'top',
                    'font': {
                        'size': 24,              
                        'color': 'black'         
                    }
                },
       
        margin=dict(l=60, r=80, t=80, b=60)
    )
#fig_color_mag.show()
#fig_color_mag.write_image(r"C:\\Users\\elyon\\OneDrive\\Desktop\\Cosmo-Edu_Lab\\images\\mag_gal.png", width=800, height=500)




symbol_map = {
    "Elliptical": "circle",
    "Lenticular": "square",
    "Spiral": "diamond",
    "Irregular": "x"
}

fig_z_quantified = px.scatter(
    morpho_df,
    x="color_ur",
    y="rMAG",
    color="z",                  
    symbol="morphology",        
    symbol_map=symbol_map,
    color_continuous_scale="Magma_r", 
    hover_data=["morphology", "z"],
    title="Color-Magnitude Plot: Color=Redshift, Shape=Morphology",
    opacity=0.7,               
    size_max=10              
)


fig_z_quantified.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')))

fig_z_quantified.update_yaxes(autorange='reversed')

fig_z_quantified.update_layout(
    
    title={'text': "<b>Color-Magnitude Plot: Color=Redshift, Shape=Morphology</b>",'x':0.5, 'xanchor':'center'},
    margin=dict(t=100), 
    
 
    legend=dict(
        orientation="h",       
        yanchor="bottom",
        y=1.02,                 
        xanchor="right",
        x=1,
        title_text="Morphology" 
    ),
    

    coloraxis_colorbar=dict(
        title="Redshift (z)",
        len=0.9,                
    )
)

# fig_z_quantified.show()
#fig_z_quantified.write_image(r"C:\\Users\\elyon\\OneDrive\\Desktop\\Cosmo-Edu_Lab\\images\\mag_gal_z_color.png")



cats, bin_edges = pd.qcut(morpho_df["z"], q=3, retbins=True)


labels = [
    f"Low z ({bin_edges[0]:.3f} < z ≤ {bin_edges[1]:.3f})",
    f"Mid z ({bin_edges[1]:.3f} < z ≤ {bin_edges[2]:.3f})",
    f"High z ({bin_edges[2]:.3f} < z ≤ {bin_edges[3]:.3f})"
]


morpho_df["z_category"] = pd.cut(
    morpho_df["z"], 
    bins=bin_edges, 
    labels=labels, 
    include_lowest=True 
)


morpho_df = morpho_df.sort_values("z_category")


fig_color_mag_z = px.scatter(
    morpho_df,
    x="color_ur",
    y="rMAG",
    color="morphology",
    facet_col="z_category",     
    color_discrete_map=color_map,
    hover_data=["z", "morphology"],
    title="SDSS Galaxy Morphology — Evolution by Redshift",
    opacity=0.6,
    height=550,               
    width=1300
)

fig_color_mag_z.update_yaxes(autorange='reversed')


fig_color_mag_z.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


fig_color_mag_z.update_layout(
    title={
        'text': "<b>SDSS Galaxy Morphology — Evolution by Redshift</b>",
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 24, 'color': 'black'}
    },
    margin=dict(l=60, r=80, t=100, b=60), 
    legend_title_text='Morphology'
)
# fig_color_mag_z.show()
#fig_color_mag_z.write_image(r"C:\\Users\\elyon\\OneDrive\\Desktop\\Cosmo-Edu_Lab\\images\\mag_gal_redshift.png")

morpho_df = morpho_df.sort_values('z').reset_index(drop=True)


output_folder = "temp_frames"
os.makedirs(output_folder, exist_ok=True)


color_map = {
    "Elliptical": "red",
    "Lenticular": "orange",
    "Spiral": "blue",
    "Irregular": "green"
}


n_frames = 60  
chunk_size = len(morpho_df) // n_frames

print(f"Generazione di {n_frames} frame...")

filenames = []

xlims = (morpho_df['color_ur'].min() - 0.5, morpho_df['color_ur'].max() + 0.5)
ylims = (morpho_df['concentration'].min() - 0.1, morpho_df['concentration'].max() + 0.1)

for i in range(n_frames + 1):
    current_idx = min((i + 1) * chunk_size, len(morpho_df))
    subset = morpho_df.iloc[:current_idx]
    
    current_z = subset['z'].max() if not subset.empty else 0
    total_current = len(subset)

  
    counts = subset['morphology'].value_counts() if total_current > 0 else {}

    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    for m_type, color in color_map.items():
        mask = subset['morphology'] == m_type
        
        
        count = counts.get(m_type, 0)
        pct = (count / total_current * 100) if total_current > 0 else 0
        label_str = f"{m_type} ({pct:.1f}%)" 
        
        ax.scatter(
            subset.loc[mask, 'color_ur'], 
            subset.loc[mask, 'concentration'], 
            c=color, 
            s=25, 
            alpha=0.85, 
            label=label_str, 
           
            linewidth=0.5
        )
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_title("Galaxy Evolution by Redshift", fontsize=16, fontweight='bold')
    ax.set_xlabel("Color (u - r)", fontsize=12)
    ax.set_ylabel("Concentration", fontsize=12)
    
    ax.grid(True, alpha=0.3)
    
  
    ax.legend(loc='upper right', fontsize=11, frameon=True, framealpha=0.9, edgecolor='gray')

   
    info_str = (
        f"Redshift: z = {current_z:.3f}\n"
        f"Total Galaxies: {total_current}"
    )
    ax.text(0.02, 0.97, info_str, transform=ax.transAxes, fontsize=13, fontweight='bold',
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()

    filename = f"{output_folder}/frame_{i:03d}.png"
    plt.savefig(filename)
    plt.close()
    filenames.append(filename)
    
    if i % 10 == 0:
        print(f"Frame {i}/{n_frames} generato (z={current_z:.3f})")
  


gif_path = r"C:\\Users\\elyon\\OneDrive\\Desktop\\Cosmo-Edu_Lab\\images\\galaxy_evolution.gif"
print("Creazione GIF...")

with imageio.get_writer(gif_path, mode='I', duration=0.8, loop=1) as writer: 
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

for filename in filenames:
    os.remove(filename)
os.rmdir(output_folder)

print(f"Fatto! GIF salvata in: {gif_path}")
