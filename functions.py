import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy import array
from pandas import DataFrame, MultiIndex
from pyproj import Transformer

sns.set_style('white')


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2],16)/255 for i in (0,2,4))


def find_closest_points_to_input(data, latitude, longitude, delta_lat=0.001, delta_lon=0.001):
    df_box = data.reset_index()[
        (data.reset_index()['lat'] >= latitude - delta_lat) & (data.reset_index()['lat'] <= latitude + delta_lat) &
        (data.reset_index()['lon'] >= longitude - delta_lon) & (data.reset_index()['lon'] <= longitude + delta_lon)
    ]

    print(
        f"found {len(df_box[['lat', 'lon']].drop_duplicates())} unique lat/lon pairs:\n"
        f"{df_box[['lat', 'lon']].drop_duplicates()}"
    )
    return df_box


def convert_rd_into_geocoordinates(data):
    x_rd, y_rd, z = data.index.codes[0], data.index.codes[1], data.index.codes[2]

    x_vals = data.index.get_level_values(0).to_numpy()
    y_vals = data.index.get_level_values(1).to_numpy()
    z_vals = data.index.get_level_values(2).to_numpy()

    transformer = Transformer.from_crs("epsg:28992", "epsg:4326", always_xy=True)
    lon, lat = transformer.transform(x_vals, y_vals)

    data.index = MultiIndex.from_arrays([lon, lat, z_vals], names=['lon', 'lat', 'z'])
    return data


def get_unique_points(points_around_input):
    profiles = points_around_input.groupby(['lat', 'lon'])
    

    unique_points = []
    unique_pairs = 0
    for (lat, lon), group in profiles: 
        if 'z' in group.columns:
            sort_column = 'z'
            columns = ['z', 'lithoklasse_material', 'lithoklasse_color']
        else:
            sort_column = 'z_min'
            columns = ['z_min', 'z_max', 'lithoklasse_material', 'lithoklasse_color'] 
        
        layers = group[columns].sort_values(sort_column)
        unique_points.append({
            'lat': lat,
            'lon': lon,
            'layers': layers.to_dict(orient='records')
        })
        unique_pairs +=1
    print(f"processed {unique_pairs} unique lat/lon pairs")
        
    profiles = dict()
    for g in unique_points:
        profiles[tuple([g['lat'], g['lon']])] = DataFrame(g['layers']) 

    return profiles, unique_points


def plot_voxel(
    unique_points, layer_label, dx=0.002, dy=0.002, elev=30, azim=60, save=False, save_name=None, 
    display_plot=False, figsize=(18,6)
    ):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    for pt in unique_points:
        x, y = pt['lon'], pt['lat']
        layers = pt[layer_label]

        # Sort depending on available depth keys
        if "z" in layers[0]:
            layers = sorted(layers, key=lambda l: l['z'])
        else:
            layers = sorted(layers, key=lambda l: l['z_min'])

        for i, lyr in enumerate(layers):

            # --- CASE 1: old format ('z') ---
            if "z" in lyr and "z_min" not in lyr:
                z_bottom = lyr['z']
                z_top = layers[i+1]['z'] if i < len(layers)-1 else lyr['z'] + 0.5

            # --- CASE 2: new format ('z-start', 'z-end') ---
            else:
                z_bottom = lyr['z_min']
                z_top = lyr['z_max']

            color = hex_to_rgb(lyr['lithoklasse_color'])

            xx = [x - dx/2, x + dx/2]
            yy = [y - dy/2, y + dy/2]
            zz = [z_bottom, z_top]

            corners = array([
                [xx[0], yy[0], zz[0]],
                [xx[1], yy[0], zz[0]],
                [xx[1], yy[1], zz[0]],
                [xx[0], yy[1], zz[0]],
                [xx[0], yy[0], zz[1]],
                [xx[1], yy[0], zz[1]],
                [xx[1], yy[1], zz[1]],
                [xx[0], yy[1], zz[1]],
            ])

            faces = [
                [corners[0], corners[1], corners[2], corners[3]],
                [corners[4], corners[5], corners[6], corners[7]],
                [corners[0], corners[1], corners[5], corners[4]],
                [corners[1], corners[2], corners[6], corners[5]],
                [corners[2], corners[3], corners[7], corners[6]],
                [corners[3], corners[0], corners[4], corners[7]],
            ]

            ax.add_collection3d(
                Poly3DCollection(
                    faces,
                    facecolors=color,
                    linewidths=0.,
                    edgecolors='none',
                    alpha=0.8
                )
            )

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Depth')

    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    
    if display_plot:
        plt.show()
    if save:
        fig.savefig(f"{save_name}_{str(azim)}.png")
        plt.close(fig)
    
    return fig


def plot_voxel_comparison(
    unique_points_A, unique_points_B, layer_label, dx=0.002, dy=0.002, elev=30, azim=60, save=False, save_name=None,
    display_plot=False, figsize=(20, 8)
    ):
    """
    Plot and compare two voxel models side-by-side.

    Parameters
    ----------
    unique_points_A : list
        First voxel dataset.
    unique_points_B : list
        Second voxel dataset.
    layer_label : str
        Key for layer information in each point dict.
    """
    
    fig = plt.figure(figsize=figsize)

    # --- LEFT subplot (Model A) ---
    ax1 = fig.add_subplot(121, projection='3d')
    _plot_voxels_on_axis(ax1, unique_points_A, layer_label, dx, dy)
    ax1.set_title("Model A")
    ax1.view_init(elev=elev, azim=azim)

    # --- RIGHT subplot (Model B) ---
    ax2 = fig.add_subplot(122, projection='3d')
    _plot_voxels_on_axis(ax2, unique_points_B, layer_label, dx, dy)
    ax2.set_title("Model B")
    ax2.view_init(elev=elev, azim=azim)

    plt.tight_layout()

    if display_plot:
        plt.show()

    if save:
        fig.savefig(f"{save_name}_{str(azim)}.png")
        plt.close(fig)

    return fig


# Helper function reused by both subplots
def _plot_voxels_on_axis(ax, unique_points, layer_label, dx, dy):
    for pt in unique_points:
        x, y = pt['lon'], pt['lat']
        layers = pt[layer_label]

        # Sort depending on available keys
        if "z" in layers[0]:
            layers = sorted(layers, key=lambda l: l['z'])
        else:
            layers = sorted(layers, key=lambda l: l['z_min'])

        for i, lyr in enumerate(layers):

            # --- CASE 1: old format, single depth 'z' ---
            if "z" in lyr and "z_min" not in lyr:
                z_bottom = lyr['z']
                z_top = layers[i+1]['z'] if i < len(layers)-1 else lyr['z'] + 0.5

            # --- CASE 2: new format, interval depth ---
            else:
                z_bottom = lyr['z_min']
                z_top = lyr['z_max']

            color = hex_to_rgb(lyr['lithoklasse_color'])

            xx = [x - dx/2, x + dx/2]
            yy = [y - dy/2, y + dy/2]
            zz = [z_bottom, z_top]

            corners = array([
                [xx[0], yy[0], zz[0]],
                [xx[1], yy[0], zz[0]],
                [xx[1], yy[1], zz[0]],
                [xx[0], yy[1], zz[0]],
                [xx[0], yy[0], zz[1]],
                [xx[1], yy[0], zz[1]],
                [xx[1], yy[1], zz[1]],
                [xx[0], yy[1], zz[1]],
            ])

            faces = [
                [corners[0], corners[1], corners[2], corners[3]],
                [corners[4], corners[5], corners[6], corners[7]],
                [corners[0], corners[1], corners[5], corners[4]],
                [corners[1], corners[2], corners[6], corners[5]],
                [corners[2], corners[3], corners[7], corners[6]],
                [corners[3], corners[0], corners[4], corners[7]],
            ]

            ax.add_collection3d(
                Poly3DCollection(
                    faces,
                    facecolors=color,
                    linewidths=0.,
                    edgecolors='none',
                    alpha=0.8
                )
            )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Depth")
