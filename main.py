import argparse
import json
import os
from collections import Counter
from datetime import datetime
from glob import glob

import geopandas as gpd
import pyarrow.dataset as ds
import requests
from joblib import Parallel, delayed
from numpy import (abs, any, argmax, array, full, isnan, nan_to_num, round,
                   sort, stack, unique, where, zeros, zeros_like)
from pandas import DataFrame, concat, read_csv
from shapely.geometry import Point
from shapely.strtree import STRtree
from tqdm import tqdm

import functions as ft

dir_export = 'output/'
dir_surface = 'input/maaiveld_dtm/'

projection_rd_amersfoort = 'epsg:28992'
projection_geocoordinates = 'epsg:4326'

map_lithoclasses = {
    0: 'NaN', 1: 'veen', 2: 'klei', 3: 'kleiig_zand', 
    4: 'vervallen', 5: 'zand_fijn', 6: 'zand_matig_grof',
    7: 'zand_grof', 8: 'grind', 9: 'schelpen'
}

material_color_mapping = {
    'NaN': '#ffffff',
    'veen': '#64564c',
    'klei':'#b2a38d', 
    'kleiig_zand':'#8a8783', 
    'vervallen':'#ee82ee', 
    'zand_fijn':'#000000', 
    'zand_matig_grof': '#c5c5c5',  
    'zand_grof': '#616160',
    'grind': '#ffff82',
    'schelpen': '#eb611e' 
}

MAX_BYTES = 5 * 1024 * 1024  # 5 MB


# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------
def get_coordinates_to_address(user_input_address):
    try:
        geo = requests.get(
            url="https://nominatim.openstreetmap.org/search",
            headers={"User-Agent": "CaraLogic (contact: silvia@caralogic.com)"}, 
            params={"q": user_input_address, "format": "json", "limit": 1}
        )
        geo.raise_for_status()
        if len(geo.json()) == 0:
            print(f"No data found for {user_input_address}")
            latitude, longitude = None, None
        else:
            location = geo.json()[0]
            latitude, longitude = float(location['lat']), float(location['lon'])
    except:
        latitude, longitude = 51.9139529, 4.4711320
    print(f"Coordinates found for {user_input_address}: {latitude}, {longitude} (lat, lon)")
    return latitude, longitude


def load_geotop_data(dir_geotop):
    ls_files = sorted(glob(os.path.join(dir_geotop, '*.csv')))
    ls_data = [read_csv(file, index_col=[0,1,2], engine="pyarrow") for file in ls_files]
    data = concat(ls_data).sort_index()
    return data


def get_rotterdam_coordinates(data):
    url = (
        "https://service.pdok.nl/cbs/gebiedsindelingen/2025/wfs/v1_0?"
        "request=GetFeature&service=WFS&version=2.0.0&typeName=gemeente_gegeneraliseerd&outputFormat=json"
        )
    municipalities = gpd.read_file(url)
    rotterdam = municipalities[municipalities["statnaam"] == "Rotterdam"]
    rotterdam_rd = rotterdam.to_crs(epsg=projection_rd_amersfoort.split('epsg:')[1])

    gdf_points = gpd.GeoDataFrame(
        data, 
        geometry=gpd.points_from_xy(data.reset_index().x, data.reset_index().y), 
        crs=projection_rd_amersfoort
    )
    points_in_rotterdam = gdf_points[gdf_points.geometry.within(rotterdam_rd.union_all())]
    print(f"Data points available within Rotterdam {points_in_rotterdam.shape}")
    return points_in_rotterdam


def fill_lithoklasse_3d_vectorized(
    df: DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    z_col: str = "z",
    litho_col: str = "lithoklasse",
    method: str = "mean_round",
    round_decimals: int = 6
) -> DataFrame:
    """
    Vectorized bottom-to-top filling of lithoklasse==0 in 3D using NumPy slicing.
    Returns DataFrame with updated lithoklasse and estimated column.
    """

    df = df.copy()
    df["estimated"] = False

    xs = round(df[lon_col].values.astype(float), round_decimals)
    ys = round(df[lat_col].values.astype(float), round_decimals)
    zs = round(df[z_col].values.astype(float), round_decimals)

    ux = sort(unique(xs))
    uy = sort(unique(ys))
    uz = sort(unique(zs))
    nx, ny, nz = len(ux), len(uy), len(uz)

    x_map = {v:i for i,v in enumerate(ux)}
    y_map = {v:i for i,v in enumerate(uy)}
    z_map = {v:i for i,v in enumerate(uz)}

    arr = full((nx, ny, nz), float('nan'), dtype=float)
    idx_map = full((nx, ny, nz), -1, dtype=int)
    present = zeros((nx, ny, nz), dtype=bool)

    for idx, row in df.iterrows():
        ix, iy, iz = x_map[row[lon_col]], y_map[row[lat_col]], z_map[row[z_col]]
        arr[ix, iy, iz] = float(row[litho_col])
        idx_map[ix, iy, iz] = idx
        present[ix, iy, iz] = True

    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
    offsets_same = [o for o in offsets if o != (0,0)]

    for iz in range(1, nz):
        mask_zero = (arr[:,:,iz] == 0) & present[:,:,iz]
        if not any(mask_zero):
            continue

        filled_values = zeros_like(arr[:,:,iz])
        filled_count = zeros_like(arr[:,:,iz])

        for dx, dy in offsets:
            x_start = max(0, dx*-1)
            x_end = nx - max(0, dx)
            y_start = max(0, dy*-1)
            y_end = ny - max(0, dy)

            if iz > 0:
                neighbor = arr[x_start:x_end, y_start:y_end, iz-1]
                filled_values[x_start:x_end, y_start:y_end] += nan_to_num(neighbor)
                filled_count[x_start:x_end, y_start:y_end] += (~isnan(neighbor)) & (neighbor != 0)

            if (dx, dy) in offsets_same:
                neighbor = arr[x_start:x_end, y_start:y_end, iz]
                filled_values[x_start:x_end, y_start:y_end] += nan_to_num(neighbor)
                filled_count[x_start:x_end, y_start:y_end] += (~isnan(neighbor)) & (neighbor != 0)

        new_vals = zeros_like(arr[:,:,iz])
        mask_fill = (filled_count > 0) & mask_zero

        if method == "mean_round":
            new_vals[mask_fill] = round(filled_values[mask_fill] / filled_count[mask_fill])
        else:
            ix_fill, iy_fill = where(mask_fill)
            for x, y in zip(ix_fill, iy_fill):
                neighbors = []
                for dx, dy in offsets:
                    nx2, ny2 = x+dx, y+dy
                    if 0 <= nx2 < nx and 0 <= ny2 < ny:
                        v = arr[nx2, ny2, iz-1]
                        if not isnan(v) and int(v) != 0:
                            neighbors.append(int(round(v)))
                for dx, dy in offsets_same:
                    nx2, ny2 = x+dx, y+dy
                    if 0 <= nx2 < nx and 0 <= ny2 < ny:
                        v = arr[nx2, ny2, iz]
                        if not isnan(v) and int(v) != 0:
                            neighbors.append(int(round(v)))
                if neighbors:
                    cnt = Counter(neighbors)
                    max_count = max(cnt.values())
                    candidates = [val for val,c in cnt.items() if c==max_count]
                    new_vals[x,y] = int(min(candidates))

        arr[:,:,iz][mask_fill] = new_vals[mask_fill]

    ix_arr, iy_arr, iz_arr = where(present)
    for x, y, z in zip(ix_arr, iy_arr, iz_arr):
        ridx = idx_map[x, y, z]
        orig_val = df.at[ridx, litho_col]
        new_val = arr[x,y,z]
        df.at[ridx, litho_col] = int(new_val)
        if orig_val == 0 and new_val != 0:
            df.at[ridx, "estimated"] = True

    for c in ["_gx","_gy","_gz"]:
        if c in df:
            df.drop(columns=[c], inplace=True)

    return df


def fill_lithoklasse_3d_sparse(
    df: DataFrame,
    z_col: str = "z",
    litho_col: str = "lithoklasse",
    method: str = "mean_round",
    delta: float = 0.0005
) -> DataFrame:
    """
    Memory-efficient 3D bottom-to-top filling of lithoklasse==0.
    For each zero layer, looks at the layer immediately below
    in the 3x3 neighborhood (lon/lat ± delta) and fills using
    mean or mode of non-zero neighbors.
    
    Parameters:
        df: DataFrame with columns ['lon', 'lat', z_col, litho_col']
        z_col: column representing depth
        litho_col: column representing lithoklasse
        method: "mean_round" or "mode"
        delta: search radius in lon/lat for neighbors
    Returns:
        DataFrame with filled lithoklasse and estimated flag
    """
    
    df = df.copy()
    df["estimated"] = False
    
    coord_dict = {}
    for idx, row in df.iterrows():
        coord_dict[(row['lon'], row['lat'], row[z_col])] = idx
    
    for (lon, lat), group in df.groupby(['lon','lat']):
        group_sorted = group.sort_values(z_col)
        zs = group_sorted[z_col].values
        for i in range(1, len(zs)):
            idx_curr = coord_dict[(lon, lat, zs[i])]
            if df.at[idx_curr, litho_col] == 0:
                
                z_below = zs[i-1]
                neighbor_mask = (
                    (df[z_col] == z_below) &
                    (df['lon'].between(lon - delta, lon + delta)) &
                    (df['lat'].between(lat - delta, lat + delta)) &
                    (df[litho_col] != 0)
                )
                neighbors = df.loc[neighbor_mask, litho_col].values
                
                if len(neighbors) == 0:
                    continue 
                
                if method == "mean_round":
                    fill_val = int(round(neighbors.mean()))
                elif method == "mode":
                    cnt = Counter(neighbors)
                    max_count = max(cnt.values())
                    candidates = [v for v,c in cnt.items() if c==max_count]
                    fill_val = int(min(candidates))
                else:
                    raise ValueError(f"Unknown method '{method}'")
                
                df.at[idx_curr, litho_col] = fill_val
                df.at[idx_curr, "estimated"] = True
                
    return df


def fill_lithoklasse_fast(df, z_col='z', litho_col='lithoklasse', delta=0.0005, method='mean_round'):
    df = df.sort_values(['lon','lat',z_col]).copy()
    df['estimated'] = False

    lons = df['lon'].values
    lats = df['lat'].values
    zs = df[z_col].values
    litho = df[litho_col].values

    n = len(df)

    for i in tqdm(range(1, n), desc="Filling lithoklasse"):
        if lons[i] == lons[i-1] and lats[i] == lats[i-1]:
            
            if litho[i] == 0:
                z_below = zs[i-1]

                mask = (
                    (zs == z_below) &
                    (abs(lons - lons[i]) <= delta) &
                    (abs(lats - lats[i]) <= delta) &
                    (litho != 0)
                )

                neighbors = litho[mask]

                if neighbors.size == 0:
                    continue

                if method == "mean_round":
                    fill_val = int(round(neighbors.mean()))
                else:
                    vals, counts = unique(neighbors, return_counts=True)
                    fill_val = vals[argmax(counts)]

                litho[i] = fill_val
                df.at[df.index[i], 'estimated'] = True

    df[litho_col] = litho
    return df


def fill_lithoklasse_numpy(df, z_col='z', litho_col='lithoklasse', delta=0.0005, method='mean_round'):
    """
    Vectorized, bottom-to-top filling of lithoklasse==0 per column.
    Pure NumPy, no Numba. Maintains your original logic.
    
    Parameters:
        df: pandas DataFrame with ['lon', 'lat', z_col, litho_col']
        z_col: column representing depth
        litho_col: column representing lithoklasse
        delta: search radius in lon/lat for neighbor averaging
        method: "mean_round" or "mode"
        
    Returns:
        df: same DataFrame with filled lithoklasse and 'estimated' flag
    """

    df = df.sort_values(['lon', 'lat', z_col]).copy()
    df['estimated'] = False

    lons = df['lon'].values
    lats = df['lat'].values
    zs = df[z_col].values
    litho = df[litho_col].values
    n = len(df)

    coords = stack([lons, lats, zs], axis=1)
    unique_cols, inverse_idx = unique(stack([lons, lats], axis=1), axis=0, return_inverse=True)

    for col_idx, (lon_val, lat_val) in enumerate(unique_cols):
        col_mask = (inverse_idx == col_idx)
        idxs = where(col_mask)[0]
        zs_col = zs[idxs]
        litho_col = litho[idxs]

        for i in range(1, len(idxs)):
            idx_curr = idxs[i]
            if litho[idx_curr] == 0:
                z_below = zs[idxs[i-1]]

                neighbor_mask = (
                    (zs == z_below) &
                    (abs(lons - lon_val) <= delta) &
                    (abs(lats - lat_val) <= delta) &
                    (litho != 0)
                )
                neighbors = litho[neighbor_mask]

                if neighbors.size == 0:
                    continue

                if method == "mean_round":
                    fill_val = int(round(neighbors.mean()))
                elif method == "mode":
                    vals, counts = unique(neighbors, return_counts=True)
                    fill_val = vals[argmax(counts)]
                else:
                    raise ValueError(f"Unknown method '{method}'")

                litho[idx_curr] = fill_val
                df.at[df.index[idx_curr], 'estimated'] = True

    df[litho_col] = litho
    return df


def fill_lithoklasse_numpy_fast(
    df,
    z_col='z',
    litho_col='lithoklasse',
    delta=0.0005,
    method='mean_round',
    show_progress=True
):
    """
    Optimized bottom-to-top filling of lithoklasse==0 per column.
    Much faster than naive neighbor scanning.
    """

    df = df.sort_values(['lon', 'lat', z_col]).copy()
    df['estimated'] = False

    lons = df['lon'].to_numpy()
    lats = df['lat'].to_numpy()
    zs = df[z_col].to_numpy()
    litho = df[litho_col].to_numpy()

    unique_z, z_inverse = unique(zs, return_inverse=True)
    z_groups = {i: where(z_inverse == i)[0] for i in range(len(unique_z))}

    coords_xy = stack([lons, lats], axis=1)
    unique_cols, col_inverse = unique(coords_xy, axis=0, return_inverse=True)

    total_cols = len(unique_cols)
    iterator = range(total_cols)
    if show_progress:
        iterator = tqdm(iterator, desc="Filling columns")

    for col_idx in iterator:
        idxs = where(col_inverse == col_idx)[0]

        for i in range(1, len(idxs)):
            idx_curr = idxs[i]

            if litho[idx_curr] != 0:
                continue

            z_below = zs[idxs[i - 1]]
            z_group_idx = where(unique_z == z_below)[0][0]
            layer_indices = z_groups[z_group_idx]

            dx = abs(lons[layer_indices] - lons[idx_curr])
            dy = abs(lats[layer_indices] - lats[idx_curr])

            neighbor_mask = (
                (dx <= delta) &
                (dy <= delta) &
                (litho[layer_indices] != 0)
            )

            neighbors = litho[layer_indices][neighbor_mask]

            if neighbors.size == 0:
                continue

            if method == "mean_round":
                fill_val = int(round(neighbors.mean()))
            elif method == "mode":
                vals, counts = unique(neighbors, return_counts=True)
                fill_val = vals[argmax(counts)]
            else:
                raise ValueError(f"Unknown method '{method}'")

            litho[idx_curr] = fill_val
            df.iloc[idx_curr, df.columns.get_loc('estimated')] = True

    df[litho_col] = litho
    return df


def aggregate_to_same_class(filled):
    likelihood_cols = [col for col in filled.columns if col.startswith('kans_')]
    agg_dict = {
        'lon': 'first',
        'lat': 'first',
        'z': ['min','max'],
        'lithostrat':'first',
        'lithoklasse':'first',
    }
    for col in likelihood_cols:
        agg_dict[col] = 'mean'

    df_grouped = filled.groupby('group').agg(agg_dict)
    df_grouped.columns = [f"{c[0]}_{c[1]}" if isinstance(c, tuple) else c for c in df_grouped.columns]
    df_grouped = df_grouped.rename(columns={
        'lon_first':'lon',
        'lat_first':'lat',
        'lithoklasse_first':'lithoklasse_id',
        'lithostrat_first':'lithostrat',
        'z_min':'z_bottom',
        'z_max':'z_top'
    }).reset_index(drop=True)
    
    df_grouped['estimated'] = filled.groupby('group')['estimated'].max().values
    df_grouped['lithoklasse'] = df_grouped['lithoklasse_id'].map(map_lithoclasses)
    df_grouped['lithoklasse_color'] = df_grouped['lithoklasse'].map(material_color_mapping)

    for col in likelihood_cols:
        mean_col = f"{col}_mean"
        if mean_col in df_grouped:
            df_grouped[mean_col] = df_grouped[mean_col].round(4)

    selected_columns = [
        'lon', 'lat', 'z_top', 'z_bottom', 'lithoklasse_id', 'lithoklasse',
        *[f"{col}_mean" for col in likelihood_cols], 'estimated'
    ]
    return df_grouped[selected_columns]
    

def saving_batches(profiled, dir_export):
    list_of_lists = profiled['data'].tolist()
    
    output_dir = os.path.join(dir_export, f"json_5MB_chunks_{datetime.now().date().isoformat()}/")
    os.makedirs(output_dir, exist_ok=True)

    batch, batch_size, file_index = [], 0, 0
    profiled['batchID'] = -1

    for idx, sublist in enumerate(list_of_lists):
        sublist_bytes = len(json.dumps(sublist, separators=(',', ':')).encode('utf-8'))
        if batch_size + sublist_bytes > MAX_BYTES and batch:
            file_name = os.path.join(output_dir, f"litho_batch_{file_index}.json")
            with open(file_name, 'w') as f:
                json.dump(batch, f, separators=(',', ':'))
            batch, batch_size = [], 0
            file_index += 1
        batch.append(sublist)
        batch_size += sublist_bytes
        profiled.loc[idx, 'batchID'] = file_index

    if batch:
        file_name = os.path.join(output_dir, f"litho_batch_{file_index}.json")
        with open(file_name, 'w') as f:
            json.dump(batch, f, separators=(',', ':'))

    #profiled[['lon','lat','batchID']].to_csv(os.path.join(output_dir,'map_coordinates2batch.txt'), index=False)
    batch_summary = profiled.groupby('batchID').agg({
        'lon': ['min', 'max'],
        'lat': ['min', 'max']
    })

    batch_summary.columns = ['minLon', 'maxLon', 'minLat', 'maxLat']
    batch_summary = batch_summary.reset_index()

    batch_summary_dict = batch_summary.set_index('batchID').to_dict(orient='index')

    batch_summary_file = os.path.join(output_dir, 'batch_index.json')
    with open(batch_summary_file, 'w') as f:
        json.dump(batch_summary_dict, f, indent=2)

    print(f"Batch bounding boxes saved as {batch_summary_file}")
    return output_dir


def saving_batches_v2(df_merged, dir_export):
    batch = {}
    batch_index = 1
    current_size = 0
    mapping_rows = []

    output_dir = dir_export + f"json_5MB_chunks_{datetime.now().date().isoformat()}/"
    os.makedirs(output_dir, exist_ok=True)

    layer_cols = [
        "z_top", "z_bottom", "lithoklasse_id", "lithoklasse",
        "kans_1_veen_mean", "kans_2_klei_mean", "kans_3_kleiig_zand_mean",
        "kans_4_vervallen_mean", "kans_5_zand_fijn_mean", "kans_6_zand_matig_grof_mean",
        "kans_7_zand_grof_mean", "kans_8_grind_mean", "kans_9_schelpen_mean", "estimated"
    ]
    
    for (lat, lon), group in df_merged.groupby(['lat', 'lon']):
        surface_level = group['surface_level_m_NAP'].iloc[0]
        layers = group[layer_cols].to_dict(orient='records')
        
        key = f"{lat},{lon},{surface_level}"
        batch[key] = layers
        
        mapping_rows.append({
            "lat": lat,
            "lon": lon,
            "surface_level": surface_level,
            "batch_id": batch_index
        })
        
        current_size += len(json.dumps({key: layers}))
        
        if current_size >= MAX_BYTES:
            filename = output_dir + f"litho_batch_{batch_index}.json"
            with open(filename, "w") as f:
                json.dump(batch, f, indent=2)
            print(f"Saved {filename} ({current_size / 1024 / 1024:.2f} MB)")

            batch_index += 1
            batch = {}
            current_size = 0

    if batch:
        filename = output_dir + f"litho_batch_{batch_index}.json"
        with open(filename, "w") as f:
            json.dump(batch, f, indent=2)
        print(f"Saved {filename} ({current_size / 1024 / 1024:.2f} MB)")


    batch_summary = DataFrame(mapping_rows).groupby('batch_id').agg({
        'lon': ['min', 'max'],
        'lat': ['min', 'max']
    })

    batch_summary.columns = ['minLon', 'maxLon', 'minLat', 'maxLat']
    batch_summary = batch_summary.reset_index()
    batch_summary = batch_summary.rename(columns={'batch_id': 'batchID'})

    batch_summary_dict = batch_summary.set_index('batchID').to_dict(orient='index')
    batch_summary_file = os.path.join(output_dir, 'batch_index.json')
    with open(batch_summary_file, 'w') as f:
        json.dump(batch_summary_dict, f, indent=2)

    print(f"Batch bounding boxes saved as {batch_summary_file}")

    #df_mapping = DataFrame(mapping_rows)
    #df_mapping.to_csv(output_dir + "batch_index.txt", index=False)
    return output_dir

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main(args):
    method_filling = args.method if args.method else 'mean_round'
    print('Loading geotop data...')
    data = load_geotop_data(dir_geotop='input/GeoTOP_v01r6s1_csv_bestanden/')
    
    print('Crop data to the city of Rotterdam...')
    points_in_rotterdam = get_rotterdam_coordinates(data)
    points_in_rotterdam = ft.convert_rd_into_geocoordinates(points_in_rotterdam)
    points_in_rotterdam = points_in_rotterdam.reset_index()
    
    points_in_rotterdam['group'] = (
        (points_in_rotterdam['lithoklasse'] != points_in_rotterdam['lithoklasse'].shift()) |
        (points_in_rotterdam['lon'] != points_in_rotterdam['lon'].shift()) |
        (points_in_rotterdam['lat'] != points_in_rotterdam['lat'].shift())
    ).cumsum()

    print("Filling lithoklasse==0 values bottom-to-top (column-wise)...")
    filled = fill_lithoklasse_numpy_fast(
        df=points_in_rotterdam, z_col='z', litho_col='lithoklasse', delta=0.0005, method=method_filling,
        )
    print("Lithoklasse filling complete.")

    print('Aggregate to same class')
    cropped = aggregate_to_same_class(filled)

    print('Get surface layer information')
    unique_pairs_coords_selected = ft.get_surface_layer_of_area(points_in_rotterdam, dir_surface)

    df_merged = cropped.merge(
        unique_pairs_coords_selected[['lat', 'lon', 'surface_level_m_NAP']],
        on=['lat', 'lon'],
        how='left'
        )
    df_merged.loc[df_merged['surface_level_m_NAP'].isna(), 'surface_level_m_NAP'] = 0

    output_dir = saving_batches_v2(df_merged, dir_export)
    print(f"Data successfully saved in batches under {output_dir}")


# --------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D vertical filling")
    
    parser.add_argument(
        "-method", type=str, default="mean_round",
        help="Select method for computing replacement from neighbor values: mean_round (default) or mode."
    )
        
    args = parser.parse_args()
    main(args)
