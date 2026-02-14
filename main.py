import argparse
import json
import os
from collections import Counter
from datetime import datetime
from glob import glob

import geopandas as gpd
import requests
from numpy import (any, full, isnan, nan_to_num, round, sort, unique, where,
                   zeros, zeros_like)
from pandas import DataFrame, concat, read_csv

import functions as ft

dir_export = 'output/'

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
    url = ("https://service.pdok.nl/cbs/gebiedsindelingen/2025/wfs/v1_0?"
           "request=GetFeature&service=WFS&version=2.0.0&typeName=gemeente_gegeneraliseerd&outputFormat=json")
    municipalities = gpd.read_file(url)
    rotterdam = municipalities[municipalities["statnaam"] == "Rotterdam"]
    rotterdam_rd = rotterdam.to_crs(epsg=projection_rd_amersfoort.split('epsg:')[1])

    gdf_points = gpd.GeoDataFrame(
        data, 
        geometry=gpd.points_from_xy(data.reset_index().x, data.reset_index().y), 
        crs=projection_rd_amersfoort
    )
    points_in_rotterdam = gdf_points[gdf_points.geometry.within(rotterdam_rd.unary_union)]
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

    # remove helper columns if exist
    for c in ["_gx","_gy","_gz"]:
        if c in df:
            df.drop(columns=[c], inplace=True)

    return df


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main(args):
    user_input_address = args.input_address
    method_filling = args.method if args.method else 'mean_round'

    if not user_input_address:
        latitude, longitude = 51.9139529, 4.4711320
        print(f"No user input defined; fallback to default: {latitude}, {longitude}")
    else:
        latitude, longitude = get_coordinates_to_address(user_input_address)

    data = load_geotop_data(dir_geotop='input/GeoTOP_v01r6s1_csv_bestanden/')
    points_in_rotterdam = get_rotterdam_coordinates(data)
    points_in_rotterdam = ft.convert_rd_into_geocoordinates(points_in_rotterdam)
    points_in_rotterdam = points_in_rotterdam.reset_index()

    points_in_rotterdam['lon'] = points_in_rotterdam.geometry.x
    points_in_rotterdam['lat'] = points_in_rotterdam.geometry.y

    points_in_rotterdam['group'] = (
        (points_in_rotterdam['lithoklasse'] != points_in_rotterdam['lithoklasse'].shift()) |
        (points_in_rotterdam['lon'] != points_in_rotterdam['lon'].shift()) |
        (points_in_rotterdam['lat'] != points_in_rotterdam['lat'].shift())
    ).cumsum()

    print("Filling lithoklasse==0 values bottom-to-top...")
    filled = fill_lithoklasse_3d_vectorized(points_in_rotterdam, method=method_filling)
    print("Lithoklasse filling complete.")

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
        'lithoklasse_first':'lithoklasse',
        'lithostrat_first':'lithostrat',
        'z_min':'z_bottom',
        'z_max':'z_top'
    }).reset_index(drop=True)
    df_grouped['estimated'] = filled.groupby('group')['estimated'].max().values

    df_grouped['lithoklasse_material'] = df_grouped['lithoklasse'].map(map_lithoclasses)
    df_grouped['lithoklasse_color'] = df_grouped['lithoklasse_material'].map(material_color_mapping)

    for col in likelihood_cols:
        mean_col = f"{col}_mean"
        if mean_col in df_grouped:
            df_grouped[mean_col] = df_grouped[mean_col].round(4)

    selected_columns = [
        'lon', 'lat', 'z_top', 'z_bottom', 'lithoklasse', 'lithoklasse_material',
        *[f"{col}_mean" for col in likelihood_cols],
        'estimated'
    ]
    df_final = df_grouped[selected_columns]

    profiled = df_final.groupby(['lon','lat']).apply(
        lambda g: sorted(g.to_dict(orient='records'), key=lambda d: d['z_top'], reverse=True)
    ).reset_index(name='data')

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

    profiled[['lon','lat','batchID']].to_csv(os.path.join(output_dir,'map_coordinates2batch.txt'), index=False)
    print(f"Data successfully saved in batches under {output_dir}")


# --------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D vertical filling")
    
    parser.add_argument(
        "-input_address", type=str, default="Depot Boijmans Van Beuningen",
        help="Specify address to display."
    )
    
    parser.add_argument(
        "-method", type=str, default="mean_round",
        help="Select method for computing replacement from neighbor values: mean_round (default) or mode."
    )
        
    args = parser.parse_args()
    main(args)
    main(args)
    main(args)
