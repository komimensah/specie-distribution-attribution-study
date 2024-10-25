# Loading the required packages.
import os
import glob
import warnings
import rasterio
import numpy as np
import xarray as xr
from rasterio.transform import from_origin

# Suppress any userwarning.
warnings.filterwarnings("ignore")

# Define the input & output directories paths.
pr_dir_path = r"F:\Kanji_Projects\2024\4.Komi\nc_to_tiff\nc\gswp3v109-w5e5_counterclim_pr_global_daily"
tas_dir_path = r"F:\Kanji_Projects\2024\4.Komi\nc_to_tiff\nc\gswp3v109-w5e5_counterclim_tas_global_daily"
tasmax_dir_path = r"F:\Kanji_Projects\2024\4.Komi\nc_to_tiff\nc\gswp3v109-w5e5_counterclim_tasmax_global_daily"
tasmin_dir_path = r"F:\Kanji_Projects\2024\4.Komi\nc_to_tiff\nc\gswp3v109-w5e5_counterclim_tasmin_global_daily"
output_dir_path = r"F:\Kanji_Projects\2024\4.Komi\nc_to_tiff\bioclim_var_tif"
os.makedirs(output_dir_path, exist_ok=True)

# Use glob to find all NetCDF files in the respective input directories.
file_paths_pr = glob.glob(os.path.join(pr_dir_path, "*.nc"))
file_paths_tas = glob.glob(os.path.join(tas_dir_path, "*.nc"))
file_paths_tasmax = glob.glob(os.path.join(tasmax_dir_path, "*.nc"))
file_paths_tasmin = glob.glob(os.path.join(tasmin_dir_path, "*.nc"))

# List of file path lists.
all_file_paths = [file_paths_pr, file_paths_tas, file_paths_tasmax, file_paths_tasmin]

# Check if netcdf files exist.
for file_path in all_file_paths:
    if not file_path:
        raise OSError(f"No NetCDF files found in: {file_path}")


# Merge NetCDF files along the coordinates dimension.
pr_dt = xr.open_mfdataset(file_paths_pr, combine="by_coords")
tas_dt = xr.open_mfdataset(file_paths_tas, combine="by_coords")
tasmax_dt = xr.open_mfdataset(file_paths_tasmax, combine="by_coords")
tasmin_dt = xr.open_mfdataset(file_paths_tasmin, combine="by_coords")

print("Loaded all the variables.")

# Converting variable units.
pr = pr_dt["pr"] * 86400  # Convert from kg m-2 s-1 to mm/day.
tas = tas_dt["tas"] - 273.15  # Convert from Kelvin to Celsius.
tasmax = tasmax_dt["tasmax"] - 273.15  # Max temperature.
tasmin = tasmin_dt["tasmin"] - 273.15  # Min temperature.

print("Converted the units of the variables to the desired units.")

# Resample daily data to monthly averages/totals.
pr_monthly = pr.resample(time="ME").sum(dim="time", keep_attrs=True)  # Monthly total precipitation.
tas_monthly = tas.resample(time="ME").mean(dim="time", keep_attrs=True)  # Monthly Avg temperature.
tasmax_monthly = tasmax.resample(time="ME").mean(dim="time", keep_attrs=True)  # Monthly max temperature.
tasmin_monthly = tasmin.resample(time="ME").mean(dim="time", keep_attrs=True)  # Monthly min temperature.

print("Completed resampling the variables from daily to monthly averages/totals.")

# Creating Long-Term monthly averages by grouping by month and then take the mean over all years.
pr_monthly_ltm = pr_monthly.groupby("time.month").mean("time")  # Long Term Monthly Average for pr.
tas_monthly_ltm = tas_monthly.groupby("time.month").mean("time")  # Long Term Monthly Average for tas.
tasmax_monthly_ltm = tasmax_monthly.groupby("time.month").mean("time")  # Long Term Monthly Average for tasmax.
tasmin_monthly_ltm = tasmin_monthly.groupby("time.month").mean("time")  # Long Term Monthly Average for tasmin.

print("Completed converting the variables into Long-Term monthly averages.")

# Ensure 'month' becomes a coordinate
pr_monthly_ltm = pr_monthly_ltm.assign_coords(month=pr_monthly_ltm["month"])
tas_monthly_ltm = tas_monthly_ltm.assign_coords(month=tas_monthly_ltm["month"])
tasmax_monthly_ltm = tasmax_monthly_ltm.assign_coords(month=tasmax_monthly_ltm["month"])
tasmin_monthly_ltm = tasmin_monthly_ltm.assign_coords(month=tasmin_monthly_ltm["month"])

# Calculate Bioclimatic Variables Directly from the Long-Term monthly averaged Data.
BIO1 = tas_monthly_ltm.mean("month")  # Annual Mean Temperature.
BIO2 = (tasmax_monthly_ltm - tasmin_monthly_ltm).mean("month")  # Mean Diurnal Range.
BIO4 = tas_monthly_ltm.std("month") * 100  # Temperature Seasonality.
BIO5 = tasmax_monthly_ltm.max("month")  # Max Temperature of Warmest Month.
BIO6 = tasmin_monthly_ltm.min("month")  # Min Temperature of Coldest Month.
BIO7 = BIO5 - BIO6  # Temperature Annual Range (BIO5 - BIO6).
BIO3 = (BIO2 / BIO7) * 100  # Isothermality (BIO2 / BIO7) × 100.

print("Completed deriving: BIO1, BIO2, BIO3, BIO4, BIO5, BIO6, BIO7")


# Function to calculate the quarterly (3-months) sum/average.
def compute_rolling_quarter(data, metric="sum"):
    """Compute the rolling 3-month quarter for a given dataset.
    Args:
    - data: xarray DataArray (e.g., precipitation or temperature).
    - metric: "sum" for precipitation or "mean" for temperature.

    Returns:
    - rolling_quarter: the 3-month rolling value (sum or mean) over time.
    """
    if metric == "sum":
        return data.rolling(month=3, center=False, min_periods=1).sum(skipna=True)
    elif metric == "mean":
        return data.rolling(month=3, center=False, min_periods=1).mean(skipna=True)


# Function to identify the months corresponding to the extreme quarters.
def extract_extreme_quarter(data, extreme="max"):
    """Extract the month index for either the max or min 3-month rolling window.
    Args:
    - data: 3-month rolling data array (temperature/precipitation).
    - extreme: "max" or "min" depending on the desired quarter.

    Returns:
    - indices of the max/min quarter.
    """
    if extreme == "max":
        # Ignore NaN values by replacing them with -inf ensures that the NaNs won't be selected as the maximum.
        return data.fillna(float("-inf")).argmax(dim="month")
    elif extreme == "min":
        # Ignore NaN values by replacing them with inf ensures that the NaNs won't be selected as the minimum.
        return data.fillna(float("inf")).argmin(dim="month")


# Rolling sums for precipitation and rolling means for temperature (3-month quarters).
pr_rolling = compute_rolling_quarter(pr_monthly_ltm, metric="sum")
tas_rolling = compute_rolling_quarter(tas_monthly_ltm, metric="mean")
tasmax_rolling = compute_rolling_quarter(tasmax_monthly_ltm, metric="mean")
tasmin_rolling = compute_rolling_quarter(tasmin_monthly_ltm, metric="mean")

print("Completed calculating the rolling quarterly precipitation sums and temperature averages")

# BIO8: Mean Temp of Wettest Quarter (max 3-month pr).
BIO8_idx = extract_extreme_quarter(pr_rolling, extreme="max").compute()  # Extract the months indices of the maximum 3-month rolling pr window.
BIO8 = tas_monthly_ltm.where(tas_monthly_ltm["month"].isin(BIO8_idx), drop=True).mean(dim="month")  # Filter the data on its months dimension by the months indices in BIO8_idx and then get the average value.

# BIO9: Mean Temp of Driest Quarter (min 3-month pr).
BIO9_idx = extract_extreme_quarter(pr_rolling, extreme="min").compute()  # Extract the months indices of the minimum 3-month rolling pr window.
BIO9 = tas_monthly_ltm.where(tas_monthly_ltm["month"].isin(BIO9_idx), drop=True).mean(dim="month")  # Filter the data on its months dimension by the months indices in BIO9_idx and then get the average value.

print("Completed deriving: BIO8, BIO9")

# BIO10: Mean Temp of Warmest Quarter (max 3-month tmax).
BIO10_idx = extract_extreme_quarter(tasmax_rolling, extreme="max").compute()  # Extract the months indices of the maximum 3-month rolling tasmax window.
BIO10 = tas_monthly_ltm.where(tas_monthly_ltm["month"].isin(BIO10_idx), drop=True).mean(dim="month")  # Filter the data on its months dimension by the months indices in BIO10_idx and then get the average value.

# BIO11: Mean Temp of Coldest Quarter (min 3-month tmin).
BIO11_idx = extract_extreme_quarter(tasmin_rolling, extreme="min").compute()  # Extract the months indices of the minimum 3-month rolling tasmin window.
BIO11 = tas_monthly_ltm.where(tas_monthly_ltm["month"].isin(BIO11_idx), drop=True).mean(dim="month")  # Filter the data on its months dimension by the months indices in BIO11_idx and then get the average value.

print("Completed deriving: BIO10, BIO11")

BIO12 = pr_monthly_ltm.sum("month")  # BIO12: Annual Precipitation.
BIO13 = pr_monthly_ltm.max("month")  # BIO13: Precipitation of Wettest Month.
BIO14 = pr_monthly_ltm.min("month")  # BIO14: Precipitation of Driest Month.
BIO15 = (pr_monthly_ltm.std("month") / pr_monthly_ltm.mean("month")) * 100  # BIO15: Precipitation Seasonality (Coefficient of Variation).

print("Completed deriving: BIO12, BIO13, BIO14, BIO15")

# BIO16: Precipitation of Wettest Quarter (max 3-month pr).
BIO16_idx = extract_extreme_quarter(pr_rolling, extreme="max").compute()  # Extract the months indices of the maximum 3-month rolling pr window.
BIO16 = pr_monthly_ltm.where(pr_monthly_ltm["month"].isin(BIO16_idx), drop=True).sum(dim="month")  # Filter the data on its months dimension by the months indices in BIO16_idx and then get the sum value.

# BIO17: Precipitation of Driest Quarter (max 3-month pr).
BIO17_idx = extract_extreme_quarter(pr_rolling, extreme="min").compute()  # Extract the months indices of the minimum 3-month rolling pr window.
BIO17 = pr_monthly_ltm.where(pr_monthly_ltm["month"].isin(BIO17_idx), drop=True).sum(dim="month")  # Filter the data on its months dimension by the months indices in BIO17_idx and then get the sum value.

print("Completed deriving: BIO16, BIO17")

# BIO18: Precipitation of Warmest Quarter (max 3-month pr).
BIO18_idx = extract_extreme_quarter(tasmax_rolling, extreme="max").compute()  # Extract the months indices of the maximum 3-month rolling tasmax window.
BIO18 = pr_monthly_ltm.where(pr_monthly_ltm["month"].isin(BIO18_idx), drop=True).sum(dim="month")  # Filter the data on its months dimension by the months indices in BIO18_idx and then get the sum value.

# BIO19: Precipitation of Coldest Quarter (max 3-month pr).
BIO19_idx = extract_extreme_quarter(tasmin_rolling, extreme="min").compute()  # Extract the months indices of the minimum 3-month rolling tasmin window.
BIO19 = pr_monthly_ltm.where(pr_monthly_ltm["month"].isin(BIO19_idx), drop=True).sum(dim="month")  # Filter the data on its months dimension by the months indices in BIO18_idx and then get the sum value.

print("Completed deriving: BIO18, BIO19")

# Exporting the bioclimatic variables as GeoTIFF rasters.
bioclim_vars = {
    "BIO1": BIO1,
    "BIO2": BIO2,
    "BIO3": BIO3,
    "BIO4": BIO4,
    "BIO5": BIO5,
    "BIO6": BIO6,
    "BIO7": BIO7,
    "BIO8": BIO8,
    "BIO9": BIO9,
    "BIO10": BIO10,
    "BIO11": BIO11,
    "BIO12": BIO12,
    "BIO13": BIO13,
    "BIO14": BIO14,
    "BIO15": BIO15,
    "BIO16": BIO16,
    "BIO17": BIO17,
    "BIO18": BIO18,
    "BIO19": BIO19,
}


# Define function to export an xarray DataArray to GeoTIFF.
def save_as_geotiff(dataarray, output_file_path, transform, crs):
    with rasterio.open(
        output_file_path,
        "w",
        driver="GTiff",
        height=dataarray.y.size,
        width=dataarray.x.size,
        count=1,
        dtype=dataarray.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(dataarray.values, 1)


# Save each bioclimatic variable as GeoTIFF.
for var_name, var_data in bioclim_vars.items():
    print(f"Exporting bioclimatic variable: {var_name}")

    # set spatial dimensions
    var_data = var_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

    # Define the CRS (coordinate reference system) - change this to match your data.
    crs = "EPSG:4326"  # WGS84 lat/lon

    # Set the crs to EPSG:4326.
    var_data = var_data.rio.write_crs(crs)

    # Reproject the data array to the desired crs and extent (-180 to 180 degrees).
    var_data_reprojected = var_data.rio.reproject(dst_crs="EPSG:4326", extent=(-180, -90, 180, 90))

    # Extract spatial coordinates and define rasterio transform (for georeferencing).
    lon = var_data["lon"].values
    lat = var_data["lat"].values

    # Define a transform using the spatial resolution of your dataset
    resolution_x = (lon[-1] - lon[0]) / len(lon)
    resolution_y = (lat[-1] - lat[0]) / len(lat)
    transform = from_origin(west=lon.min(), north=lat.max(), xsize=resolution_x, ysize=-resolution_y)

    save_as_geotiff(var_data_reprojected, os.path.join(output_dir_path, f"{var_name}.tif"), transform, crs)

print("All bioclimatic variables have been exported & saved as GeoTIFFs.")