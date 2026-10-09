import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import matplotlib.dates as mdates
from datetime import datetime
import pangaeapy.pandataset as pd
from scipy.stats import linregress
import numpy as np
import pandas as ppd


from IPython.display import IFrame
#%matplotlib inline
import matplotlib.pyplot as plt
import getpass
import xarray as xr
import panel.widgets as pnw
import panel as pn
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import copernicusmarine
import matplotlib.dates as mdates
from datetime import datetime
import os
# To avoid warning messages
import warnings
warnings.filterwarnings('ignore')
x_map = {
    "day": "cmems_mod_bal_bgc_my_P1D-m",
    "month": "cmems_mod_bal_bgc_my_P1M-m",
    "year": "cmems_mod_bal_bgc_my_P1Y-m",
}


def load_data_of_one_parameter_cop(parameter_1, parameter_2, date_beginn, date_end, longitude=10.039330 , latitude=54.529500, min_longitude=9.5 ,max_longitude=10.5, min_latitude=54, max_latitude=55, depth_min=0.51, depth_max=1.5, increments="day"):
    port_lon = slice(min_longitude, max_longitude)     #longitude
    port_lat = slice(min_latitude, max_latitude)    #latitude
    dates_with_time_start = f"{date_beginn}T00:00:00"
    dates_with_time_end = f"{date_end}T00:00:00"
    import copernicusmarine

    copernicusmarine.subset(
    dataset_id=x_map[increments],
    variables=[parameter_1, parameter_2],#["chl", "nh4", "no3", "nppv", "o2", "o2b", "ph", "po4", "spco2", "zsd"],#[parameter_1, parameter_2]
    minimum_longitude=min_longitude,
    maximum_longitude=max_longitude,
    minimum_latitude=min_latitude,
    maximum_latitude=max_latitude,
    start_datetime= dates_with_time_start,
    end_datetime= dates_with_time_end,
    minimum_depth= depth_min,
    maximum_depth= depth_max,
    )

    datasetID = x_map[increments]
    DS = copernicusmarine.open_dataset(dataset_id = datasetID)

    
    
    return DS, port_lon, port_lat, date_beginn, date_end, parameter_1, parameter_2

def load_into_subset(DS, parameter, target_date, date_beginn, date_end, target_depth, longitude=10.039330 , latitude=54.529500,depth_min=0.51):
    subset = DS[[parameter]].sel(time = slice(date_beginn, date_end))
    subset_T = subset[parameter].isel(depth=target_depth).sel(time = target_date, method = 'nearest')             # method='nearest' -> Method to use for inexact matches (use the nearest valid index value) selects the wanted data from the loaded data system
    datum = datetime.strptime(str(target_date), "%Y-%m-%d")

    # datetime → gewünschtes Format
    readable_date = datum.strftime("%d.%m.%Y")
    
    d_thetao = DS[parameter] \
        .sel(longitude=longitude, latitude=latitude, method='nearest') \
        .where((DS.depth >= depth_min) & (DS.depth <= 32), drop=True) \
        .sel(time=target_date)

    return subset_T, readable_date, d_thetao

def map_baltic_sea(subset_T,  readable_date, target_depth, plotsize=8, name_dataset="Baltic Sea"):
    subset_T.plot(size = plotsize)
    map_long_name = subset_T.long_name.replace("_", " ")

    ## Add title
    title = f'{str(name_dataset)} {map_long_name} on {readable_date} at {str(target_depth)} meters'       # set the title
    plt.title(title, fontsize=16)
    return

def depth_profile(DS, readable_date, parameter, target_date, plot_x=10, plot_y=10,longitude=10.039330 , latitude=54.529500, min_longitude=9.5 ,max_longitude=10.5, min_latitude=54, max_latitude=55, depth_min=0.51):
    f = plt.figure(figsize=(plot_x,plot_y))   

    # Define the parameters (variable, gridpoint, depth range, time) 
    #d_thetao = DS[parameter].sel(longitude=longitude, latitude=latitude, method='nearest').isel(depth=slice(0, -2)).sel(time=target_date) #tiefe rausschneiden?
    d_thetao = DS[parameter] \
        .sel(longitude=longitude, latitude=latitude, method='nearest') \
        .where((DS.depth >= depth_min) & (DS.depth <= 32), drop=True) \
        .sel(time=target_date)

    
    #plot the vertical profiles
    d_thetao.plot(y = 'depth', yincrease=False, color='green', size=7)    # define the parameters to plot

    #title

    text_ohne_unterstriche = d_thetao.long_name.replace("_", " ")

    if latitude == 54.5295 and longitude == 10.03933:
        title_location= "Boknis Eck"

    else: 
        title_location= f"lat={latitude} and lon={longitude}"

    plt.xlabel(text_ohne_unterstriche)   # Beschriftung der x-Achse

    plt.title(f"Depth profile of {text_ohne_unterstriche} at {title_location} on {readable_date}", fontsize=15)
    return

def evo_maps(date_begin, date_end, port_T, target_depth):
        # Define the target period of interest (from January to December 2021)
    target_t_period=slice(date_begin, date_end)
    t_evolution = port_T.sel(time=target_t_period).isel(depth=target_depth)

    #plot
    t_evolution.plot(col="time", col_wrap=6, size=4)  
    return
    '''if export_as_csv == "yes":
        os.makedirs(ordner, exist_ok=True)
        DS.to_csv(f"{ordner}/copernicus_export.csv", index=False)


    subset = DS[[parameter_1]].sel(time = slice(date_beginn, date_end))
    
    diese Funktion gibt XXX zurück
    
    subset_T = subset[parameter_1].isel(depth=target_depth).sel(time = target_date, method = 'nearest')             # method='nearest' -> Method to use for inexact matches (use the nearest valid index value)
    
    target_t_period=slice(date_beginn, date_end)

    port_T = DS[parameter_1].sel(longitude = port_lon, latitude = port_lat)
    
    evolution = port_T.sel(time=target_t_period).isel(depth=target_depth)


    subset_2 = DS[[parameter_2]].sel(time = slice(date_beginn, date_end))

    subset_T_2 = subset_2[parameter_2].isel(depth=target_depth).sel(time = target_date, method = 'nearest')             # method='nearest' -> Method to use for inexact matches (use the nearest valid index value)
    
    target_t_period=slice(date_beginn, date_end)

    port_T_2= DS[parameter_2].sel(longitude = port_lon, latitude = port_lat)

    evolution_2 = port_T_2.sel(time=target_t_period).isel(depth=target_depth)

    values_para_1 = evolution.sel(longitude, latitude, method='nearest')
    values_para_2 = evolution_2.sel(longitude, latitude, method='nearest')

    para_1_too_long_name = values_para_1.long_name.replace("_", " ")
    para_2_too_long_name = values_para_2.long_name.replace("_", " ")
    para_1_long_name = para_1_too_long_name.replace(" Concentration", "")
    para_2_long_name=para_2_too_long_name.replace(" Concentration", "")
    fig, ax1 = plt.subplots()
    
    
    '''

def closeup_map(DS, parameter, readable_date, port_lon, port_lat, target_date, target_depth,  longitude=10.039330 , latitude=54.529500, name_region="Boknis Eck"):

    port_T = DS[parameter].sel(longitude = port_lon, latitude = port_lat)
    ## Add coastlines and land feature 
    f = plt.figure(figsize=(18, 7))                                                                      # define the size of the plot
    ax = plt.axes(projection=ccrs.PlateCarree())                                                         # define the projection                                    
    ax.coastlines()                                                                                      # add the coastlines
    ax.add_feature(cfeature.LAND, zorder=1, edgecolor='k')                                               # add continent
    gl = ax.gridlines(draw_labels=True)                                                                  # add gridlines
    gl.left_labels = False                                                                               # remove latitude labels on the right
    gl.top_labels = False     

    ax.scatter(
        longitude, latitude,
        color='red',
        s=100,                 # Punktgröße
        marker='x',
        linewidths= 3,
        transform=ccrs.PlateCarree(),
        zorder=5
    )
    ## Generate plot with colobar
    port_T.sel(time=target_date).isel(depth=target_depth).plot(cmap='bwr')                                                                      # cmap -> colorbar                                                   
    snapmap_long_name = port_T.long_name.replace("_", " ")

    ## Add title
    plt.title(f'{str(name_region)} - {snapmap_long_name} on {readable_date} at {str(target_depth)} meters' , fontsize=15)   # add title to the plot

    return port_T

def cop_load_two_parameters(parameter_1, parameter_2, target_depth,   target_date, DS, date_beginn, date_end, min_longitude=9.5 ,max_longitude=10.5, min_latitude=54, max_latitude=55, longitude=10.039330 , latitude=54.529500):
    
    
    port_lon = slice(min_longitude, max_longitude)     #longitude
    port_lat = slice(min_latitude, max_latitude)    #latitude
    
    #load first parameter
    #if parameter_1 != load_data_of_one_parameter_cop[5] or parameter_2 != load_data_of_one_parameter_cop[6]:
        #warnings.warn("The parameters used do not align with the parameters loaded into the dataset with load_data_of_one_parameter_cop.") 
    subset_para_1 = DS[[parameter_1]].sel(time = slice(date_beginn, date_end))

    subset_para_1_cut_out = subset_para_1[parameter_1].isel(depth=target_depth).sel(time = target_date, method = 'nearest')             # method='nearest' -> Method to use for inexact matches (use the nearest valid index value)

    target_period=slice(date_beginn, date_end)

    port_para_1 = DS[parameter_1].sel(longitude = port_lon, latitude = port_lat)
    
    evolution_para_1 = port_para_1.sel(time=target_period).isel(depth=target_depth)

    '''subset_para_2 = DS[[parameter_2]].sel(time = slice(date_beginn, date_end))

    subset_para_2_cut_out = subset_para_2[parameter_2].isel(depth=target_depth).sel(time = target_date, method = 'nearest')             # method='nearest' -> Method to use for inexact matches (use the nearest valid index value)

    port_para_2= DS[parameter_2].sel(longitude = port_lon, latitude = port_lat)

    evolution_para_2 = port_para_2.sel(time=target_period).isel(depth=target_depth)'''

    values_para_1 = evolution_para_1.sel(longitude= longitude, latitude=latitude, method='nearest')
    values_para_2 = evolution_para_2.sel(longitude=longitude, latitude=latitude, method='nearest')

    para_1_too_long_name = values_para_1.long_name.replace("_", " ")
    para_2_too_long_name = values_para_2.long_name.replace("_", " ")
    para_1_long_name = para_1_too_long_name.replace(" Concentration", "")
    para_2_long_name=para_2_too_long_name.replace(" Concentration", "")

    return values_para_1, values_para_2, para_1_long_name, para_2_long_name, subset_para_1_cut_out, subset_para_2_cut_out


def cop_evolution_of_two_parameters( target_depth, values_para_1, values_para_2, para_1_long_name,para_2_long_name, plot_x_size=10, plot_y_size=10, filename=None, output_folder="Plots Copernicus"):
    f = plt.figure(figsize=(plot_x_size,plot_y_size))     


    fig, ax1 = plt.subplots()

    # linke Achse
    ax1.plot(values_para_1.time, values_para_1, color="tab:blue", label=f"{para_2_long_name}")
    ax1.set_xlabel("Date")
    ax1.set_ylabel(f"{para_1_long_name} " , color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # rechte Achse
    ax2 = ax1.twinx()
    ax2.plot(values_para_2.time, values_para_2, color="tab:red", label=f"{para_2_long_name}")
    ax2.set_ylabel(f"{para_2_long_name} ", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))

    plt.xticks(rotation=45)
    plt.title(f"Evolution of {para_1_long_name} and {para_2_long_name} at Boknis Eck at {target_depth} m")
    plt.show()

    if filename:
        os.makedirs(output_folder, exist_ok=True)
        filepath = os.path.join(output_folder, filename)
        fig.savefig(filepath, bbox_inches="tight", dpi=300)
        print(f"Plot saved as: {filename}")
    
    return 

