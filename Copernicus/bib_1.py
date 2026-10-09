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
x_map = {
    "Chl a": "Chlorophyll a",
    "[NO2]-": "Nitrite",
    "O2": "Oxygen",
    "[PO4]3-": "Phosphate",
    "Sal": "Salinity",
    "SiO2": "Silicon dioxide",
    "Temp":"Water Temperature",
    "[NO3]-": "Nitrate"
}
x_map_units = {
    "Chl a": "[\u00b5g/l]",
    "[NO2]-": "[\u00b5mol/l]",
    "O2": "[\u00b5mol/kg]",
    "[PO4]3-": "[\u00b5mol/l]",
    "Sal": "",
    "SiO2": "[\u00b5mol/l]",
    "Temp":"[\u00b0C]",
    "[NO3]-": "[\u00b5mol/l]"
}
# whenever you want to save a plot as a specific format, you just have to put that format at the end of your filename e.g. pdf --> pangea_plot.pdf
def display_mini_map(longitude_of_exact_location=10.039330, latitude_of_exact_location=54.529500, min_longitude=9.5, max_longitude=10.5, min_latitude=54, max_latitude=55, 
                     fig_size_y=10, fig_size_x= 10, output_folder="minimap", filename=None):
    """
    this function display a mini window of a map between the given min_longitude, max_longitude, min_latitude, max_latitude. It displays a point for the exact location that is being investigated. If no further information is given, it will default to Boknis Eck.
    """
    f = plt.figure(figsize=(fig_size_x,fig_size_y))                                                                           # define plot size
    
    ## Display the locations of the glider on a mini map
    ax_mini_map = f.add_axes([0.5, 1, 0.3, 0.2], projection=ccrs.PlateCarree())                               # create the minimap and define its projection
    gl = ax_mini_map.gridlines(draw_labels=True)                                                              # add the lon/lat gridlines
    gl.right_labels = False                                                                                   # remove latitude labels on the right
    gl.top_labels = False                                                                                     # remove longitude labels on the top

    
    # Customize mini map
    ax_mini_map.add_feature(cfeature.LAND, zorder=0, edgecolor='k')                                           # add land mask 
    ax_mini_map.set_extent([min_longitude, max_longitude, min_latitude, max_latitude], crs = ccrs.PlateCarree())                                      # define the extent of the mini map [lon_min,lon_max,lat_min,lat_max]
    ax_mini_map.scatter(longitude_of_exact_location, latitude_of_exact_location, 25,'tab:blue',transform=ccrs.PlateCarree())             # plot location of point
   
    if filename:        # saves the created map if a filename is given in the output folder chosen
       os.makedirs(output_folder, exist_ok=True)
       filepath = os.path.join(output_folder, filename)
       f.savefig(filepath, bbox_inches="tight", dpi=300) # dpi defines the resolution of the saved diagram. can be changed in this function
       print(f"Plot saved as: {filename}")          # prints the filename so you can find it easily after saving
    
    plt.show()
    return

def pangaea_evolution_one_parameter(parameter, period_begin, period_end, target_depth, name_exact="Boknis Eck", values_min_parameter=None, values_max_parameter=None,filename=None, output_folder= "Plots Pangea", with_zeros="yes", parameter_limits="no", x_length=12, y_length=10, data_ID=855693, lower_depth_margin=0, upper_depth_margin=1):
    '''gives you the evolution for one parameter over a period of time 
    lower_depth_margin means the inserted value will be subtracted from target_depth to include more depth measurements. the upper_depth:margin will be added to the upper end of the target-depth'''
    
    ds = pd.PanDataSet(data_ID) # opens the data set and puts it into "ds"
    df = ds.data 
    # the next few if loops cut out the data for the evolution diagram. it depends on the infos given in the function. you can cut out zeros if they dont make sense appearing. 
    if with_zeros =="yes":
        parameter_timeframe = df.loc[
            (df["Date/Time"].between(period_begin, period_end)) & #cuts out the desired time period
            (df["Depth water"].between(target_depth-lower_depth_margin, target_depth+upper_depth_margin)), # cuts out the wanted depth. in this plot there is only one depth. the margin is given because there are sometimes irregular depths in the data (26 m instead of the usual 25m)
            [parameter, "Depth water"]      
        ]
        
        print("Zeros included.")        #gives the info that there are zeros included
    else:
        parameter_timeframe = df.loc[
            (df["Date/Time"].between(period_begin, period_end)) &
            (df["Depth water"].between(target_depth-lower_depth_margin, target_depth+upper_depth_margin)) &
            (df[parameter] != 0), #cuts out the zeros from any data
            [parameter, "Depth water"]
        ]
        print("Zeros taken out of reasonable values.")      #gives the info that the zeros are taken out of the data
    #puts every wanted value in a list to work with
    if parameter_limits=="yes":
        parameter_list = parameter_timeframe[parameter].where((parameter_timeframe[parameter] >= values_min_parameter) & (parameter_timeframe[parameter] <= values_max_parameter))
        print("Parameter limits are applied.")
    else:
        parameter_list = parameter_timeframe[parameter]
        print("Every value included.")
    
    if parameter_limits=="no" and with_zeros=="yes":
        print("Unfiltered data.")

    parameter_list = parameter_list.tolist()
    #chooses the times to align with the chosen values and edits the time display
    andere_spalte = df.loc[parameter_timeframe.index, "Date/Time"]
    dates_clean = [ts.date() for ts in andere_spalte]
    x = dates_clean
    y = parameter_list
    #plt.figure(figsize=(x_length,y_length))                                                                           # define plot size


    if len(parameter_list)==0: #prints a warning if the parameter list is empty. this corresponds to no data in the chosen timeframe. this might be cue to gaps in the data or it is outside of the time any data was taken
        print(f"\033[1;31mWarning: Empty list. The chosen timeframe has no entries for {x_map[parameter]}. Choose another timeframe and try again\033[0m")

    else:   #otherwise a plot is given out
        fig, ax = plt.subplots(figsize=(x_length, y_length))    #define size of the plot
    
        ax.plot(x, y)   # plots the dates and the parameters

    
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y')) #formats the dates on the x scale

        ax.set_xlabel("Date")   # sets the labek for the x label
        ax.set_ylabel(f"{x_map[parameter]} {x_map_units[parameter]}")    #selts the label for the y scale
        ax.set_title(f"{name_exact} - Evolution of {x_map[parameter]} at a depth of {target_depth} m"   # sets the title for the plot
    )
    if parameter == "Chl a":    # puts a log scale on chlorophyll data
        ax.set_yscale("log")
    plt.show()

    if filename:    #saves the file if filename is given
        os.makedirs(output_folder, exist_ok=True)
        filepath = os.path.join(output_folder, filename)
        fig.savefig(filepath, bbox_inches="tight", dpi=300)
        print(f"Plot saved as: {filename}")
        
    return

    
def pangeae_evolution_two_parameters(parameter, parameter_2,period_begin, period_end, target_depth, name_exact="Boknis Eck", values_min_parameter_1=None, values_max_parameter_1=None,values_min_parameter_2=None, values_max_parameter_2=None, filename=None, output_folder= "Plots Pangea", include_zeros_for_parameter_1="yes", apply_limits_to_parameter_1="no", include_zeros_for_parameter_2="yes", apply_limits_to_parameter_2="no", x_length=12, y_length=10, data_ID=855693, lower_depth_margin=0,upper_depth_margin=1):
    ds = pd.PanDataSet(data_ID)
    df = ds.data
    #from here the data is cut out with the chosen limits/applications
    if include_zeros_for_parameter_1 =="yes":
        parameter_timeframe = df.loc[
            (df["Date/Time"].between(period_begin, period_end)) &
            (df["Depth water"].between(target_depth-lower_depth_margin, target_depth+upper_depth_margin)),
            [parameter, "Depth water"]
        ]
        
        
    else:
        parameter_timeframe = df.loc[
            (df["Date/Time"].between(period_begin, period_end)) &
            (df["Depth water"].between(target_depth-lower_depth_margin, target_depth+upper_depth_margin)) &
            (df[parameter] != 0), 
            [parameter, "Depth water"]
        ]
        print(f"Zeros taken out of reasonable values for {parameter}.")
    #puts every wanted value in a list to work with\n",
    if apply_limits_to_parameter_1=="yes":
        parameter_list = parameter_timeframe[parameter].where((parameter_timeframe[parameter] >= values_min_parameter_1) & (parameter_timeframe[parameter] <= values_max_parameter_1))
        print(f"Parameter limits are applied for {parameter}. Only values betweeen {values_min_parameter_1} and {values_max_parameter_1} used.")
    else:
        parameter_list = parameter_timeframe[parameter]
        print(f"No parameter limits applied for {parameter}.")
    
    if apply_limits_to_parameter_1 !="yes" and include_zeros_for_parameter_1=="yes":
        print(f"Unfiltered data for {parameter}.")
    
    if apply_limits_to_parameter_1== "yes" and include_zeros_for_parameter_1 == "yes":
        print(f"Zeros included for {parameter}.")
    
    if apply_limits_to_parameter_1!="yes" and include_zeros_for_parameter_1!="yes":
        print(f"No parameter limits applied for {parameter}.")

    parameter_list = parameter_list.tolist()
    #chooses the times to align with the chosen values and edits the time display
    andere_spalte = df.loc[parameter_timeframe.index, "Date/Time"]
    dates_clean = [ts.date() for ts in andere_spalte]



    #cut out relevant data for secondparameter
    if include_zeros_for_parameter_2 =="yes":
        parameter_2_timeframe = df.loc[
            (df["Date/Time"].between(period_begin, period_end)) &
            (df["Depth water"].between(target_depth-lower_depth_margin, target_depth+upper_depth_margin)),
            [parameter_2, "Depth water"]
        ]
        
        print(f"Zeros included for {parameter_2}.")
    else:
        parameter_2_timeframe = df.loc[
            (df["Date/Time"].between(period_begin, period_end)) &
            (df["Depth water"].between(target_depth-lower_depth_margin, target_depth+upper_depth_margin)) &
            (df[parameter_2] != 0), 
            [parameter_2, "Depth water"]
        ]
        print(f"Zeros taken out of reasonable values for {parameter_2}.")
    
    #puts every wanted value in a list to work with
    if apply_limits_to_parameter_2 == "yes":
        parameter_2_list = parameter_2_timeframe[parameter_2].where((parameter_2_timeframe[parameter_2] >= values_min_parameter_2) & (parameter_2_timeframe[parameter_2] <= values_max_parameter_2))
        print(f"Parameter limits are applied for {parameter_2}. Only values betweeen {values_min_parameter_2} and {values_max_parameter_2} used.")
    
    else:
        parameter_2_list = parameter_2_timeframe[parameter_2]
        #print(f"Every value included for {parameter_2}.")
    parameter_2_list = parameter_2_list.tolist()
    
    if apply_limits_to_parameter_2 !="yes" and include_zeros_for_parameter_2=="yes":
        print(f"Unfiltered data for {parameter_2}.")
    
    if apply_limits_to_parameter_2== "yes" and include_zeros_for_parameter_2 == "yes":
        print(f"Zeros included for {parameter_2}.")
    
    if apply_limits_to_parameter_2!="yes" and include_zeros_for_parameter_2!="yes":
        print(f"No parameter limits applied for {parameter_2}.")


    #chooses the times to align with the chosen values and edits the time display
    andere_spalte = df.loc[parameter_2_timeframe.index, "Date/Time"]

    dates_clean_2 = [ts.date() for ts in andere_spalte]

    #display two different parameters in one graph

    fig, ax1 = plt.subplots(figsize=(x_length, y_length))
    x_1=dates_clean
    x_2 = dates_clean_2
    y = parameter_list
    y_2=parameter_2_list

    # left axis
    line_1=ax1.plot(x_1,y, color="tab:blue", label=f"{x_map[parameter]} {x_map_units[parameter]}")

    ax1.set_xlabel("Date")
    ax1.set_ylabel(f"{x_map[parameter]} {x_map_units[parameter]}" , color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    if parameter == "Chl a":
        ax1.set_yscale("log")
    # right axis
    ax2 = ax1.twinx()
    line_2=ax2.plot(x_2,y_2, color="tab:red", label= f"{x_map[parameter_2]} {x_map_units[parameter_2]}‚")
    ax2.set_ylabel(f"{x_map[parameter_2]} {x_map_units[parameter_2]}", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    
    if parameter_2 == "Chl a":
        ax2.set_yscale("log")
    plt.xticks(rotation=45)
    plt.title(f"Evolution of {x_map[parameter]} and {x_map[parameter_2]} at {name_exact} at {target_depth} m")
    
    #define and manipulate the legend
    lines = line_1 + line_2
    labels = [l.get_label() for l in lines]

    ax1.legend(lines, labels,
            loc='upper left',
            edgecolor='black',
            framealpha=1)
    plt.show()

    if filename:
        os.makedirs(output_folder, exist_ok=True)
        filepath = os.path.join(output_folder, filename)
        fig.savefig(filepath, bbox_inches="tight", dpi=300)
        print(f"Plot saved as: {filename}")
    return

def pangaea_depth_profile_one_parameter(parameter, target_month,  name_exact="Boknis Eck", values_min_parameter=None, values_max_parameter=None,filename=None, output_folder= "Plots Pangea", with_zeros="yes", parameter_limits="no", x_length=12, y_length=10, data_ID=855693):
    ds = pd.PanDataSet(data_ID)
    df = ds.data

    #help find the right date (there is mostly one date per month. this helps identify the closest date to the one given in the function)
    target_month_start= target_month + "-01"        #the first day of the month is added to the given date

    next_date=df.loc[df["Date/Time"] > target_month_start, "Date/Time"].min()   #the next biggest date in the data is chosen to be displayed
    
    
    #cut out the relevant data for depth profile
    if parameter_limits == "yes" and with_zeros == "yes":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date, next_date)) &
            (df["Depth water"].between(1, 27)) &
            (df[parameter].between(values_min_parameter, values_max_parameter)) &
            (df[parameter].notna()),
            [parameter, "Depth water", "Date/Time"]
        ]
    elif parameter_limits == "no" and with_zeros == "yes":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date, next_date)) &
            (df["Depth water"].between(1, 27)) &
            (df[parameter].notna()),
            [parameter, "Depth water", "Date/Time"]
        ]
    elif parameter_limits == "yes" and with_zeros == "no":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date, next_date)) &
            (df["Depth water"].between(1, 27)) &
            (df[parameter].between(values_min_parameter, values_max_parameter)) &
            (df[parameter] != 0) &
            (df[parameter].notna()),
            [parameter, "Depth water", "Date/Time"]
        ]
    elif parameter_limits == "no" and with_zeros == "no":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date, next_date)) &
            (df["Depth water"].between(1, 27)) &
            (df[parameter] != 0) &
            (df[parameter].notna()),
            [parameter, "Depth water", "Date/Time"]
        ]
    

   
    para_list_depth = para_timeframe_depth[parameter].dropna().tolist()
    target_date = ppd.Timestamp(target_month_start)
    if (target_date.year, target_date.month) != (next_date.year, next_date.month):
        print(f"\033[1;31mWarning: Empty list. The chosen timeframe has no entries for {x_map[parameter]}. No data was taken in this month. The closest following date has been automatically chosen as {next_date.year}.{next_date.month}.{next_date.day}.\033[0m. ")

    depth_values = df.loc[para_timeframe_depth.index, "Depth water"]
    date_list=para_timeframe_depth["Date/Time"].dropna().tolist()
    #make the date readable
    ts= date_list[1]
    
    date=ts.strftime("%Y-%m-%d")
    #plots the plot
    plt.figure(figsize=(x_length, y_length))
    plt.plot(para_list_depth, depth_values, marker="o")
    plt.xlabel(f" {x_map[parameter]} {x_map_units[parameter]}")
    plt.ylabel("Depth in m")
    plt.title(f"Depth profile of {x_map[parameter]} at {name_exact} on {date}")
    plt.gca().invert_yaxis()  # depth down the y axis
    plt.grid(True)
    
    if filename:
        os.makedirs(output_folder, exist_ok=True)
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        print(f"Plot saved as: {filename}")
    plt.show()
    return 

def pangaea_depth_profile_two_parameter(parameter_1, parameter_2, target_month,  name_exact="Boknis Eck", values_min_parameter_1=None, values_max_parameter_1=None,values_min_parameter_2=None, values_max_parameter_2=None,filename=None, output_folder= "Plots Pangea", include_zeros_for_parameter_1="yes", apply_limits_to_parameter_1="no", include_zeros_for_parameter_2="yes", apply_limits_to_parameter_2="no", x_length=12, y_length=10, data_ID=855693):
    ds = pd.PanDataSet(data_ID)
    df = ds.data
    #cut out the relevant data for depth profile
    #addd 01 to the given month
    target_month_start= target_month + "-01"
    #apply to both parameters
    mask = (
        (df["Date/Time"] > target_month_start)
        & df[parameter_1].notna()
        & df[parameter_2].notna()
    )
    #only use. if both oarameters have entries for the given date
    next_date = df.loc[mask, "Date/Time"].min()
    if apply_limits_to_parameter_1 == "yes" and include_zeros_for_parameter_1 == "yes":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date, next_date)) &
            (df["Depth water"].between(1, 27)) &
            (df[parameter_1].between(values_min_parameter_1, values_max_parameter_1)) &
            #(df[parameter_1] != 0) &
            (df[parameter_1].notna()),
            [parameter_1, "Depth water", "Date/Time"]
        ]
    elif apply_limits_to_parameter_1 == "no" and include_zeros_for_parameter_1 == "yes":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date, next_date)) &
            (df["Depth water"].between(1, 27)) &
            (df[parameter_1].notna()),
            [parameter_1, "Depth water", "Date/Time"]
        ]
    elif apply_limits_to_parameter_1 == "yes" and include_zeros_for_parameter_1 == "no":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date, next_date)) &
            (df["Depth water"].between(1, 27)) &
            (df[parameter_1].between(values_min_parameter_1, values_max_parameter_1)) &
            (df[parameter_1] != 0) &
            (df[parameter_1].notna()),
            [parameter_1, "Depth water", "Date/Time"]
        ]
    elif apply_limits_to_parameter_1 == "no" and include_zeros_for_parameter_1== "no":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date, next_date)) &
            (df["Depth water"].between(1, 27)) &
            (df[parameter_1] != 0) &
            (df[parameter_1].notna()),
            [parameter_1, "Depth water", "Date/Time"]
        ]
    para_list_depth = para_timeframe_depth[parameter_1].dropna().tolist()
    depth_values = df.loc[para_timeframe_depth.index, "Depth water"]

    if apply_limits_to_parameter_2 == "yes" and include_zeros_for_parameter_2 == "yes":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date, next_date)) &
            (df["Depth water"].between(1, 27)) &
            (df[parameter_2].between(values_min_parameter_2, values_max_parameter_2)) &
            (df[parameter_2].notna()),
            [parameter_2, "Depth water", "Date/Time"]
        ]
    elif apply_limits_to_parameter_2 == "no" and include_zeros_for_parameter_2 == "yes":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date, next_date)) &
            (df["Depth water"].between(1, 27)) &
            (df[parameter_2].notna()),
            [parameter_2, "Depth water", "Date/Time"]
        ]
    elif apply_limits_to_parameter_2 == "yes" and include_zeros_for_parameter_2 == "no":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date, next_date)) &
            (df["Depth water"].between(1, 27)) &
            (df[parameter_2].between(values_min_parameter_2, values_max_parameter_2)) &
            (df[parameter_2] != 0) &
            (df[parameter_2].notna()),
            [parameter_2, "Depth water", "Date/Time"]
        ]
    elif apply_limits_to_parameter_2 == "no" and include_zeros_for_parameter_2 == "no":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date, next_date)) &
            (df["Depth water"].between(1, 27)) &
            (df[parameter_2] != 0) &
            (df[parameter_2].notna()),
            [parameter_2, "Depth water", "Date/Time"]
        ]
    para_list_depth_2 = para_timeframe_depth[parameter_2].dropna().tolist()
    depth_values_2 = df.loc[para_timeframe_depth.index, "Depth water"]

    target_date = ppd.Timestamp(target_month_start)
    #gives out warning if not both paras are present for the timeframe
    if (target_date.year, target_date.month) != (next_date.year, next_date.month):
        print(f"\033[1;31mWarning: Empty list. The chosen timeframe has no entries for {x_map[parameter_1]} and {x_map[parameter_2]}. No data was taken in this month. The closest following date has been automatically chosen as {next_date.year}.{next_date.month}.{next_date.day}.\033[0m ")
    date_list=para_timeframe_depth["Date/Time"].dropna().tolist()
    fig, ax1 = plt.subplots(figsize=(x_length,y_length))
    ts= date_list[1]
    #change date format
    date=ts.strftime("%Y-%m-%d")
    # left axis
    line_1=ax1.plot(para_list_depth,depth_values ,color="tab:blue", label=f"{parameter_1} {x_map_units[parameter_1]}")
    ax1.set_ylabel("Depth", color="black")
    ax1.set_xlabel(f"{parameter_1} {x_map_units[parameter_1]}" , color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.tick_params(axis="x", labelcolor="tab:blue")

    # right axis
    ax2 = ax1.twiny()
    line_2= ax2.plot(para_list_depth_2, depth_values_2, color="tab:red", label=f"{x_map[parameter_2]} {x_map_units[parameter_2]}")
    ax2.set_xlabel(f"{parameter_2} {x_map_units[parameter_2]}", color="tab:red")
    ax2.tick_params(axis="x", labelcolor="tab:red")

    plt.xticks(rotation=45)
    plt.title(f"Depth profile of {parameter_1} and {parameter_2} at Boknis Eck on {date}")

    

    
    # manage legends
    lines = line_1 + line_2
    labels = [l.get_label() for l in lines]

    ax1.legend(lines, labels,
            loc='upper left',
            edgecolor='black',
            framealpha=1)

    plt.gca().invert_yaxis()  # depth down the y axis
    ax1.grid(True)

    plt.show()
    if filename:
        os.makedirs(output_folder, exist_ok=True)
        filepath = os.path.join(output_folder, filename)
        fig.savefig(filepath, bbox_inches="tight", dpi=300)
        print(f"Plot saved as: {filename}")

    return


def pangaea_depth_profile_two_months(parameter_1, target_month_1, target_month_2, name_exact="Boknis Eck", values_min_parameter_1=None, values_max_parameter_1=None,filename=None, output_folder= "Plots Pangea", include_zeros_for_parameter_1="yes", apply_limits_to_parameter_1="no",  x_length=12, y_length=10, data_ID=855693):
    ds = pd.PanDataSet(data_ID)
    df = ds.data
    #cut out the relevant data for depth profile
    target_month_1_start= target_month_1 + "-01"
    
    #choose the right date
    next_date_1=df.loc[df["Date/Time"] > target_month_1_start, "Date/Time"].min()
    
    target_month_2_start= target_month_2 + "-01"
    

    next_date_2=df.loc[df["Date/Time"] > target_month_2_start, "Date/Time"].min()
    
   


    if apply_limits_to_parameter_1 == "yes" and include_zeros_for_parameter_1 == "yes":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date_1, next_date_1)) &
            (df["Depth water"].between(1, 27)) &
            (df[parameter_1].between(values_min_parameter_1, values_max_parameter_1)) &
            (df[parameter_1].notna()),
            [parameter_1, "Depth water", "Date/Time"]
        ]
    elif apply_limits_to_parameter_1 == "no" and include_zeros_for_parameter_1 == "yes":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date_1, next_date_1)) &
            (df["Depth water"].between(1, 27)) &
            (df[parameter_1].notna()),
            [parameter_1, "Depth water", "Date/Time"]
        ]
    elif apply_limits_to_parameter_1 == "yes" and include_zeros_for_parameter_1 == "no":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date_1, next_date_1)) &
            (df["Depth water"].between(1, 27)) &
            (df[parameter_1].between(values_min_parameter_1, values_max_parameter_1)) &
            (df[parameter_1] != 0) &
            (df[parameter_1].notna()),
            [parameter_1, "Depth water", "Date/Time"]
        ]
    elif apply_limits_to_parameter_1 == "no" and include_zeros_for_parameter_1== "no":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date_1, next_date_1)) &
            (df["Depth water"].between(1, 27)) &
            (df[parameter_1] != 0) &
            (df[parameter_1].notna()),
            [parameter_1, "Depth water", "Date/Time"]
        ]
    para_list_depth = para_timeframe_depth[parameter_1].dropna().tolist()
    depth_values = df.loc[para_timeframe_depth.index, "Depth water"]
    date_list=para_timeframe_depth["Date/Time"].dropna().tolist()
    
    # #makes date readeable
    ts_1= date_list[1]
    date_1=ts_1.strftime("%Y-%m-%d")


    if apply_limits_to_parameter_1 == "yes" and include_zeros_for_parameter_1 == "yes":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date_2, next_date_2)) &
            (df["Depth water"].between(1, 27)) &
            (df[parameter_1].between(values_min_parameter_1, values_max_parameter_1)) &
            #(df[parameter_2] != 0) &
            (df[parameter_1].notna()),
            [parameter_1, "Depth water", "Date/Time"]
        ]
    elif apply_limits_to_parameter_1 == "no" and include_zeros_for_parameter_1 == "yes":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date_2, next_date_2)) &
            (df["Depth water"].between(1, 27)) &
            #(df[parameter].between(values_min_parameter_2, values_max_parameter_2)) &
            #(df[parameter] != 0) &
            (df[parameter_1].notna()),
            [parameter_1, "Depth water", "Date/Time"]
        ]
    elif apply_limits_to_parameter_1 == "yes" and include_zeros_for_parameter_1 == "no":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date_2, next_date_2)) &
            (df["Depth water"].between(1, 27)) &
            (df[parameter_1].between(values_min_parameter_1, values_max_parameter_1)) &
            (df[parameter_1] != 0) &
            (df[parameter_1].notna()),
            [parameter_1, "Depth water", "Date/Time"]
        ]
    elif apply_limits_to_parameter_1 == "no" and include_zeros_for_parameter_1 == "no":
        para_timeframe_depth = df.loc[
            (df["Date/Time"].between(next_date_2, next_date_2)) &
            (df["Depth water"].between(1, 27)) &
            #(df[parameter].between(values_min_parameter_2, values_max_parameter_2)) &
            (df[parameter_1] != 0) &
            (df[parameter_1].notna()),
            [parameter_1, "Depth water", "Date/Time"]
        ]
    para_list_depth_2 = para_timeframe_depth[parameter_1].dropna().tolist()
    depth_values_2 = df.loc[para_timeframe_depth.index, "Depth water"]

   
    target_date_1 = ppd.Timestamp(target_month_1_start)
    target_date_2 = ppd.Timestamp(target_month_2_start)
    
    if (target_date_1.year, target_date_1.month) != (next_date_1.year, next_date_1.month):
        print(f"\033[1;31mWarning: Empty list. The month {target_month_1} has no entries for {x_map[parameter_1]}. No data was taken in this month. The closest following date has been automatically chosen as {next_date_1.year}.{next_date_1.month}.{next_date_1.day}.\033[0m. ")

    if (target_date_2.year, target_date_2.month) != (next_date_2.year, next_date_2.month):
        print(f"\033[1;31mWarning: Empty list. The month {target_month_2} has no entries for {x_map[parameter_1]}. No data was taken in this month. The closest following date has been automatically chosen as {next_date_2.year}.{next_date_2.month}.{next_date_2.day}.\033[0m. ")

    
    date_list=para_timeframe_depth["Date/Time"].dropna().tolist()
    #make_the second date readable
    ts_2= date_list[1]
    date_2=ts_2.strftime("%Y-%m-%d")

    #plot
    fig, ax1 = plt.subplots(figsize=(x_length,y_length))

    # left axis
    ax1.plot(para_list_depth,depth_values ,color="tab:blue", label=f"{parameter_1} on {date_1}")
    ax1.set_ylabel("Depth", color="black")
    ax1.set_xlabel(f"{parameter_1} {x_map_units[parameter_1]}" , color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.tick_params(axis="x", labelcolor="black")

    # right qxis
    ax1.plot(para_list_depth_2, depth_values_2, color="tab:red", label=f"{parameter_1} on {date_2}")
    
    plt.xticks(rotation=45)
    plt.title(f"Depth profile of {parameter_1} at {name_exact} on {date_1} and {date_2}")
    plt.legend(frameon=True,
    edgecolor='black')


    plt.gca().invert_yaxis()  # Tiefe nach unten
    ax1.grid(True)

    plt.show()
    if filename:
        os.makedirs(output_folder, exist_ok=True)
        filepath = os.path.join(output_folder, filename)
        fig.savefig(filepath, bbox_inches="tight", dpi=300)
        print(f"Plot saved as: {filename}")
    
    return

def pangaea_two_parameters_scatter_plot(parameter, parameter_2,period_start, period_end, target_depth, name_exact="Boknis Eck", values_min_parameter_1=None, values_max_parameter_1=None,values_min_parameter_2=None, values_max_parameter_2=None, filename=None, output_folder= "Plots Pangea", include_zeros_for_parameter_1="yes", apply_limits_to_parameter_1="no", include_zeros_for_parameter_2="yes", apply_limits_to_parameter_2="no", x_length=12, y_length=10, data_ID=855693):
    
    ds = pd.PanDataSet(data_ID)
    df = ds.data
    #choose date
    year_start, month_start, day_start = period_start.split("-")
    year_end, month_end, day_end = period_end.split("-")

    if include_zeros_for_parameter_1 =="yes":
        parameter_timeframe = df.loc[
            (df["Date/Time"].between(period_start, period_end)) &
            (df["Depth water"].between(target_depth, target_depth+1)),
            #(df[parameter].between(values_min, values_max)) &
            #(df[parameter] != 0), 
            [parameter, "Depth water"]
        ]
        
        #print("Zeros included.")
    else:
        parameter_timeframe = df.loc[
            (df["Date/Time"].between(period_start, period_end)) &
            (df["Depth water"].between(target_depth, target_depth+1)) &
            #(df[parameter].between(values_min, values_max)) &
            (df[parameter] != 0), 
            [parameter, "Depth water"]
        ]
        print("Zeros taken out of reasonable values.")
    #puts every wanted value in a list to work with\n",
    if apply_limits_to_parameter_1=="yes":
        parameter_list = parameter_timeframe[parameter].where((parameter_timeframe[parameter] >= values_min_parameter_1) & (parameter_timeframe[parameter] <= values_max_parameter_1))
        print("Parameter limits are applied.")
    else:
        parameter_list = parameter_timeframe[parameter]
        #print("Every value included.")
    
    if apply_limits_to_parameter_1 !="yes" and include_zeros_for_parameter_1=="yes":
        print(f"Unfiltered data for {parameter}.")
    
    if apply_limits_to_parameter_1== "yes" and include_zeros_for_parameter_1 == "yes":
        print(f"Zeros included for {parameter}.")
    
    if apply_limits_to_parameter_1!="yes" and include_zeros_for_parameter_1!="yes":
        print(f"No parameter limits applied for {parameter}.")

    parameter_list = parameter_list.tolist()
    #chooses the times to align with the chosen values and edits the time display
    andere_spalte = df.loc[parameter_timeframe.index, "Date/Time"]
    


    #cut out relevant data for secondparameter
    if include_zeros_for_parameter_2 =="yes":
        parameter_2_timeframe = df.loc[
            (df["Date/Time"].between(period_start, period_end)) &
            (df["Depth water"].between(target_depth, target_depth+1)),
            #(df[parameter].between(values_min, values_max)) &
            #(df[parameter] != 0), 
            [parameter_2, "Depth water"]
        ]
        
        #print("Zeros included.")
    else:
        parameter_2_timeframe = df.loc[
            (df["Date/Time"].between(period_start, period_end)) &
            (df["Depth water"].between(target_depth, target_depth+1)) &
            #(df[parameter].between(values_min, values_max)) &
            (df[parameter_2] != 0), 
            [parameter_2, "Depth water"]
        ]
   
    #puts every wanted value in a list to work with
    if apply_limits_to_parameter_2 == "yes":
        parameter_2_list = parameter_2_timeframe[parameter_2].where((parameter_2_timeframe[parameter_2] >= values_min_parameter_2) & (parameter_2_timeframe[parameter_2] <= values_max_parameter_2))
    else:
        parameter_2_list = parameter_2_timeframe[parameter_2]
    parameter_2_list = parameter_2_list.tolist()

    if apply_limits_to_parameter_2 !="yes" and include_zeros_for_parameter_2=="yes":
        print(f"Unfiltered data for {parameter_2}.")
    
    if apply_limits_to_parameter_2== "yes" and include_zeros_for_parameter_2 == "yes":
        print(f"Zeros included for {parameter_2}.")
    
    if apply_limits_to_parameter_2!="yes" and include_zeros_for_parameter_2!="yes":
        print(f"No parameter limits applied for {parameter_2}.")
    y = parameter_list
    y_2=parameter_2_list
    #This is the linear regression
    #we set np.nan for every not accounted value
    parameter_list=np.array(parameter_list)
    parameter_2_list=np.array(parameter_2_list)
    
    #cut out the 0.01 as this is equivalent to no reliable data taken
    '''parameter_list = [
        np.nan if x in [np.array([0.01])] else x
        for x in parameter_list
    ]
    parameter_2_list = [
        np.nan if x in [np.array([0.01])] else x
        for x in parameter_2_list
    ]'''
    mask_without_nans = ~np.isnan(parameter_list) & ~np.isnan(parameter_2_list)
    result_lin_reg= linregress(parameter_list[mask_without_nans], parameter_2_list[mask_without_nans])
    g=result_lin_reg.slope*parameter_list+result_lin_reg.intercept
    
    #plot
    fig, ax = plt.subplots(figsize=(x_length,y_length))
    ax.scatter(y, y_2, label= f"{parameter} vs {parameter_2}")
    ax.plot(parameter_list, g, label= f"Linear Regression with p value= {result_lin_reg.pvalue} \n and standard error {result_lin_reg.stderr} ")
    ax.set_xlabel(f"{x_map[parameter]} in {x_map_units[parameter]}")
    ax.set_ylabel(f"{x_map[parameter_2]} in {x_map_units[parameter_2]}")
    ax.set_title(f"{x_map[parameter]} vs {x_map[parameter_2]} at {name_exact} \n from {month_start}.{year_start} to {month_end}.{year_end}")
    plt.legend(frameon=True,
    edgecolor='black')
    plt.show()
    
    if filename:
        os.makedirs(output_folder, exist_ok=True)
        filepath = os.path.join(output_folder, filename)
        fig.savefig(filepath, bbox_inches="tight", dpi=300)
        print(f"Plot saved as: {filename}")
    
    return parameter_list, parameter_2_list,{result_lin_reg}

