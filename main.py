import matplotlib.pyplot as plt
import pandas as pd
import requests
import matplotlib
from matplotlib.gridspec import GridSpec
from requests.auth import HTTPBasicAuth
import PySimpleGUI as Sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import seaborn as sns
from matplotlib import animation

# Close all existing figures
plt.close('all')
# Display all columns when printing the DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

matplotlib.use("TkAgg")

Sg.theme('DarkTeal12')

lwd_model_names = ['LWD_DPD_air', 'LWD_RH_thresh', 'LWD_RH_extended']
dates = ['3-15', '4-1', '4-15', '5-1', '5-15', '6-1', '6-15']

sensors = [316, 383, 442, 453, 458, 463, 467, 468, 473, 474, 481, 488, 492, 500, 514, 516, 522, 529]
sensor_lookup_data = {
    'sensor_id': [383, 500, 316, 463, 467, 473, 442, 514, 453, 474, 458, 492, 481, 516, 488, 529, 522, 468],
    'device': [160, 274, 105, 237, 241, 247, 195, 288, 227, 248, 232, 266, 255, 290, 262, 304, 297, 242],
    'x_loc': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
    'y_loc': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    }
sensor_lookup = pd.DataFrame(sensor_lookup_data)
sensor_lookup = sensor_lookup.set_index('sensor_id')


limit = 145152
sensor = 468
start = "4-02"
end = "5-01"
# limit for 1 month 149152


def get_data(sensor_id, start_date=start, end_date=end, item_limit=145152):
    """
    Retrieves hourly averaged temperature and relative humidity data for a specific sensor in a specific date range

    :param sensor_id:   sensor identification number
    :param start_date:  string of start date in form mm-dd
    :param end_date:    string of end date in form mm-dd
    :param item_limit:  limit of number of values to return
    :return:
    :rtype dataframe:
    """
    base_url = 'https://insight.quantified.eu/api/'

    # Authenticate and obtain a token
    response = requests.post('https://insight.quantified.eu/api/token/login/',
                             auth=HTTPBasicAuth('ACT_project', 'yNuTxRraAe#Heq!af9CG9J*&E'))
    print(response.status_code)
    token = response.json()['token']

    temp = requests.get(base_url +
                        "temperature_events/" +
                        "?device_id=" + str(sensor_id) +
                        "&gateway_receive_time_after=2023-" + start_date +
                        "T17%3A00%3A00Z" +
                        "&gateway_receive_time_before=2023-" + end_date +
                        "T01%3A00%3A00Z" +
                        "&limit=" + str(item_limit),
                        headers={'Authorization': 'Bearer ' + token})
    # Fetch relative humidity events data for device ID 458
    hum = requests.get(base_url +
                       "relative_humidity_events/" +
                       "?device_id=" + str(sensor_id) +
                       "&gateway_receive_time_after=2023-" + start_date +
                       "T17%3A00%3A00Z" +
                       "&gateway_receive_time_before=2023-" + end_date +
                       "T01%3A00%3A00Z" +
                       "&limit=" + str(item_limit),
                       headers={'Authorization': 'Bearer ' + token})

    print(temp.status_code)
    print(hum.status_code)

    # Convert the responses JSON to a DataFrame
    temp_df = pd.DataFrame.from_dict(temp.json()['results'])
    hum_df = pd.DataFrame.from_dict(hum.json()['results'])
    temp_df = temp_df.rename(columns={'value': 'Temperature'})
    hum_df = hum_df.rename(columns={'value': 'RH'})
    df = temp_df
    df['RH'] = hum_df['RH']

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('datetime', inplace=True)
    df_hour = pd.DataFrame(
        {'Temperature': df['Temperature'].resample('1H').mean().values, 'RH': df['RH'].resample('1H').mean()})
    return df_hour


def leaf_wetness(df_hour, wetness_onset=3.5, wetness_dry_off=4.0, rh_thresh=80,
                 rh_thresh_ex_high=87, rh_thresh_ex_low=70, rh_thresh_ex_rate_wet=6,
                 rh_thresh_ex_rate_dry=-4, lwd_threshold_modifier=1):
    """
    Calculates leaf wetness through 3 separate mechanistic models using dew point depression of the air, relative
    humidity thresholds, and extended relative humidity thresholds

    :param df_hour:                 data frame with hourly data for one sensor with temperature, humidity, and datetime
    :param wetness_onset:           dew point depression threshold for wetness onset
    :param wetness_dry_off:         dew point depression threshold for wetness dry off
    :param rh_thresh:               relative humidity threshold for wetness onset
    :param rh_thresh_ex_high:       relative humidity extended threshold for wetness onset
    :param rh_thresh_ex_low:        relative humidity extended threshold for wetness dry off
    :param rh_thresh_ex_rate_wet:   relative humidity extended rate for wetness onset
    :param rh_thresh_ex_rate_dry:   relative humidity extended rate for wetness dry off
    :param lwd_threshold_modifier:
    :return:
    :rtype dataframe:
    """
    wetness_onset *= lwd_threshold_modifier
    wetness_dry_off *= lwd_threshold_modifier
    rh_thresh /= lwd_threshold_modifier
    rh_thresh_ex_high /= lwd_threshold_modifier
    rh_thresh_ex_low /= lwd_threshold_modifier
    rh_thresh_ex_rate_wet /= lwd_threshold_modifier
    rh_thresh_ex_rate_dry *= lwd_threshold_modifier

    # Dew point depression relative to canopy temperature - Not possible no canopy temp
    # __________________________________
    # Dew point depression of the air
    # __________________________________

    df_hour['DPD_air'] = ((100 - df_hour['RH']) / 5)

    # Calculating leaf wetness with dew point depression air method outlined in the paper

    df_hour['leaf_wetness_DPD_air'] = None
    df_hour['LWD_DPD_air'] = None
    prev_wet = 0
    prev_lwd = 0

    for index in df_hour.index:
        dpd_air = df_hour['DPD_air'][index]

        if dpd_air <= wetness_onset:
            df_hour.at[index, 'leaf_wetness_DPD_air'] = 1
        elif dpd_air >= wetness_dry_off:
            df_hour.at[index, 'leaf_wetness_DPD_air'] = 0
        else:
            df_hour.at[index, 'leaf_wetness_DPD_air'] = prev_wet

        prev_wet = df_hour['leaf_wetness_DPD_air'][index]
        if prev_wet:
            df_hour.at[index, 'LWD_DPD_air'] = prev_lwd + 1
        else:
            df_hour.at[index, 'LWD_DPD_air'] = 0

        prev_lwd = df_hour['LWD_DPD_air'][index]

    # Constant relative humidity threshold
    # __________________________________
    df_hour['leaf_wetness_RH_thresh'] = None
    df_hour['LWD_RH_thresh'] = None
    prev_lwd = 0

    for index in df_hour.index:
        rh = df_hour['RH'][index]
        if rh >= rh_thresh:
            df_hour.at[index, 'leaf_wetness_RH_thresh'] = 1
            prev_lwd += 1
        else:
            df_hour.at[index, 'leaf_wetness_RH_thresh'] = 0
            prev_lwd = 0
        df_hour.at[index, 'LWD_RH_thresh'] = prev_lwd

    # Extended relative humidity threshold
    # __________________________________
    df_hour['leaf_wetness_RH_extended'] = None
    df_hour['LWD_RH_extended'] = None

    prev_wet = 0
    prev_lwd = 0
    prev_humidity = 0

    for index in df_hour.index:
        rh = df_hour['RH'][index]
        rh_rate = rh - prev_humidity

        if rh <= rh_thresh_ex_low:
            df_hour.at[index, 'leaf_wetness_RH_extended'] = 0
        elif rh >= rh_thresh_ex_high:
            df_hour.at[index, 'leaf_wetness_RH_extended'] = 1
        else:
            if rh_rate >= rh_thresh_ex_rate_wet:
                df_hour.at[index, 'leaf_wetness_RH_extended'] = 1
            elif rh_rate <= rh_thresh_ex_rate_dry:
                df_hour.at[index, 'leaf_wetness_RH_extended'] = 0
            else:
                df_hour.at[index, 'leaf_wetness_RH_extended'] = prev_wet

        prev_wet = df_hour['leaf_wetness_RH_extended'][index]

        if prev_wet:
            df_hour.at[index, 'LWD_RH_extended'] = prev_lwd + 1
        else:
            df_hour.at[index, 'LWD_RH_extended'] = 0

        prev_lwd = df_hour['LWD_RH_extended'][index]

    return df_hour


def disease_prediction(df_hour, dm_rh_thresh=80, dm_temp_thresh_low=15, dm_temp_thresh_high=20, dm_lwd_thresh=2,
                       bot_rh_thresh=93, bot_temp_thresh_low=9, bot_temp_thresh_high=21, bot_lwd_thresh=7,
                       pm_rh_thresh=90, pm_temp_thresh_low=5, pm_temp_thresh_high=35, pm_lwd_thresh=0,
                       dm_threshold_modifier=1, bot_threshold_modifier=1, pm_threshold_modifier=1,
                       lwd_model='LWD_RH_extended'):
    """
    Calculates where disease conditions for downy mildew, botrytis, and powdery mildew are satisfied given temperature,
    relative humidity, and leaf wetness duration data

    :param df_hour:                 data frame with hourly data for one sensor with temperature, humidity, leaf wetness,
                                    leaf wetness duration, and datetime
    :param dm_rh_thresh:            downy mildew low-end relative humidity threshold
    :param dm_temp_thresh_low:      downy mildew low-end temperature threshold
    :param dm_temp_thresh_high:     downy mildew high-end temperature threshold
    :param dm_lwd_thresh:           downy mildew low-end leaf wetness duration threshold
    :param bot_rh_thresh:           botrytis low-end relative humidity threshold
    :param bot_temp_thresh_low:     botrytis low-end temperature threshold
    :param bot_temp_thresh_high:    botrytis high-end temperature threshold
    :param bot_lwd_thresh:          botrytis low-end leaf wetness duration threshold
    :param pm_rh_thresh:            powdery mildew low-end relative humidity threshold
    :param pm_temp_thresh_low:      powdery mildew low-end temperature threshold
    :param pm_temp_thresh_high:     powdery mildew high-end temperature threshold
    :param pm_lwd_thresh:           powdery mildew low-end leaf wetness duration threshold
    :param dm_threshold_modifier:   downy mildew threshold modification factor
    :param bot_threshold_modifier:  botrytis threshold modification factor
    :param pm_threshold_modifier:   powdery mildew threshold modification factor
    :param lwd_model:               selected leaf wetness model to base disease prediction off
    :return df_hour:
    """
    dm_rh_thresh /= dm_threshold_modifier
    dm_temp_thresh_low /= dm_threshold_modifier
    dm_temp_thresh_high *= dm_threshold_modifier
    dm_lwd_thresh /= dm_threshold_modifier
    bot_rh_thresh /= bot_threshold_modifier
    bot_temp_thresh_low /= bot_threshold_modifier
    bot_temp_thresh_high *= bot_threshold_modifier
    bot_lwd_thresh /= bot_threshold_modifier
    pm_rh_thresh /= pm_threshold_modifier
    pm_temp_thresh_low /= pm_threshold_modifier
    pm_temp_thresh_high *= pm_threshold_modifier
    pm_lwd_thresh /= pm_threshold_modifier
    # Modeling disease from:Estimation of leaf wetness duration for greenhouse roses using a dynamic greenhouse
    # climate model in Zimbabwe
    # _________________________
    # Downy Mildew
    df_hour['downy_mildew_conditions'] = 0
    df_hour.loc[(df_hour['RH'] >= dm_rh_thresh) &
                (dm_temp_thresh_low <= df_hour['Temperature']) &
                (df_hour['Temperature'] <= dm_temp_thresh_high) &
                (df_hour[lwd_model] >= dm_lwd_thresh), 'downy_mildew_conditions'] = 1
    # Botrytis
    df_hour['botrytis_conditions'] = 0
    df_hour.loc[(df_hour['RH'] >= bot_rh_thresh) &
                (df_hour['Temperature'] <= bot_temp_thresh_high) &
                (bot_temp_thresh_low <= df_hour['Temperature']) &
                (df_hour[lwd_model] >= bot_lwd_thresh), 'botrytis_conditions'] = 1
    # Powdery Mildew
    df_hour['powdery_mildew_conditions'] = 0
    df_hour.loc[(df_hour['RH'] >= pm_rh_thresh) &
                (pm_temp_thresh_low <= df_hour['Temperature']) &
                (df_hour['Temperature'] <= pm_temp_thresh_high) &
                (df_hour[lwd_model] >= pm_lwd_thresh), 'powdery_mildew_conditions'] = 1

    return df_hour


def leaf_wetness_summary(df_hour):
    """
    Prints total hours of leaf wetness for each leaf wetness model
    
    :param df_hour: data frame with hourly data for one sensor with temperature, humidity, leaf wetness, leaf wetness 
                    duration, and datetime
    """
    print('LFD totals:',
          '\n\tDPD air:\t\t', df_hour['leaf_wetness_DPD_air'].sum(),
          '\n\tRH threshold:\t', df_hour['leaf_wetness_RH_thresh'].sum(),
          '\n\tRH extended:\t', df_hour['leaf_wetness_RH_extended'].sum())


def plot_leaf_wetness(df_hour, lwd_model='LWD_RH_extended'):
    """
    Generates plots for relative humidity with all leaf wetness models as well as temperature with all disease models

    :param df_hour:
    :param lwd_model:
    :return figure, figure:
    """

    # Plotting the leaf wetness duration time series graph with all mechanistic models

    leaf_wetness_fig = matplotlib.figure.Figure(figsize=(7, 5.5), dpi=100)
    spec = GridSpec(ncols=1, nrows=4, width_ratios=[2], height_ratios=[4, 1, 1, 1])

    # Add subplots to the figure
    ax31 = leaf_wetness_fig.add_subplot(spec[0])  # First (larger)
    ax32 = leaf_wetness_fig.add_subplot(spec[1])  # Second subplot
    ax33 = leaf_wetness_fig.add_subplot(spec[2])  # Third subplot
    ax34 = leaf_wetness_fig.add_subplot(spec[3])  # Fourth subplot
    ax31.plot(df_hour.index, df_hour['RH'], color='b')
    ax31.set_ylabel('Relative Humidity [%]')

    col = ['y', 'y', 'y']
    if lwd_model == 'LWD_DPD_air':
        col[0] = 'm'
    elif lwd_model == 'LWD_RH_thresh':
        col[1] = 'm'
    elif lwd_model == 'LWD_RH_extended':
        col[2] = 'm'

    ax32.bar(df_hour.index, df_hour['leaf_wetness_DPD_air'], width=0.05, color=col[0])
    ax33.bar(df_hour.index, df_hour['leaf_wetness_RH_thresh'], width=0.05, color=col[1])
    ax34.bar(df_hour.index, df_hour['leaf_wetness_RH_extended'], width=0.05, color=col[2])

    ax32.set_ylim(0, 1)
    ax32.set_yticks([0, 1])
    ax33.set_ylim(0, 1)
    ax33.set_yticks([0, 1])
    ax34.set_ylim(0, 1)
    ax34.set_yticks([0, 1])

    ax31.set_xticklabels([])
    ax32.set_xticklabels([])
    ax33.set_xticklabels([])
    ax34.tick_params(axis='x', labelrotation=45)
    leaf_wetness_fig.subplots_adjust(bottom=0.2)

    # Set the titles and labels for the subplots
    ax31.set_ylabel('Relative Humidity [%]')
    ax32.set_ylabel('DPD_air')
    ax33.set_ylabel('RH_thresh')
    ax34.set_ylabel('RH_extended')
    ax31.set_title('Relative Humidity')
    ax32.set_title('Leaf Wetness')

    # Adjust the spacing between subplots
    leaf_wetness_fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # Plotting the satisfied disease conditions time series graph with all diseases
    
    disease_fig = matplotlib.figure.Figure(figsize=(7, 5.5), dpi=100)
    spec = GridSpec(ncols=1, nrows=4, width_ratios=[2], height_ratios=[4, 1, 1, 1])

    # Add subplots to the figure
    ax7 = disease_fig.add_subplot(spec[0])  # First subplot (larger)
    ax8 = disease_fig.add_subplot(spec[1])  # Second subplot
    ax9 = disease_fig.add_subplot(spec[2])  # Third subplot
    ax10 = disease_fig.add_subplot(spec[3])  # Fourth subplot

    # Plot your data on each subplot
    ax7.plot(df_hour.index, df_hour['Temperature'], color='r')
    ax8.bar(df_hour.index, df_hour['downy_mildew_conditions'], width=0.05, color='m')
    ax9.bar(df_hour.index, df_hour['botrytis_conditions'], width=0.05, color='m')
    ax10.bar(df_hour.index, df_hour['powdery_mildew_conditions'], width=0.05, color='m')

    ax8.set_ylim(0, 1)
    ax8.set_yticks([0, 1])
    ax9.set_ylim(0, 1)
    ax9.set_yticks([0, 1])
    ax10.set_ylim(0, 1)
    ax10.set_yticks([0, 1])

    ax7.set_xticklabels([])
    ax8.set_xticklabels([])
    ax9.set_xticklabels([])
    ax10.tick_params(axis='x', labelrotation=45)
    disease_fig.subplots_adjust(bottom=0.2)

    # Set the titles and labels for the subplots
    ax7.set_ylabel('Temperature [C]')
    ax8.set_ylabel('Downy Mildew')
    ax9.set_ylabel('Botrytis')
    ax10.set_ylabel('Powdery Mildew')
    ax7.set_title('Temperature')
    ax8.set_title('Disease conditions satisfied')

    # Adjust the spacing between subplots
    disease_fig.subplots_adjust(hspace=0.4, wspace=0.4)

    return leaf_wetness_fig, disease_fig


def plot_heatmap(sensor_list, sensor_lookup_list, disease='downy_mildew_conditions', start_date=start, end_date=end,
                 lwd_model='LWD_RH_extended'):
    """
    Generates animation of heatmap of satisfied disease occurrence conditions in the given time range and saves the
    dataframe as a csv file

    :param sensor_list:         list of all sensor numbers
    :param sensor_lookup_list:  lookup table of sensor values to sensor location within the greenhouse
    :param disease:             target disease to observe
    :param start_date:          starting date of data range
    :param end_date:            ending date of data range
    :param lwd_model:           leaf wetness model type to use
    """
    df_sensors = dict()
    jackie_df = pd.DataFrame()
    for sensor_id in sensor_list:
        print("Getting data for sensor ", sensor_id)
        df_sensors[sensor_id] = get_data(sensor_id, start_date=start_date, end_date=end_date)
        df_sensors[sensor_id] = leaf_wetness(df_sensors[sensor_id])
        df_sensors[sensor_id] = disease_prediction(df_sensors[sensor_id], lwd_model=lwd_model)
        df_sensors[sensor_id]['device'] = sensor_lookup_list.loc[sensor_id]['device']
        df_sensors[sensor_id]['x_loc'] = sensor_lookup_list.loc[sensor_id]['x_loc']
        df_sensors[sensor_id]['y_loc'] = sensor_lookup_list.loc[sensor_id]['y_loc']
        frames = [jackie_df, df_sensors[sensor_id]]
        jackie_df = pd.concat(frames)

    jackie_df.to_csv("Disease_conditions_satisfied.csv")

    df_sensors_list = np.array([
        [df_sensors[383][disease], df_sensors[463][disease], df_sensors[442][disease],
         df_sensors[474][disease], df_sensors[481][disease], df_sensors[529][disease]],
        [df_sensors[500][disease], df_sensors[467][disease], df_sensors[514][disease],
         df_sensors[458][disease], df_sensors[516][disease], df_sensors[522][disease]],
        [df_sensors[316][disease], df_sensors[473][disease], df_sensors[453][disease],
         df_sensors[492][disease], df_sensors[488][disease], df_sensors[468][disease]]
    ])

    colors = ((52.0 / 255.0, 175.0 / 255.0, 33.0 / 255.0), (125.0 / 255.0, 64.0 / 255.0, 17.0 / 255.0))
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list('Custom', colors, len(colors))
    heatmap = plt.figure()
    heat_ax = sns.heatmap(df_sensors_list[:, :, 0], cmap=color_map, linewidths=2, vmax=1, vmin=0, square=True)
    color_bar = heat_ax.collections[0].colorbar
    color_bar.set_ticks([0.25, 0.75])
    color_bar.set_ticklabels(['Healthy', 'Diseased'])

    def heatmap_init():
        sns.heatmap(np.zeros(df_sensors_list[:, :, 0].shape), cmap=color_map, vmax=1, cbar=False, square=True)

    def heatmap_animate(i):
        sns.heatmap(df_sensors_list[:, :, i], cmap=color_map, cbar=False, square=True)
        plt.title(df_sensors[383].index[i])

    anim = animation.FuncAnimation(heatmap, heatmap_animate, init_func=heatmap_init(),
                                   frames=df_sensors_list[0, 0, :].size, repeat=True)

    anim.save('DiseaseHeatmap.gif')


def draw_figure(canvas, graph):
    """
    Draws plots onto the given canvas environment

    :param canvas:  environment to store plots
    :param graph:   graph to display on the canvas
    :return:
    """
    figure_canvas_agg = FigureCanvasTkAgg(graph, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


def update_chart(df, canvas_components, lwd_modifier, dm_modifier, bot_modifier, pm_modifier, lwd_type):
    """
    Updates the application plots with new functions related to which data is shown. Does not change the dataframe

    :param df:                  data frame with hourly data for one sensor with temperature, humidity, leaf wetness,
                                leaf wetness duration, and datetime
    :param canvas_components:   environment to display plots
    :param lwd_modifier:        leaf wetness duration modification factor
    :param dm_modifier:         downy mildew modification factor
    :param bot_modifier:        botrytis modification factor
    :param pm_modifier:         powdery mildew modification factor
    :param lwd_type:            leaf wetness model
    """
    leaf_wetness(df, lwd_threshold_modifier=lwd_modifier)
    disease_prediction(df, lwd_model=lwd_type, dm_threshold_modifier=dm_modifier, bot_threshold_modifier=bot_modifier,
                       pm_threshold_modifier=pm_modifier)
    lw_figure, dis_figure = plot_leaf_wetness(df, lwd_model=lwd_type)
    canvas_components['fig_agg_lwd'].get_tk_widget().forget()
    canvas_components['fig_agg_lwd'] = draw_figure(canvas_components['window']["lwdCanvas"].TKCanvas, lw_figure)
    canvas_components['fig_agg_dis'].get_tk_widget().forget()
    canvas_components['fig_agg_dis'] = draw_figure(canvas_components['window']["disCanvas"].TKCanvas, dis_figure)


def run_application(df_disease):
    """
    Runs and displays the application for tuning of leaf wetness modification and disease modification parameters

    :param df_disease:  data frame with hourly data for one sensor with temperature, humidity, leaf wetness,
                        leaf wetness duration, and datetime
    """
    lw_figure, dis_figure = plot_leaf_wetness(df_disease)

    font = ("Arial", 11)

    _VARS = {'window': False,
             'fig_agg_lwd': FigureCanvasTkAgg,
             'fig_agg_dis': False,
             'pltFig': False}

    # Define the window layout
    lwd_column = [
        [Sg.Text("")],
        [Sg.Text("lwd_mod: ", font=font),
         Sg.Slider(range=(0, 2), default_value=1, resolution=.01, orientation='horizontal', key='lwd_mod_slider')],
        [Sg.Text("")],
    ]

    disease_column = [
        [Sg.Text("dm_mod: ", font=font),
         Sg.Slider(range=(0, 2), size=(20, 10), default_value=1, resolution=.01, orientation='horizontal',
                   key='dm_mod_slider')],
        [Sg.Text("bot_mod: ", font=font),
         Sg.Slider(range=(0, 2), size=(20, 10), default_value=1, resolution=.01, orientation='horizontal',
                   key='bot_mod_slider')],
        [Sg.Text("pm_mod: ", font=font),
         Sg.Slider(range=(0, 2), size=(20, 10), default_value=1, resolution=.01, orientation='horizontal',
                   key='pm_mod_slider')],
    ]

    layout = [[Sg.Combo(dates, font=('Arial', 14), default_value=dates[1], readonly=False, key='start_date'),
               Sg.Combo(sensors, font=('Arial', 14), default_value=sensors[0], readonly=True, key='device'),
               Sg.Combo(dates, font=('Arial', 14), default_value=dates[2], readonly=False, key='end_date'),
               Sg.Button("Set")],
              [Sg.Column(lwd_column),
               Sg.VSeparator(),
               Sg.Column(disease_column)],
              [Sg.Combo(lwd_model_names, font=('Arial', 14), default_value=lwd_model_names[2], readonly=True,
                        key='dropdown'),
              Sg.Button("Update")],
              [Sg.Canvas(key="lwdCanvas"),
               Sg.VSeparator(),
               Sg.Canvas(key="disCanvas")]]

    # Create the form and show it without the plot
    _VARS['window'] = Sg.Window("Matplotlib Single Graph",
                                layout,
                                location=(0, 0),
                                finalize=True,
                                element_justification="center",
                                font="Helvetica 18",
                                )

    # Add the plot to the window
    _VARS['fig_agg_lwd'] = draw_figure(_VARS['window']["lwdCanvas"].TKCanvas, lw_figure)
    _VARS['fig_agg_dis'] = draw_figure(_VARS['window']["disCanvas"].TKCanvas, dis_figure)

    while True:
        event, values = _VARS['window'].read()
        if event == 'Update':
            lwd_mod = float(values['lwd_mod_slider'])
            dm_slider = float(values['dm_mod_slider'])
            bot_slider = float(values['bot_mod_slider'])
            pm_slider = float(values['pm_mod_slider'])
            lwd_type = values['dropdown']

            update_chart(df_disease, _VARS, lwd_mod, dm_slider, bot_slider, pm_slider, lwd_type)
        elif event == "Set":
            device_id = int(values['device'])
            start_date = values['start_date']
            end_date = values['end_date']
            df_disease = get_data(device_id, start_date, end_date)
            lwd_mod = float(values['lwd_mod_slider'])
            dm_slider = float(values['dm_mod_slider'])
            bot_slider = float(values['bot_mod_slider'])
            pm_slider = float(values['pm_mod_slider'])
            lwd_type = values['dropdown']
            update_chart(df_disease, _VARS, lwd_mod, dm_slider, bot_slider, pm_slider, lwd_type)

        elif event == Sg.WIN_CLOSED:
            break

    _VARS['window'].close()


# Get data from the web API for the specified sensor ID, date range, and limit
df_hourly = get_data(sensor, start, end)

# Calculate leaf wetness using the retrieved data
leaf_wetness(df_hourly)

# Calculate disease conditions using retrieved and calculated data
disease_prediction(df_hourly)

# Print summary of leaf wetness hours
leaf_wetness_summary(df_hourly)

# Generate heatmap of the disease occurrences and saves animation gif and csv
# plot_heatmap(sensors, sensor_lookup)

# Runs the application for modification factor tuning and overall visualisation
run_application(df_hourly)
