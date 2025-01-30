# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13, 2021
Last updated: Jan. 19, 2025

@author: Climate Lead Group; Luis Victor-Gallardo, Jairo Quirós-Tortós,
        Andrey Salazar-Vargas
Suggested citation: UNEP (2022). Is Natural Gas a Good Investment for Latin 
                                America and the Caribbean? From Economic to 
                                Employment and Climate Impacts of the Power
                                Sector. https://wedocs.unep.org/handle/20.500.11822/40923
"""

import pandas as pd
import pickle
import sys
from copy import deepcopy
import math
import numpy as np
import time
import os
import warnings
import yaml

# Import functions that support this model:
from model_bulac_funcs import intersection_2, interpolation_to_end, \
    fun_reverse_dict_data, fun_extract_new_dict_data, fun_dem_model_projtype, \
    fun_dem_proj, fun_unpack_costs, fun_unpack_taxes, \
    interpolation_non_linear_final, unpack_values_df_2, \
    compute_delta_for_technology, discounted_values
 
# Import Tier 2 of this model:
import bulac_tier2      

pd.options.mode.chained_assignment = None  # default='warn'

# Globally suppress the FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

# > Define booleans to control process:
# if True, it overwrites energy projections with transport model
overwrite_transport_model = True
model_agro_and_waste = False
model_rac = False
model_irote = False

##############################################################################
# SIMULATION: Implement the equations designed in "model_design" #
# Recording initial time of execution
start_1 = time.time()
di_nam = 'data_inputs_20250119.xlsx'

###############################################################################
# 0) open the reference data bases.

dict_database = pickle.load(open('dict_db.pickle', 'rb'))
try:
    iea_baseyear_prod = deepcopy(dict_database['IEA'])
    iea_baseyear_prod.fillna(0, inplace = True)
    del dict_database['IEA']
except:
    pass
###############################################################################
# Will save all files in one outputs folder. 
# Folder is created if it does not exist
cwd = os.getcwd()
path = cwd + "/outputs"

if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
    
###############################################################################
# Read the tier 2 config file   
with open('config_tier2.yaml', 'r') as file:
    params_tier2 = yaml.safe_load(file)    

###############################################################################
# Here we relate power plants and production per carrier in the base year:
'''
#> Select the technologies that exist in the base year:
PP_Offshore_Wind: -
PP_Onshore_Wind: base year
PP_PV Utility_Solar: base year
PP_PV DistComm_Solar: -
PP_PV DistResi_Solar: -
PP_CSP_Solar: -
PP_Geothermal: base year
PP_Hydro: base year
PP_Nuclear: base year
PP_Thermal.re_Sugar cane and derivatives: base year
PP_PV Utility+Battery_Solar: -
PP_Thermal_Coal: base year
PP_Thermal_Natural Gas: base year
ST_Utility Scale Battery: -
ST_Commercial Battery: -
ST_Residential Battery: -
ST_Pumped Hydro: -
PP_Other: -
PP_Thermal_Fuel oil: - (we cannot distinguish Diesel/Fuel Oil with our data)
PP_Thermal_Diesel: base year
'''

'''
#> Select the fuels that exist in the base year:
Coal: base year
Oil: base year
Natural gas: base year
Biofuels: base year
Waste: -
Nuclear: base year
Hydro: base year
Geothermal: base year
Solar PV: base year
Solar thermal: -
Wind: base year
Tide: -
Other sources: -
'''

dict_equiv_country = { \
    'Argentina':'Argentina',  # this works for iea dataframe
    'Barbados':'Barbados',
    'Belize':'Belice',
    'Bolivia':'Bolivia',
    'Brazil':'Brasil',
    'Chile':'Chile',
    'Colombia':'Colombia',
    'Costa Rica':'Costa Rica',
    'Cuba':'Cuba',
    'Ecuador':'Ecuador ',
    'El Salvador':'El Salvador',
    'Grenada':'Grenada',
    'Guatemala':'Guatemala',
    'Guyana':'Guyana ',
    'Haiti':'Haiti',
    'Honduras':'Honduras',
    'Jamaica':'Jamaica',
    'Mexico':'Mexico ',
    'Nicaragua':'Nicaragua',
    'Panama':'Panama',
    'Paraguay':'Paraguay',
    'Peru':'Peru',
    'Dominican Republic':'Republica Dominicana',
    'Suriname':'Suriname',
    'Trinidad and Tobago':'Trinidad & Tobago',
    'Uruguay':'Uruguay',
    'Venezuela':'Venuezuela'}

dict_equiv_country_2 = { \
    'Argentina':'Argentina',  # this works for iea dataframe
    'Barbados':'Barbados',
    'Belize':'Belize',
    'Bolivia':'Bolivia',
    'Brazil':'Brazil',
    'Chile':'Chile',
    'Colombia':'Colombia',
    'Costa Rica':'Costa Rica',
    'Cuba':'Cuba',
    'Ecuador':'Ecuador',
    'El Salvador':'El Salvador',
    'Grenada':'Grenada',
    'Guatemala':'Guatemala',
    'Guyana':'Guyana',
    'Haiti':'Haiti',
    'Honduras':'Honduras',
    'Jamaica':'Jamaica',
    'Mexico':'Mexico',
    'Nicaragua':'Nicaragua',
    'Panama':'Panama',
    'Paraguay':'Paraguay',
    'Peru':'Peru',
    'Dominican Republic':'Dominican Republic',
    'Suriname':'Suriname',
    'Trinidad and Tobago':'Trinidad & Tobago',
    'Uruguay':'Uruguay',
    'Venezuela':'Venezuela'}

# Find the common countries per region:
unique_reg = []
dict_regs_and_countries_raw = {}
dict_regs_and_countries = {}

k1_count = 0
k1_list = []
for k1 in list(dict_database.keys()):  # across databases
    k1_list.append(k1)
    for k2 in list(dict_database[k1].keys()):  # across regions
        dummy_list = list(dict_database[k1][k2].keys())
        add_dummy = {k1:dummy_list}
        if k1_count == 0:
            dict_regs_and_countries_raw.update({k2:add_dummy})
        else:
            if k2 == '2_CA':
                k2 = '2_Central America'
            else:
                pass
            dict_regs_and_countries_raw[k2].update(add_dummy)
    k1_count += 1

for reg in list(dict_regs_and_countries_raw.keys()):
    if 'Trinidad & Tobago' in dict_regs_and_countries_raw[reg][k1_list[0]]:
        fix_idx = dict_regs_and_countries_raw[reg][k1_list[0]].index('Trinidad & Tobago')
        dict_regs_and_countries_raw[reg][k1_list[0]][fix_idx] = 'Trinidad and Tobago'
    country_list = intersection_2(dict_regs_and_countries_raw[reg][k1_list[0]],
                                  dict_regs_and_countries_raw[reg][k1_list[1]])
    dict_regs_and_countries.update({reg:country_list})

###############################################################################
# *Input parameters are listed below*:
# Capacity: (initial condition by power plant Tech, OLADE, dict_database)
# Production: (initial condition by power plant Tech, OLADE, dict_database (total supply))
# Imports: (initial condition by power plant Tech, OLADE, dict_database (total supply))
# Exports: (initial condition by power plant Tech, OLADE, dict_database (total supply))
# Externality cost: (by Fuel, IMF, data_inputs => costs_externalities)
# CAPEX (capital unit cost per unit of capacity, ATB, data_inputs => costs_power_techs)
# CAU (max. activity production per unit of capacity, ATB, data_inputs => costs_power_techs)
# Fixed FOM (fixed FOM unit cost per unit of capacity, ATB, data_inputs => costs_power_techs)
# Grid connection cost (GCC unit cost per unit of capacity, ATB, data_inputs => costs_power_techs)
# Heat Rate (IAR (only for fossil-based plants), ATB, data_inputs => costs_power_techs)
    # WARNING! may need IAR for renewables
# Net capacity factor: (real activity per max. activity possible, ATB, data_inputs => costs_power_techs)
# Operational life (by power plant Tech, ATB, data_inputs => costs_power_techs)
# Variable FOM (by power plant activity, ATB, data_inputs => costs_power_techs)
# Emission factor: (by Fuel, type == consumption, see README of data_inputs)
# Demand energy intensity (by Demand sector == Tech, data_inputs => scenarios)
# Distribution of end-use consumption (by Fuel, data_inputs => scenarios)
# Distribution of new electrical energy generation (by power plant Tech, data_inputs => scenarios)
# %Imports (by Fuel, data_inputs => scenarios)
# %Exports (by Fuel, data_inputs => scenarios)
# GDP growth (combine with energy intensity, data_inputs => scenarios)
# GDP (grab from historic data, and combine with GDP growth)
# Export price (by Fuel) # WARNING! MISSING CORRECT CALIBRATION!
# Import price (by Fuel) # WARNING! MISSING CORRECT CALIBRATION!

# *Output parameters are listed below*:
# Energy Demand by Fuel and Demand sector (Tech)
# New capacity: endogenous // f(demand, scenarios, net capacity factor)
# CAPEX (by power plant Tech)
# Fixed OPEX (by power plant Tech)
# Var OPEX (by Fuel)
# Imports expenses (by Fuel)
# Exports revenue (by Fuel)
# Emissions (by Fuel)
# Externality Global Warming (by Fuel)
# Externality Local Pollution (by Fuel)

# Review the EB structure:
# Capacity: Cap > region > country > tech (power plant) > year (2019)
# Demand: EB > region > country > Energy consumption > Demand sector > Fuel > year str(2019)
# Transformation: EB > region > country > Total transformation > Demand sector > Fuel > year str(2019)
    # here, negative: use // positive: produce
# Local production: EB > Total supply > Exports/Imports/Production > Fuel > year str(2019)

# Then, simply produce a table with inputs and outputs for each dimension combination, but make sure to have the calculations done first.

###############################################################################
# 3) implementing...
# 3a) open the "data_inputs.xlsx", each one of the sheets into dfs:

'''
Name definitions:
di_nam: data_inputs_name
'''

# General sheets:
df1_general = pd.read_excel(di_nam, sheet_name="2_general")

# Calibraton and sets sheets:
'''
This code the energy balances introduced by the user:
'''
df2_fuel_eq = pd.read_excel(di_nam, sheet_name="3_FUEQ")
df2_EB = pd.read_excel(di_nam, sheet_name="4_EB")
df2_InsCap = pd.read_excel(di_nam, sheet_name="5_InsCap")
df2_scen_sets = pd.read_excel(di_nam, sheet_name="6_scen_sets")
df2_sets2pp = pd.read_excel(di_nam, sheet_name="7_set2pp")
df2_trans_sets = pd.read_excel(di_nam, sheet_name="8_trans_sets")
df2_trans_sets_eq = pd.read_excel(di_nam, sheet_name="9_trans_sets_eq")
# df2_agr_sets_eq = pd.read_excel(di_nam, sheet_name="10_agro_sets")
# df2_res_sets_eq = pd.read_excel(di_nam, sheet_name="11_res_sets")

# Scenarios sheets:
df3_scen = pd.read_excel(di_nam, sheet_name="12_scen")
df3_scen_matrix = pd.read_excel(di_nam, sheet_name="12.2_scen_matrix") # viene de Relac
df3_scen_dems = pd.read_excel(di_nam, sheet_name="13_scen_dems")
df3_tpt_data = pd.read_excel(di_nam, sheet_name="14_trans_data")
# df3_agr_data = pd.read_excel(di_nam, sheet_name="15_agro_data")
# df3_res_data = pd.read_excel(di_nam, sheet_name="16_res_data")

# Technical sheets:
# df4_rac_data = pd.read_excel(di_nam, sheet_name="17_rac_data")  # nueva!
# df4_ef_agro_res = \
#     pd.read_excel(di_nam, sheet_name="18_agro_res_emissions")
# df4_ar_emi = pd.read_excel(di_nam, sheet_name="19_ar_emissions")  # nueva!
df4_cfs = pd.read_excel(di_nam, sheet_name="20_cfs")
df4_ef = pd.read_excel(di_nam, sheet_name="21_emissions")
# df4_rac_emi = pd.read_excel(di_nam, sheet_name="22_rac_emissions")  # nueva!
df4_job_fac = pd.read_excel(di_nam, sheet_name="23_job_fac")
df4_tran_dist_fac = pd.read_excel(di_nam, sheet_name="24_t&d")
df4_caps_rest = pd.read_excel(di_nam, sheet_name="25_cap_rest")

# Economic sheets:
df5_ext = pd.read_excel(di_nam, sheet_name="26_ext")
# d5_res = pd.read_excel(di_nam, sheet_name="27_res_cost")
d5_power_techs = pd.read_excel(di_nam, sheet_name="28_power_cost")
d5_tpt = pd.read_excel(di_nam, sheet_name="29_trans_cost")
# d5_agr = pd.read_excel(di_nam, sheet_name="30_agro_cost")
# d5_rac = pd.read_excel(di_nam, sheet_name="31_rac_cost")  # nueva!
d5_tax = pd.read_excel(di_nam, sheet_name="32_tax")

##############################################################################
# Process the content of the general sheet (1_general):
df1_general.set_index('Parameter', inplace = True)
dict_general_inp = df1_general.to_dict('index')
'''
The columns of the general dictionary are:
['Value', 'Year', 'Attribute', 'Unit', 'Source', 'Description']

The keys (variables) of the dictionary are:
 ['ini_year', 'fin_year', 'country', 'gdp', 'discount_rate', 'discount_year']
'''

# Call years
per_first_yr = dict_general_inp['ini_year']['Value']
per_last_yr = dict_general_inp['fin_year']['Value']
time_vector = [i for i in range(per_first_yr, per_last_yr+1)]

# Call countries
dict_gen_all_param_names = list(dict_general_inp.keys())
dict_gen_country_params = [i for i in dict_gen_all_param_names
                           if 'country' in i]

general_country_list, gen_cntry_list_param_idx = [], []
for cntry_idx in dict_gen_country_params:
    general_country_list.append(dict_general_inp[cntry_idx]['Value'])
    gen_cntry_list_param_idx.append(cntry_idx)

# Get the regions of interest and store them for future use:
regions_list = []
all_regions = dict_regs_and_countries.keys()
country_2_reg = {}
for areg in all_regions:
    all_country_list = dict_regs_and_countries[areg]
    for cntry in all_country_list:
        country_2_reg.update({cntry:areg})

        # Store useful regions for future use:
        if areg not in regions_list and cntry in general_country_list:
            regions_list.append(areg)

# Call GDP and population.
gdp_dict = {}  # per country index
popbase_dict = {}
popfinal_dict = {}
popproj_dict = {}
for cntry in general_country_list:
    cntry_list_idx = general_country_list.index(cntry)
    cntry_idx = gen_cntry_list_param_idx[cntry_list_idx]
    gdp_idx = 'gdp_' + str(cntry_idx.split('_')[-1])
    popbase_idx = 'pop_base_' + str(cntry_idx.split('_')[-1])
    popfinal_idx = 'pop_final_' + str(cntry_idx.split('_')[-1])
    popproj_idx = 'pop_proj_' + str(cntry_idx.split('_')[-1])

    if gdp_idx in list(dict_general_inp.keys()):
        gdp_value = dict_general_inp[gdp_idx]['Value']
        gdp_dict.update({cntry: gdp_value})

        popbase_value = dict_general_inp[popbase_idx]['Value']
        popbase_dict.update({cntry: popbase_value})

        popfinal_value = dict_general_inp[popfinal_idx]['Value']
        popfinal_dict.update({cntry: popfinal_value})

        popproj_value = dict_general_inp[popproj_idx]['Value']
        popproj_dict.update({cntry: popproj_value})
    else:
        print('There is no GDP value defined for: ' + cntry)

'''
Development note: only introduce 1 GDP year for the future. The rest of the
years should be controlled by the GDP growth parameter.

The population is introduced for two years: first and last. The interpolation
is linear. This can change of other data is provided.
'''

# Call information for discounting:
r_rate = dict_general_inp['discount_rate']['Value']
r_year = dict_general_inp['discount_year']['Value']
ini_simu_yr = dict_general_inp['ini_simu_yr']['Value']

##############################################################################
# Process the content of structural sheets:

# This code extracts the sets used for energy balancing:
list_scen_fuels = df2_scen_sets['Fuel'].tolist()
list_scen_fuel_primary_and_secondary = \
    df2_scen_sets['Primary, Secondary or Power'].tolist()
list_scen_fuels_u = list(dict.fromkeys(list_scen_fuels))
list_scen_fuels_u_prim_and_sec = []
list_scen_fuels_cat_u = []
for af in list_scen_fuels_u:
    this_fuel_idx = list_scen_fuels.index(af)
    this_fuel_cat = list_scen_fuel_primary_and_secondary[this_fuel_idx]
    list_scen_fuels_cat_u.append(this_fuel_cat)
    if this_fuel_cat in ['Primary', 'Secondary']:
        list_scen_fuels_u_prim_and_sec.append(af)
        
# This code extracts sets to connect power plants to energy balance:
dict_equiv_pp_fuel = {}
dict_equiv_pp_fuel_rev = {}
for n in range(len(df2_sets2pp['Technology'])):
    dict_equiv_pp_fuel.update(
        {df2_sets2pp['Technology'][n]:\
         df2_sets2pp['Fuel'][n]})
    dict_equiv_pp_fuel_rev.update(
        {df2_sets2pp['Fuel'][n]:\
         df2_sets2pp['Technology'][n]})

# This code extracts the transport sets and its structure:
list_trn_type = df2_trans_sets['Type'].tolist()
list_trn_fuel = df2_trans_sets['Fuel'].tolist()
list_trn_type_and_fuel = []
for n in range(len(list_trn_type)):
    this_type, this_fuel = list_trn_type[n], list_trn_fuel[n]
    if this_fuel != '-':
        this_type_and_fuel = this_type + '_' + this_fuel
    else:
        this_type_and_fuel = this_type
    list_trn_type_and_fuel.append(this_type_and_fuel)

list_trn_lvl1_u_raw = df2_trans_sets['Demand set level 1'].tolist()
list_trn_lvl2_u_raw = df2_trans_sets['Demand set level 2'].tolist()
list_trn_lvl1_u = \
    [i for i in list(dict.fromkeys(list_trn_lvl1_u_raw)) if '-' != i]
list_trn_lvl2_u = \
    [i for i in list(dict.fromkeys(list_trn_lvl2_u_raw)) if '-' != i]
# The level 2 list only applies to passenger vehicles
dict_trn_nest = {}
for l1 in range(len(list_trn_lvl1_u)):
    this_l1 = list_trn_lvl1_u[l1]
    dict_trn_nest.update({this_l1:{}})
    if this_l1 != 'Passenger':
        this_l2 = 'All'
        mask_trans_t_and_f = \
            (df2_trans_sets['Demand set level 1'] == this_l1) & \
            (df2_trans_sets['Fuel'] == '-')
        df_transport_t_and_f = df2_trans_sets.loc[mask_trans_t_and_f]
        list_trn_types = df_transport_t_and_f['Type'].tolist()
        dict_trn_nest[this_l1].update({this_l2:deepcopy(list_trn_types)})
    else:
        for l2 in range(len(list_trn_lvl2_u)):
            this_l2 = list_trn_lvl2_u[l2]
            mask_trans_t_and_f = \
                (df2_trans_sets['Demand set level 1'] == this_l1) & \
                (df2_trans_sets['Demand set level 2'] == this_l2) & \
                (df2_trans_sets['Fuel'] == '-')
            df_transport_t_and_f = df2_trans_sets.loc[mask_trans_t_and_f]
            list_trn_types = df_transport_t_and_f['Type'].tolist()
            dict_trn_nest[this_l1].update({this_l2:deepcopy(list_trn_types)})

# This code extracts set change equivalence:
pack_fe = {'new2old':{}, 'old2new':{}}
for n in range(len(df2_fuel_eq['OLADE_structure'].tolist())):
    old_struc = df2_fuel_eq['OLADE_structure'].tolist()[n]
    new_struc = df2_fuel_eq['New_structure'].tolist()[n]
    pack_fe['new2old'].update({new_struc: old_struc})
    pack_fe['old2new'].update({old_struc: new_struc})

# we have to open the first data frame:
# 1) list all the unique elements from the "df3_tpt_data" parameters:

# Function #1
def get_list_and_unique_list(dataframe_column):
    """
    Extracts all elements from a DataFrame column and creates a list of unique elements from it.

    Parameters:
    - dataframe_column (pandas.core.series.Series): The column from a DataFrame whose elements are to be extracted. This column is expected to contain a series of values from which a list and a unique list will be generated.

    Returns:
    - complete_list (list): A list containing all elements from the dataframe_column.
    - unique_list (list): A list containing only the unique elements from the dataframe_column, with duplicates removed.
    """
    complete_list = dataframe_column.tolist()
    unique_list = list(dict.fromkeys(complete_list))
    return complete_list, unique_list


tr_list_scenarios, tr_list_scenarios_u = get_list_and_unique_list(df3_tpt_data['Scenario'])
tr_list_app_countries, tr_list_app_countries_u = get_list_and_unique_list(df3_tpt_data['Application_Countries'])
tr_list_parameters, tr_list_parameters_u = get_list_and_unique_list(df3_tpt_data['Parameter'])
tr_list_type_and_fuel, tr_list_type_and_fuel_u = get_list_and_unique_list(df3_tpt_data['Type & Fuel ID'])
tr_list_type, tr_list_type_u = get_list_and_unique_list(df3_tpt_data['Type'])
tr_list_fuel, tr_list_fuel_u = get_list_and_unique_list(df3_tpt_data['Fuel'])
tr_list_projection, tr_list_projection_u = get_list_and_unique_list(df3_tpt_data['projection'])

# We must overwrite the dict-database based on OLADE for a user_defined
# input to avoid compatibility issues.
use_original_pickle = False
if use_original_pickle is True:
    pass
else:
    dict_database_freeze = deepcopy(dict_database)
    print('We must re-write the base data. This can take a while.')
    # We must use the reference EB and InstCap sheets from data_inputs
    # agile_mode = True
    agile_mode = False
    if agile_mode is False:
        dict_ref_EB, dict_ref_InstCap = \
            fun_extract_new_dict_data(df2_EB, df2_InsCap, per_first_yr)
        with open('dict_ref_EB.pickle', 'wb') as handle1:
            pickle.dump(dict_ref_EB, handle1,
                        protocol=pickle.HIGHEST_PROTOCOL)
        handle1.close()
        with open('dict_ref_InstCap.pickle', 'wb') as handle2:
            pickle.dump(dict_ref_InstCap, handle2,
                        protocol=pickle.HIGHEST_PROTOCOL)
        handle2.close()
    else:
        dict_ref_EB = pickle.load(open('dict_ref_EB.pickle', 'rb'))
        dict_ref_InstCap = \
            pickle.load(open('dict_ref_InstCap.pickle', 'rb'))

    # We must replace the dictionaries:
    dict_database['EB'] = deepcopy(dict_ref_EB)
    dict_database['Cap'] = deepcopy(dict_ref_InstCap)
#######################################################################

# 3b) create the nesting structure to iterate across:
    # future > scenario > region > country > yeat
    # WARNING! Inlcude only 1 future in this script, i.e., here we only produce future 0 inputs & outputs

# ... extracting the unique list of future...
scenarios_cases_list = params_tier2['scenarios_cases']
scenario_list = list(set(df3_scen['Scenario'].tolist()))
scenario_list.remove('ALL')
scenario_list.sort()
scenarios_exceptions = ['ACELERADO', 'ASPIRACIONAL']
for scen_del in scenarios_exceptions:
    if scen_del in scenario_list:
        scenario_list.remove(scen_del)
dict_test_transport_model = {}

# Function #2
def filter_dataframe(df, case, scenario=None, scenario_2=None, scenario_3=None,\
                    column='Scenario', column_2=None, column_3=None):
    """
    Filter a DataFrame based on the specified case.

    Agrs:
    - df (DataFrame): DataFrame to be filtered.
    - case (str): The case to apply ('scenario_full', 'scenario_simple', 
                'two_columns_scenarios', 'three_columns_scenarios', 
                'three_columns_scenarios_2', 'two_columns_scenarios_special').
    - scenario (str): The scenario name.
    - scenario_2 (str): The second scenario name.
    - scenario_3 (str): Third scenario name.
    - column (str): The column name to apply the filter, default is 'Scenario'.
    - column_2 (str): The second column name to apply the filter.
    - column_3 (str): The third column name to apply the filter.
    
    Returns:
    - df_filtered (DataFrame): Filtered DataFrame.
    """
    if case == 'scenario_full':
        mask = (df[column] == scenario) | (df[column] == scenario_2)
    elif case == 'scenario_simple':
        mask = df[column] == scenario
    elif case == 'two_columns_scenarios':
        mask = (df[column] == scenario) & (df[column_2] == scenario_2)
    elif case == 'two_columns_scenarios_special':
        mask = ((df[column].isin(['Fuel prices sales through pipeline','Fuel prices sales liquified'])) & (df[column_2] == scenario_2))
    elif case == 'three_columns_scenarios':        
        mask = (df[column] == scenario) & \
                    (df[column_2] == scenario_2) & (df[column_3] == scenario_3)
    elif case == 'three_columns_scenarios_2':        
        mask = (df[column] == scenario) | \
                    (df[column_2] == scenario_2) & (df[column_3] == scenario_3)
    else:
        raise ValueError("Invalid case specified")
        
    df_filtered = df.loc[mask]
    df_filtered.reset_index(drop=True, inplace=True)        
    return df_filtered

# Function #3
def process_types_and_df(df, case, time_vector, column='Type', column_2='Projection'):
    """
    Process and extract specific data from a DataFrame based on the given case.
    
    Args:
    - df (DataFrame): The DataFrame to be processed. It should contain relevant data for the given case and columns.
    - case (str): A string indicating the specific case to be applied. Possible values include 'only_types', 'only_types_2', 'df', 'df_types_3', etc. Each case defines a different method of processing and extracting data.
    - time_vector (list): A list representing time points, typically years, used for selecting data within the DataFrame.
    - column (str, optional): The name of the first column to be used in the processing. This parameter allows for specifying which column to filter or extract data from. Defaults to 'Type'.
    - column_2 (str, optional): The name of the second column to be used in the processing. This parameter further customizes the data extraction process. Defaults to 'Projection'.
    
    Returns:
    - Depending on the specified case, this function returns a tuple of values. The composition and types of these return values vary:
        - For cases that end with '.tolist()', the data type is 'list'.
        - For cases that end with '.iloc[]', the data type is the type of the DataFrame element at the specified index, typically 'str'.
    """
    if case == 'only_types':
        types = df[column].tolist()
        proj = df[column_2].tolist()
        by_vals = df[time_vector[0]].tolist()
        return types, proj, by_vals
    elif case == 'only_types_2':
        types = df[column].tolist()
        proj = df[column_2].iloc[0]
        by_vals = df[time_vector[0]].iloc[0]
        return types, proj, by_vals
    elif case == 'df':
        by = df[time_vector[0]].iloc[0]
        proj = df[column_2].iloc[0]
        return by, proj
    elif case == 'df_types_3':  
        by = df[time_vector[0]].iloc[0]
        proj = df[column_2].iloc[0]
        by_vals = df[time_vector[0]].iloc[0]
        return by, proj, by_vals
    elif case == 'df_types_4':
        by = df[time_vector[0]].iloc[0]
        proj = df[column].iloc[0]
        types_proj = df[column].tolist()
        by_vals = df[time_vector[0]].iloc[0]
        return by, proj, types_proj, by_vals      
    else:
        raise ValueError("Invalid case specified")

# Function #5        
def initialize_all_vals_dict(df, time_vector, case):
    """
    Initialize and populate a dictionary with values from a DataFrame based on a specified case.
    
    Args:
    - df (DataFrame): DataFrame from which values are to be extracted.
    - time_vector (lsit): List of years or time points used for initialization.
    - case (str): A string indicating the case ('tolist', 'iloc0', etc.) for processing the DataFrame
                    values.
    Returns:
    - all_vals_dict (dict): Returns a dictionary with keys corresponding to the elements in time_vector and values based on the specified case. The dictionary is populated with data from the DataFrame according to the chosen processing method.
    """                
    all_vals_dict = {}
    for year in time_vector:
        if case == 'first':
            all_vals_dict[year] = df[year].tolist()
        elif case == 'second' or case=='third' or case=='fourth':
            all_vals_dict[year] = df[year].iloc[0]
        else:
            raise ValueError("Invalid case specified. Choose 'tolist' or 'iloc0'.")
    return all_vals_dict

# Function #6
def calculate_grow_gdp_pc(base_value, gdp_pc_growth_vals, pop_vector, time_vector, imp_proj_dict, this_live, case):
    """
    Calculate projected values based on GDP per capita growth, population, and other factors under various cases.
    
    Args:
    - base_value (list): The base value for calculations in the first year.
    - gdp_pc_growth_vals (list): A list of GDP per capita growth values corresponding to the years in time_vector.
    - pop_vector (list): A list of population values corresponding to the years in time_vector.
    - time_vector (list): A list of years for which the calculations are to be made.
    - imp_proj_dict (dict): A dictionary to be updated with the calculated projections.
    - this_live (list): A variable representing the current livestock type or category under consideration.
    - case (str): A string indicating the specific case for calculation, which determines how the values are computed and returned.
    
    Returns:
    - total_imp_list (type): Returns a list of calculated values. The specific values and their format depend on the chosen case and the input parameters.
    """
    total_imp_list = []
    gen_imp_pc = []
    for y, year in enumerate(time_vector):
        gdp_pc_gr = gdp_pc_growth_vals[y]/100
        if y == 0:
            if case=='fourth':
                gen_imp_pc.append(base_value)
                total_imp_list.append(base_value*pop_vector[0]*1e6*365)
            else:
                gen_imp_pc.append(base_value/pop_vector[0]) 
                total_imp_list.append(base_value)
        else:
            next_val_gen_pc = gen_imp_pc[-1] * (1 + gdp_pc_gr)
            gen_imp_pc.append(next_val_gen_pc)
            next_val_total = next_val_gen_pc*pop_vector[y]
            if case=='fourth':
                total_imp_list.append(next_val_total*1e6*365)
            else:
                total_imp_list.append(next_val_total)
    if case=='full':
        imp_proj_dict[this_live] = total_imp_list
    return total_imp_list

# Function #7
def generate_imp_proj_dict(time_vector, df, types_livestock, types_projection, types_by_vals_imp, gdp_pc_growth_vals, pop_vector):
    """
    Generate an import projection dictionary based on various types of projections and livestock data.
    
    Args:
    - time_vector (list): List of years for which the projections are to be made.
    - df (DataFrame): DataFrame containing import data across different years and categories.
    - types_livestock (list): List of different types of livestock to be considered in the projection.
    - types_projection (list): List of projection types corresponding to each livestock type.
    - types_by_vals_imp (list): List of base import values for each type of livestock.
    - gdp_pc_growth_vals (list): List of GDP per capita growth values used in projections.
    - pop_vector (list): List of population values for each year in the time vector.
    
    Returns:
    - imp_proj_dict (dict): Dictionary containing import projections for each type of livestock, keyed by livestock type.
    """
    imp_proj_dict = {}
    all_vals_dict = initialize_all_vals_dict(df, time_vector, 'first')

    for i, this_live in enumerate(types_livestock):
        this_proj = types_projection[i]
        this_by_val = types_by_vals_imp[i]

        if this_proj == 'grow_gdp_pc':
            total_imp_list = calculate_grow_gdp_pc(this_by_val, gdp_pc_growth_vals, pop_vector, time_vector, imp_proj_dict, this_live, 'full')
        elif this_proj == 'flat':
            this_val_list = [this_by_val] * len(time_vector)
            imp_proj_dict.update({this_live: total_imp_list})
        elif this_proj == 'user_defined':
            for y in range(len(time_vector)):
                this_val_list.append(all_vals_dict[time_vector[y]][i])
            imp_proj_dict.update({this_live: total_imp_list}) 

    return imp_proj_dict

# Function #8
def calculate_projection(df, projection_type, base_value, gdp_pc_growth_vals, pop_vector, time_vector, case, df2=None):
    """
    Calculate projection values based on specified projection types and given economic and population data.
    
    Args:
    - df (DataFrame): DataFrame with user-defined projection values.
    - projection_type (str): Type of projection to calculate ('grow_gdp_pc', 'user_defined', or 'flat').
    - base_value (list): Base value for the first year in the case of 'grow_gdp_pc' projection.
    - gdp_pc_growth_vals (list): GDP per capita growth values for 'grow_gdp_pc' projection.
    - pop_vector (list): List of population values for each year in the
                        time vector, used in 'grow_gdp_pc' projection.
    - time_vector (list): List of years for which the projection is to be calculated.
    - case (str): Case identifier to determine the method of calculation.
    - df2 (DataFrame): Additional DataFrame used in some projection cases.
    Returns:
    
    - all_vals_gen_pc_dict (dict): A dictionary with calculated values for each year, structured based on the specified case.
    - total_list (list): List of total projection values for each year in the time vector.
    """ 
    all_vals_gen_pc_dict = initialize_all_vals_dict(df, time_vector, case)
    
    if projection_type == 'grow_gdp_pc':
        total_list = calculate_grow_gdp_pc(base_value, gdp_pc_growth_vals, pop_vector, time_vector, {}, None, case)

    elif projection_type == 'user_defined':
        total_list = []
        if case=='first' or case=='third':
            total_list = [all_vals_gen_pc_dict[year] * pop_vector[y] for y, year in enumerate(time_vector)]
        elif case=='fourth':
            total_list = [all_vals_gen_pc_dict[year] * pop_vector[y] * 1e6 * 365 for y, year in enumerate(time_vector)]
        else:
            total_list = [df2[year].iloc[0] for year in time_vector]
    elif projection_type == 'flat':
        if case=='fourth':
            total_list = []
            gen_pc = []
            for y in range(len(time_vector)):
                gen_pc.append(base_value)
                total_list.append(gen_pc[-1]*pop_vector[y]*1e6*365)
        else:
            total_list = [base_value] * len(time_vector)

    return all_vals_gen_pc_dict, total_list

# Function #9
def generate_projection_dict(types, projections, base_values, df, time_vector, case):
    """
    Generate a projection dictionary based on different projection types.
    
    Args:
    - types (list): List of types of categories.
    - projections (list): List of projection types for each category.
    - base_values (list): List of base values for each category.
    - df (DataFrame): Dataframe with all values for each year.
    - time_vector (list): List of years in the time vector.
    - case (str): The case identifier to determine the specific method of calculation or adjustment to be applied.
    
    Returns:
    - projection_dict (dict): Dictionary with projections for each category, keyed by category type. The values are lists, representing projected values for each year in the time vector.
    """
    values_dict = initialize_all_vals_dict(df, time_vector, case)
    projection_dict = {}

    for i, this_type in enumerate(types):
        this_proj = projections[i]
        this_by_val = base_values[i]
        this_val_list = []

        if this_proj == 'flat':
            this_val_list = [this_by_val] * len(time_vector)
        elif this_proj == 'user_defined':
            this_val_list = [values_dict[year][i] for year in time_vector]

        projection_dict[this_type] = this_val_list

    return projection_dict

# Function #10
def generate_estimation_dicts(types, proj_dict, additional_list, time_vector, r_rate, r_year, case):
    """
    Calculate projections and discounted values for different cases, including enteric fermentation emissions.
    
    Args:
    - types (list): List of types (e.g., types_sge_fopex, types_rice_opex, types_livestock).
    - proj_dict (dict): Dictionary containing projection data.
    - additional_list (list): List containing additional data for calculations.
    - time_vector (list): List of years.
    - r_rate (int): Discount rate.
    - r_year (int): Reference year for discounting.
    - case (str): Case identifier ('first', 'second', 'third', or 'fourth').
    
    Returns:
    - out_proj_dict (dict): Dictionary with projections for each type.
    - out_proj_dict_disc (dict): Dictionary with discounted projections for each type.
    """
    out_proj_dict = {}
    out_proj_dict_disc = {}

    for type_item in types:
        proj_list = []
        proj_disc_list = []

        for y, year in enumerate(time_vector):
            disc_constant = 1 / ((1 + r_rate / 100) ** (float(year) - r_year))
            value = proj_dict[type_item][y]
                
            if case == 'second':
                additional_value = additional_list[y] / 1e6
            elif case == 'third':
                additional_value = additional_list[y]
            elif case == 'fourth':
                additional_value = additional_list[type_item][y] / 1e6
            
            if case == 'first':
                proj_value = value * additional_list[type_item][y] / 1e6
                proj_disc_value = value * additional_list[type_item][y] *\
                    disc_constant / 1e6
            else:
                proj_value = value * additional_value
                proj_disc_value = proj_value * disc_constant

            proj_list.append(proj_value)
            proj_disc_list.append(proj_disc_value)

        out_proj_dict[type_item] = proj_list
        out_proj_dict_disc[type_item] = proj_disc_list
        
    return out_proj_dict, out_proj_dict_disc

# Function #11
def generate_short_projection(types, projections, base_values, time_vector):
    """
    Generate projections for SGE CAPEX or VOPEX based on different projection types, with an option to iterate over values.
    
    Args:
    - types (list): List of types of SGE (CAPEX or VOPEX).
    - projections (list): List of projection types for each SGE type.
    - base_values (list): List of base values for each SGE type.
    - time_vector (list): List of years in the time vector.
    
    Returns:
    - sge_proj_dict (dict): Dictionary containing projections for each SGE type.
    """
    sge_proj_dict = {}
    
    for i, pro_type in enumerate(types):
        projection_type = projections[i]
        base_value = base_values[i]
        
        if projection_type == 'flat':
            value_list = [base_value] * len(time_vector)
        else:
            # Implement other projection types if needed
            value_list = [0] * len(time_vector)  # Default to zero for unsupported types
        
        sge_proj_dict[pro_type] = value_list

    return sge_proj_dict

# Function #12
def calculate_emissions(df, fuels, time_vector):
    """
    Calculate emissions for a list of fuels based on a flat projection.
    
    Args:
    - df (DataFrame): DataFrame containing emission values and projections.
    - fuels (list): List of fuels to calculate emissions for.
    - time_vector (list): List of years in the time vector.
    
    Returns:
    - emissions_fuels_list (list): List of fuels for which emissions were calculated.
    - emissions_fuels_dict (dict): Dictionary with emissions for each fuel, keyed by fuel type.
    """
    emissions_fuels_list = []
    emissions_fuels_dict = {}

    for fuel in fuels:
        fuel_row = df.loc[df['Fuel'] == fuel]
        if not fuel_row.empty:
            this_proj = fuel_row['Projection'].values[0]
            base_emission_val = fuel_row[time_vector[0]].values[0]

            if this_proj == 'flat':
                list_emission_year = [base_emission_val for _ in range(len(time_vector))]

            emissions_fuels_list.append(fuel)
            emissions_fuels_dict[fuel] = list_emission_year

    return emissions_fuels_list, emissions_fuels_dict

# Function #13
def calculate_ref_values(types, dict1, dict2, additional_list=None, calculation_type='emissions'):
    """
    Calculate reference values (either emissions or prices) for different types.
    
    Args:
    - types (list): List of types (e.g., types_shares_ref).
    - dict1 (dict): First dictionary containing data (e.g., rac_GWP_out_dict or rac_price_out_dict).
    - dict2 (dict): Second dictionary containing data (e.g., ref_dem_by_type_dict).
    - additional_list (list): Additional list for emissions calculation.
    - calculation_type (str): Type of calculation to perform ('emissions' or 'prices').
    
    Returns:
    - ref_out_dict (dict): Dictionary with calculated reference values for each type.
    """
    ref_out_dict = {}

    for type_item in types:
        ref_values = []

        for i in range(len(dict2[type_item])):
            value1 = dict1[type_item][i]
            value2 = dict2[type_item][i]

            if calculation_type == 'emissions':
                lf_value = additional_list[i]
                calculated_value = (value1 * value2 * lf_value) / 1e3
            elif calculation_type == 'prices':
                calculated_value = (value1 * value2) / 1e3
            else:
                raise ValueError("Invalid calculation type. Choose 'emissions' or 'prices'.")

            ref_values.append(calculated_value)

        ref_out_dict[type_item] = ref_values

    return ref_out_dict

# Function #14
def calculate_values(types_list, dict1, dict2, total_list, additional_list=None, calculation_type='demand'):
    """
    Generic function to calculate demand, emissions, or prices.
    
    Args:
    - types_list (list): List of types.
    - dict1 (dict): Dictionary with shares, emission factors, or prices.
    - dict2 (dict): Dictionary with demand values (used only for emissions and prices calculations).
    - total_list (list): List of total demand values.
    - additional_list (list, optional): Additional list for emissions calculation (e.g., total_lf_list).
    - calculation_type (str): Type of calculation ('demand', 'emissions', 'prices').
    Returns:
    
    - calculated_values (dict): Dictionary with calculated values for each type in the types_list. The nature of these values depends on the specified calculation type (demand, emissions, or prices).
    """

    calculated_values = {}
    for type_item in types_list:
        values = []
        for y in range(len(total_list)):
            if calculation_type == 'demand':
                value = total_list[y] * dict1[type_item][y]
            elif calculation_type == 'emissions':
                value = (dict1[type_item][y] * dict2[type_item][y] * additional_list[y]) / 1e3
            elif calculation_type == 'prices':
                value = (dict1[type_item][y] * dict2[type_item][y]) / 1e3
            values.append(value)
        calculated_values[type_item] = values
    return calculated_values

# Function #15
def calculate_investment_requirements(total_list, life_list, time_vector):
    """
    Calculate the delta (change) in investment requirements based on life span.
    
    Args:
    - total_list (list): List of total values for each year, indicating the overall investment amounts.
    - life_list (list): List indicating the life span of investments, used to adjust investment calculations.
    - time_vector (list): List of years for which the investment requirements are calculated.
    
    Returns:
    - delta_list (list): List of delta values for investment requirements, showing the yearly change in investment needs.
    """
    delta_list = [0]
    # Calculate yearly deltas
    for y in range(1, len(time_vector)):
        delta_list.append(total_list[y] - total_list[y-1])

    # Adjust deltas based on life span
    for y in range(int(life_list[0]), len(time_vector)):
        if y - int(life_list[y]) >= 0:
            delta_list[y] += delta_list[y - int(life_list[y])]

    return delta_list

# Function #16
def calculate_investment_requirements_2(delta_list, olife_list, varcost_list, fcost_list, capex_list, time_vector, grs_rs_list):
    """
    Calculate investment requirements, operating expenses, and capital expenses based on delta values and life span of investments.
    
    Args:
    - delta_list (list): List of delta values for each year, representing changes in investment requirements.
    - olife_list (list): List indicating the operational life span of investments.
    - varcost_list (list): List of variable costs associated with each year.
    - fcost_list (list): List of fixed costs for each year.
    - capex_list (list): List of capital expenditure costs for each year.
    - time_vector (list): List of years for the calculation.
    - grs_rs_list (list): List of gross revenue service values for each year.
    
    Returns:
    - opex (list): List of operating expenses calculated for each year.
    - capex (list): List of capital expenses calculated for each year.
    """
    delta_inv = [0] * len(time_vector)
    for y in range(len(time_vector)):
        if delta_list[y] > 0:
            delta_inv[y] = delta_list[y]
        olife = int(olife_list[y])
        if y >= olife and delta_list[y-olife] > 0:
            delta_inv[y] += delta_list[y-olife]

    list_res_rs_opex = [varcost + fcost for varcost, fcost in zip(varcost_list, fcost_list)]
    opex = [(ucost * act)/1e9 for ucost, act in zip(list_res_rs_opex, grs_rs_list)]
    capex = [(ucost * act)/1e9 for ucost, act in zip(capex_list, delta_inv)]

    return opex, capex

# Function #17
def process_vectors(avector, dict_energy_demand, df32_sh_cat, dict_energy_demand_by_fuel, time_vector, list32_depu, avector_counter, case):
    """
    Process and return vectors based on the given avector and other parameters for energy demand and depuration.
    
    Args:
    - avector (str): The vector to be processed, representing a category of energy demand.
    - dict_energy_demand (dict): Dictionary containing energy demand data.
    - df32_sh_cat (str): Dataframe column or similar structure for selection.
    - dict_energy_demand_by_fuel (dict): Dictionary containing energy demand by fuel type.
    - time_vector (list): List of time intervals for the calculation.
    - list32_depu (list): Dataframe containing depuration data.
    - avector_counter (int): Counter or index for accessing elements in list32_depu.
    - case (str): Case identifier for subtraction ('first', 'second').
    
    Returns:
    - base_disp_vector (list): List of base displacement values for the given avector.
    - base_disp_vector_agg (list): List of aggregated base displacement values.
    - subtract_vector (list): List of values to be subtracted from the base vector.
    - loc_depu (type): Local depuration value for the given avector.
    - avector_orig (str): Original avector value, useful in case modifications were made based on the case.
    """
    # Check if avector is not 'none' and assign values accordingly
    if avector != 'none':
        if case=='second':
            if avector == 'Hydrogen':
                avector_orig = 'Hydrogen'
                avector = 'Electricity'
            else:
                avector_orig = avector
        base_disp_vector = deepcopy(dict_energy_demand[df32_sh_cat][avector])
        base_disp_vector_agg = deepcopy(dict_energy_demand_by_fuel[avector])
    else:
        base_disp_vector = [0] * len(time_vector)
        base_disp_vector_agg = [0] * len(time_vector)

    # Initialize subtract_vector
    subtract_vector = [0] * len(time_vector)

    # Retrieve location depuration value
    loc_depu = list32_depu[avector_counter]

    if case=='second' and avector != 'none':
        return base_disp_vector, base_disp_vector_agg, subtract_vector, loc_depu, avector_orig
    elif case=='second' and avector=='none':
        return base_disp_vector, base_disp_vector_agg, subtract_vector, loc_depu, 'none'
    else:
        return base_disp_vector, base_disp_vector_agg, subtract_vector, loc_depu

# Function #18
def process_target_years(time_vector, df32_ndby, df32_target_year_str, df32_dvty_list, base_disp_vector, case, subtract_vector, sh_subcat=None):
    """
    Process multiple target years and perform necessary calculations based on displacement values and base year data.
    
    Args:
    - time_vector (list): List of time intervals for the calculation.
    - df32_ndby (int): The base year for comparison.
    - df32_target_year_str (str): String of target years separated by semicolons, used to identify specific years for processing.
    - df32_dvty_list (list): List of displacement values for target years.
    - base_disp_vector (list): Base displacement vector, used as a reference for calculations.
    - case (str): Case identifier for the calculation method ('first', 'second').
    
    Returns:
    - by_idx (int): Index of the base year within the time vector.
    - ty_list_str (list of str): List of target years as strings.
    - ty_list (list of int): List of target years as integers.
    - ty_index_list (list of int): List of indexes corresponding to target years within the time vector.
    """
    by_idx = time_vector.index(df32_ndby)
    ty_list_str = df32_target_year_str.split(';')
    ty_list = [int(x) if x.isdigit() else x for x in ty_list_str]
    ty_index_list = []
    ty_counter = 0

    for ty in ty_list:
        ty_index = time_vector.index(ty)
        ty_index_list.append(ty_index)


        if case=='first':
            subtract_vector[ty_index] = df32_dvty_list[ty_counter] / 1e9
            if base_disp_vector[ty_index] - df32_dvty_list[ty_counter] / 1e9 < 0:
                print('The subtraction suggestion is not acceptable! Lower the displacement value.')
                sys.exit()
            ty_counter += 1
            return by_idx, ty_list_str, ty_list, ty_index_list
        elif case=='second':
            tval = df32_dvty_list[ty_counter]/100
            subtract_vector[ty_index] = base_disp_vector[ty_index]*sh_subcat*tval
            ty_counter += 1
            return by_idx, ty_list_str, ty_list, ty_index_list, tval

# Function #19
def fill_subtract_vector(time_vector, by_idx, ty_list, ty_index_list, df32_dvbfty, df32_dvbetty, df32_nralty, subtract_vector):
    """
    Fill the subtract vector based on various conditions.

    Args:
    - time_vector (list): List of time intervals.
    - by_idx (int): Index of the base year.
    - ty_list (list): List of target years.
    - ty_index_list (list): List of indices for the target years.
    - df32_dvbfty (str): Condition for before first target year.
    - df32_dvbetty (str): Condition for between target years.
    - df32_nralty (str): Condition for after last target year.
    - subtract_vector (list): The vector to be filled.
    """
    subsy_counter, ty_idx_counter = 0, 0
    for subsy in time_vector:
        if subsy_counter <= by_idx:
            pass  # keep the list as zero
        elif subsy_counter > by_idx and subsy < ty_list[0] and df32_dvbfty == 'interp':
            delta_den = ty_index_list[0] - by_idx
            delta_num = subtract_vector[ty_index_list[0]]
            delta = delta_num / delta_den
            subtract_vector[subsy_counter] = subtract_vector[subsy_counter-1] + delta
        elif subsy in ty_list:
            ty_idx_counter += 1
        elif subsy_counter > by_idx and subsy > ty_list[0] and subsy < ty_list[-1] and df32_dvbetty == 'interp':
            tys_idxs_last = ty_index_list[ty_idx_counter - 1]
            tys_idxs_this = ty_index_list[ty_idx_counter]
            delta_den = tys_idxs_this - tys_idxs_last
            delta_num = subtract_vector[ty_index_list[ty_idx_counter]] - subtract_vector[ty_index_list[ty_idx_counter - 1]]
            delta = delta_num / delta_den
            subtract_vector[subsy_counter] = subtract_vector[subsy_counter-1] + delta
        elif subsy_counter > by_idx and subsy > ty_list[-1] and df32_nralty == 'flat':
            subtract_vector[subsy_counter] = subtract_vector[subsy_counter - 1]
        else:
            print('No specified interpolation condition found for displacement formulas!')
            sys.exit()
        subsy_counter += 1

# Function #20
def create_add_vectors(avector, df32_nerpu, loc_depu, subtract_vector, add_vector_general, case, df32_substitute_orig=None, list_h2_effic=None):
    """
    Create add vectors based on substitution ratios, adjusting energy replacement values and depuration values.
    
    Args:
    - avector (str): The vector to be processed, indicating the category or type of energy being analyzed.
    - df32_nerpu (float): Energy replacement unit, used to calculate the substitution factor.
    - loc_depu (float): Local depuration value, used to adjust the substitution factor.
    - subtract_vector (list): The vector from which values are subtracted, representing the amount of energy to be replaced.
    - add_vector_general (list): General add vector that is updated based on the calculated substitution.
    - case (str): Case identifier for the method of calculation ('first', 'second').
    - df32_substitute_orig (type, optional): Original substitute vector, used in specific cases like 'Hydrogen'.
    - list_h2_effic (list, optional): List of hydrogen efficiencies, used when 'Hydrogen' is the substitute.
    
    Returns:
    - add_vector_general (list): Updated general add vector after applying substitution ratios.
    - add_vector_local (list): Local add vector calculated based on the substitution factor and specific cases.
    """
    if case=='first':
        if avector != 'none':
            rep_fac_ene = df32_nerpu / loc_depu
            add_vector_local = [rep_fac_ene * substval for substval in subtract_vector]
        else:
            print('This functionality is not available')
            sys.exit()
    elif case=='second':
            rep_fac_ene = df32_nerpu/loc_depu
            add_vector_local = [rep_fac_ene*substval for substval in subtract_vector]
            if df32_substitute_orig == 'Hydrogen':
                add_vector_local = [v/(list_h2_effic[i]/100) for i, v in enumerate(add_vector_local)]

    add_vector_general = [a + b for a, b in zip(add_vector_general, add_vector_local)]
    return add_vector_general, add_vector_local

# Function #21
def update_disp_vectors(base_disp_vector, base_disp_vector_agg, subtract_vector):
    """
    Update displacement vectors by subtracting specified values, used in energy demand and supply calculations.
    
    Args:
    - base_disp_vector (list): Base displacement vector, representing original values.
    - base_disp_vector_agg (list): Aggregate base displacement vector, representing aggregated values.
    - subtract_vector (list): Vector from which values are subtracted, representing the adjustment to be made.
    
    Returns:
    - changed_disp_vector (list): Updated displacement vector after subtraction.
    - change_disp_vector_agg (list): Updated aggregate displacement vector after subtraction.
    """
    changed_disp_vector = [a - b for a, b in zip(base_disp_vector, subtract_vector)]
    change_disp_vector_agg = [a - b for a, b in zip(base_disp_vector_agg, subtract_vector)]
    return changed_disp_vector, change_disp_vector_agg

# Function #22
def update_energy_demand_dicts(df32_substituted, df32_substitute, df32_sh_cat, avector, changed_disp_vector, change_disp_vector_agg, dict_energy_demand, dict_energy_demand_by_fuel, case):
    """
    Update energy demand dictionaries.

    Aegs:
    - df32_substituted (str): The substituted energy source.
    - df32_substitute (str): The substitute for the energy source.
    - df32_sh_cat (str): Category in the energy demand dictionary.
    - avector (str): Vector being processed.
    - changed_disp_vector (list): Changed displacement vector.
    - change_disp_vector_agg (list): Changed aggregate displacement vector.
    - dict_energy_demand (dict): Dictionary of energy demand.
    - dict_energy_demand_by_fuel (dict): Dictionary of energy demand by fuel.
    - case (str): Case of subtraction ('first', 'second').
    """
    if case=='first':
        if df32_substituted != df32_substitute:
            dict_energy_demand[df32_sh_cat][avector] = deepcopy(changed_disp_vector)
            dict_energy_demand_by_fuel[avector] = deepcopy(change_disp_vector_agg)
        else:
            print('Happens for increase in energy demand.')
    elif case=='second':
            dict_energy_demand[df32_sh_cat][avector] = deepcopy(changed_disp_vector)
            dict_energy_demand_by_fuel[avector] = deepcopy(change_disp_vector_agg)        

# Function #23
def assign_values_for_indices(source_list, target_list, indices):
    """
    Assign values from the source list to the target list at specified indices.

    Args:
    - source_list (list): List from which to source the values.
    - target_list (list): List to which values will be assigned.
    - indices (list): List of indices for which values need to be assigned.

    Returns:
    - target_list (list): Updated target list after the assignment.
    """
    for idx in indices:
        target_list[idx] = source_list[idx]
    return target_list
            

# ... we will work with a single dictionary containing all simulations:
# RULE: most values are time_vector-long lists, except externality unit costs (by country):
ext_by_country = {}
count_under_zero = 0
base_year = str(per_first_yr)

print('PROCESS 1 - RUNNING THE SIMULATIONS')
dict_scen = {}  # fill and append to final dict later
idict_net_cap_factor_by_scen_by_country = {}
store_percent_BAU = {}

# Here we must create an empty dictionary with the regional consumption
list_fuel_ALL = list(set([v for v in list(set(df3_scen['Fuel'].tolist()))]))
list_fuel_ALL = [i for i in list_fuel_ALL if type(i) is str]
dict_energy_demand_by_fuel_sum = {
    k: [0] * len(time_vector) for k in list_fuel_ALL}

for s in range(len(scenario_list)):
    this_scen = scenario_list[s]
    # print('# 1 - ', this_scen)

    dict_test_transport_model.update({this_scen:{}})

    dict_local_reg = {}
    idict_net_cap_factor_by_scen_by_country.update({this_scen:{}})
    
    # We need to store a dictionary for each country that store production
    # from inputted capacity and capacity factors for Anomaly identification
    dict_store_res_energy_orig = {}

    for r in range(len(regions_list)):
        this_reg = regions_list[r]
        #print('   # 2 - ', this_reg)

        country_list = dict_regs_and_countries[this_reg]
        country_list.sort()
        dict_local_country = {}

        # Add a filter to include countries with transport data only:
        country_list = [c for c in country_list if c in tr_list_app_countries_u]
        #country_list = ['Mexico']

        for c in range(len(country_list)):
            this_country = country_list[c]
            #print('      # 3 - ', this_country)

            dict_test_transport_model[this_scen].update({this_country:{}})

            # ...store the capacity factor by country:
            idict_net_cap_factor_by_scen_by_country[this_scen].update({this_country:{}})

            # ...call the GDP of the base year
            this_gdp_base = gdp_dict[this_country]

            # ...call and make population projection
            this_pop_base = popbase_dict[this_country]
            this_pop_final = popfinal_dict[this_country]
            this_pop_proj = popproj_dict[this_country]
            this_pop_vector_known = ['' for y in range(len(time_vector))]
            this_pop_vector_known[0] = this_pop_base
            this_pop_vector_known[-1] = this_pop_final
            if this_pop_proj == 'Linear':
                this_pop_vector = \
                    interpolation_to_end(time_vector, ini_simu_yr, \
                        this_pop_vector_known, 'last', 'Population')


            # ...subselect the scenario dataframe you will use
            df_scen = filter_dataframe(df3_scen, 'scenario_full', scenario=this_scen, scenario_2='ALL')

            indices_df_scen = df_scen.index.tolist()
            list_application_countries_all = \
                df_scen['Application_Countries'].tolist()
            list_application_countries = \
                list(set(df_scen['Application_Countries'].tolist()))

            for ac in list_application_countries:
                if this_country in ac.split(' ; '):
                    select_app_countries = deepcopy(ac)

            indices_df_scen_select = [i for i in range(len(indices_df_scen))
                                      if (list_application_countries_all[i]
                                          == select_app_countries) or
                                         (list_application_countries_all[i]
                                          == 'ALL') or
                                         (this_country in
                                          list_application_countries_all[i])
                                          ]

            df_scen_rc = df_scen.iloc[indices_df_scen_select]
            df_scen_rc.reset_index(drop=True, inplace=True)

            # 3c) create the demands per fuel per country in a single dictionary (follow section 2 for structure)
            # This depends on the EB dictionary, the GDP, and the scenarios' <Demand energy intensity>
            # From EB, extract the elements by demand and fuel:

            this_country_2 = dict_equiv_country_2[this_country]

            dict_base_energy_demand = \
                dict_database['EB'][this_reg][this_country_2]['Energy consumption']
            list_demand_sector_techs = list(dict_base_energy_demand.keys())
            list_demand_sector_techs.remove('none')

            list_fuel_raw = list(dict_base_energy_demand['none'].keys())
            list_fuel = [e for e in list_fuel_raw if ('Total' not in e and
                                                      'Non-' not in e)]

            # We must now create a dictionary with the parameter, the technology, and the fuel.
            # By default, demand technologies consume the fuel.
            param_related = 'Demand energy intensity'  # this is in "scenarios"
            param_related_2 = 'GDP growth'  # this is in "scenarios"
            param_related_3 = 'Distribution of end-use consumption'  # this is in "scenarios"

            # Select the "param_related"
            df_param_related = filter_dataframe(df_scen_rc, 'scenario_simple', scenario=param_related, column='Parameter')
            # Select the "param_related_2"
            df_param_related_2 = filter_dataframe(df_scen_rc, 'scenario_simple', scenario=param_related_2, column='Parameter')
            # Select the "param_related_3"          
            df_param_related_3 = filter_dataframe(df_scen_rc, 'scenario_simple', scenario=param_related_3, column='Parameter')
            # ...select an alternative "param_related_3" where scenarios can be managed easily
            df_scen_3 = filter_dataframe(df3_scen_dems, 'scenario_full', scenario=this_scen, scenario_2='ALL')

            indices_df_scen_spec = df_scen_3.index.tolist()
            list_application_countries_spec = \
                df_scen_3['Application_Countries'].tolist()

            indices_df_scen_select = [i for i in range(len(indices_df_scen_spec))
                                      if (list_application_countries_spec[i]
                                          == select_app_countries) or
                                         (list_application_countries_spec[i]
                                          == 'ALL') or
                                         (this_country in
                                          list_application_countries_spec[i])
                                          ]

            df_scen_3_spec = df_scen_3.iloc[indices_df_scen_select]
            df_scen_3_spec.reset_index(drop=True, inplace=True)  # this should be ready to use

            df_scen_3_2 = filter_dataframe(df3_scen_matrix, 'three_columns_scenarios_2', \
                                        scenario=this_scen, scenario_2='ALL', scenario_3=this_country, \
                                        column='Scenario', column_2='Scenario', column_3='Country') 

            # We may add other fuels that are not present in the original list fuel
            list_fuel_additional = [v for v in list(set(df_param_related_3['Fuel'].tolist())) if v not in list_fuel]
            list_fuel_orig = deepcopy(list_fuel)
            list_fuel = list_fuel_orig + list_fuel_additional     
            ###################################################################
            # ...acting for "GDP growth"
            this_gdp_growth_projection = df_param_related_2.iloc[0]['projection']
            this_gdp_growth_value_type = df_param_related_2.iloc[0]['value']
            this_gdp_growth_vals_raw = []
            this_gdp_growth_vals = []
            for y in time_vector:
                this_gdp_growth_vals_raw.append(df_param_related_2.iloc[0][y])
            if (this_gdp_growth_projection == 'flat' and
                    this_gdp_growth_value_type == 'constant'):
                for y in range(len(time_vector)):
                    this_gdp_growth_vals.append(this_gdp_growth_vals_raw[0])
            if this_gdp_growth_projection == 'user_defined':
                this_gdp_growth_vals = deepcopy(this_gdp_growth_vals_raw)

            # ...acting for GDP and GDP per capita:
            this_gdp_vals = []
            this_gdp_per_cap_vals = []
            this_gdp_pc_growth_vals = []
            this_pop_growth_vals = []
            for y in range(len(time_vector)):
                if y == 0:
                    this_gdp_vals.append(this_gdp_base)
                else:
                    this_growth = this_gdp_growth_vals[y]
                    next_year_gdp = this_gdp_vals[-1]*(1+this_growth/100)
                    this_gdp_vals.append(next_year_gdp)

                this_y_gdp_per_capita = \
                    this_gdp_vals[-1]/(this_pop_vector[y]*1e6)
                this_gdp_per_cap_vals.append(this_y_gdp_per_capita)
                if y != 0:
                    # Calculate the growth of the GDP per capita
                    gdp_pc_last = this_gdp_per_cap_vals[y-1]
                    gdp_pc_present = this_gdp_per_cap_vals[y]
                    this_gdp_pc_growth = \
                        100*(gdp_pc_present - gdp_pc_last)/gdp_pc_last
                    this_gdp_pc_growth_vals.append(this_gdp_pc_growth)

                    # Calculate the growth of the population
                    pop_last = this_pop_vector[y-1]
                    pop_present = this_pop_vector[y]
                    this_pop_growth = 100*(pop_present - pop_last)/pop_last
                    this_pop_growth_vals.append(this_pop_growth)
                else:
                    this_gdp_pc_growth_vals.append(0)
                    this_pop_growth_vals.append(0)

            # Create the energy demand dictionary (has a projection):
            dict_energy_demand = {}  # by sector
            dict_energy_intensity = {}  # by sector
            dict_energy_demand_by_fuel = {}  # by fuel
            tech_counter = 0
            for tech in list_demand_sector_techs:

                tech_idx = df_param_related['Demand sector'].tolist().index(tech)
                tech_counter += 1

                # ...acting for "Demand energy intensity" (_dei)
                this_tech_dei_df_param_related = df_param_related.iloc[tech_idx]
                this_tech_dei_projection = this_tech_dei_df_param_related['projection']
                this_tech_dei_value_type = this_tech_dei_df_param_related['value']
                this_tech_dei_known_vals_raw = []
                this_tech_dei_known_vals = []

                ref_energy_consumption = \
                    dict_base_energy_demand[tech]['Total'][base_year]

                y_count = 0
                for y in time_vector:
                    this_tech_dei_known_vals_raw.append(this_tech_dei_df_param_related[y])
                    # Act already by attending "endogenous" calls:
                    if this_tech_dei_df_param_related[y] == 'endogenous':
                        add_value = ref_energy_consumption*1e9/this_gdp_vals[y_count]  # MJ/USD
                    elif math.isnan(float(this_tech_dei_df_param_related[y])) is True and y_count >= 1:
                        add_value = ''
                    elif (float(this_tech_dei_df_param_related[y]) != 0.0 and
                            this_tech_dei_value_type == 'rel_by'):
                        add_value = \
                            this_tech_dei_known_vals[0]*this_tech_dei_df_param_related[y]
                    this_tech_dei_known_vals.append(add_value)
                    y_count += 1

                this_tech_dei_vals = \
                    interpolation_to_end(time_vector, ini_simu_yr, \
                        this_tech_dei_known_vals, 'last', '')

                # ...since we have the intensities, we can obtain the demands:
                this_tech_ed_vals = []
                y_count = 0
                for y in time_vector:
                    add_value = \
                        this_tech_dei_vals[y_count]*this_gdp_vals[y_count]/1e9  # PJ
                    this_tech_ed_vals.append(add_value)
                    y_count += 1

                # store the total energy demand:
                dict_energy_demand.update({tech:{'Total':this_tech_ed_vals}})
                dict_energy_intensity.update({tech:{'Total':this_tech_dei_vals}})

                # ...we can also obtain the demands per fuel, which can vary depending on the "apply_type"
                # we will now iterate across fuels to find demands:

                total_sector_demand_baseyear = 0  # this applies for a *demand technology*
                for loc_fuel in list_fuel:  # sum across al fuels for a denominator
                    if 'Total' not in loc_fuel and 'Non-' not in loc_fuel:
                        if loc_fuel in list_fuel_orig:
                            total_sector_demand_baseyear += \
                                dict_base_energy_demand[tech][loc_fuel][base_year]

                        if tech_counter == 1:  # we must calculate the total energy demand by fuel
                            zero_list = [0 for y in range(len(time_vector))]
                            dict_energy_demand_by_fuel.update({loc_fuel:zero_list})

                # ...these are variables are needed for internal distribution of demands
                check_percent = False
                store_fush = {}  # "fush"  means "fuel shares"
                store_fush_rem = {}

                # Here we need to add the None_reduction to the energy dictionary

                # Store a dictionary with the endogenous shares to use below:
                dict_endo_sh = {}

                for fuel in list_fuel:
                    #if 'Total' not in fuel and 'Other' not in fuel and 'Non-' not in fuel:
                    if 'Total' not in fuel and 'Non-' not in fuel:
                        fuel_idx = df_param_related_3['Fuel'].tolist().index(fuel)

                        # ...acting for "Distribution of end-use consumption" (_deuc)
                        this_fuel_deuc_df_param_related = df_param_related_3.iloc[tech_idx]
                        this_fuel_deuc_projection = this_fuel_deuc_df_param_related['projection']
                        this_fuel_deuc_value_type = this_fuel_deuc_df_param_related['value']
                        this_fuel_deuc_known_vals_raw = []
                        this_fuel_deuc_known_vals = []

                        # ...our goal here: obtain final demand by fuel:
                        this_tech_fuel_ed_vals = []

                        # ...here, seek the EB proportion and keep it constant using "fuel demand" (_fd)
                        # ...we also need to extract the total fuel demand, which is the correct denominator
                        total_fuel_demand_baseyear = 0
                        total_fuel_demand_baseyear_2 = 0
                        for tech_internal in list_demand_sector_techs:
                            if fuel in list_fuel_orig:
                                total_fuel_demand_baseyear += \
                                    dict_base_energy_demand[tech_internal][fuel][base_year]
                        #
                        if fuel in list_fuel_orig:
                            num_fd = dict_base_energy_demand[tech][fuel][base_year]
                        else:
                            num_fd = 0
                        if total_sector_demand_baseyear != 0:
                            den_fd = total_sector_demand_baseyear
                        else:
                            den_fd = 1
                        endo_share = num_fd/den_fd
                        
                        dict_endo_sh.update({fuel:endo_share})

                        if this_fuel_deuc_projection == 'keep_proportions' or this_fuel_deuc_projection == 'electrify_sector_2_max':
                            y_count = 0
                            for y in time_vector:
                                # ...start by summing across all demands:
                                add_value = \
                                    endo_share*this_tech_ed_vals[y_count]
                                this_tech_fuel_ed_vals.append(add_value)

                                # ...be sure to add the fuel demand too:
                                dict_energy_demand_by_fuel[fuel][y_count] += add_value

                                y_count += 1

                        elif this_fuel_deuc_projection == 'redistribute':  # here we need to change the demand relationships by sector, in a smooth manner (i.e., interpolate)
                            df_redistribute_data = filter_dataframe(df_scen_3_spec, 'two_columns_scenarios', \
                                                                    scenario=tech, scenario_2=fuel, \
                                                                    column='Demand sector', column_2='Fuel')

                            try:
                                if df_redistribute_data['value'].iloc[0] == 'percent':
                                    check_percent = True
                            except Exception:
                                check_percent = False

                            this_fush_known_vals_raw = []
                            this_fush_known_vals = []
                            if check_percent is True:  # is compatible with "interpolate"
                                for y in time_vector:
                                    try:
                                        add_value = \
                                            df_redistribute_data[y].iloc[0]
                                    except Exception:
                                        add_value = 0

                                    this_fush_known_vals_raw.append(add_value)
                                    if type(add_value) is int:
                                        if math.isnan(add_value) is False:
                                            this_fush_known_vals.append(add_value/100)
                                        else:
                                            pass
                                    elif str(y) == str(base_year):
                                        this_fush_known_vals.append(endo_share)
                                    elif str(y) <= str(ini_simu_yr):
                                        this_fush_known_vals.append(endo_share)
                                    else:
                                        this_fush_known_vals.append('')

                                if add_value != 'rem':
                                    this_fush_vals = \
                                        interpolation_to_end(time_vector, 
                                                             ini_simu_yr,
                                                             this_fush_known_vals,
                                                             'last',
                                                              '')

                                else:  # we need to fill later                               
                                    this_fush_vals = \
                                        [0 for y in range(len(time_vector))]
                                    store_fush_rem.update({fuel:this_fush_vals})
                                store_fush.update({fuel:this_fush_vals})

                                y_count = 0
                                for y in time_vector:
                                    add_value = \
                                        this_fush_vals[y_count]*this_tech_ed_vals[y_count]
                                    this_tech_fuel_ed_vals.append(add_value)
                                    dict_energy_demand_by_fuel[fuel][y_count] += add_value

                                    y_count += 1

                            if check_percent is not True:  # should do the same as in 'keep_proportions'
                                y_count = 0
                                for y in time_vector:
                                    add_value = \
                                        endo_share*this_tech_ed_vals[y_count]
                                    this_tech_fuel_ed_vals.append(add_value)
                                    dict_energy_demand_by_fuel[fuel][y_count] += add_value
                                    y_count += 1

                        dict_energy_demand[tech].update({fuel:this_tech_fuel_ed_vals})

                # ...here we need to run the remainder if necessary:
                if check_percent is True:
                    fuel_rem = list(store_fush_rem.keys())[0]
                    oneminus_rem_list_fush = store_fush_rem[fuel_rem]
                    for fuel in list_fuel:
                        if fuel != fuel_rem:
                            for y in range(len(time_vector)):
                                oneminus_rem_list_fush[y] += store_fush[fuel][y]

                    this_tech_fuel_ed_vals = []
                    for y in range(len(time_vector)):
                        store_fush[fuel_rem][y] = 1-oneminus_rem_list_fush[y]
                        add_value = \
                            store_fush[fuel_rem][y]*this_tech_ed_vals[y]
                        this_tech_fuel_ed_vals.append(add_value)
                        dict_energy_demand_by_fuel[fuel_rem][y] += add_value

                    dict_energy_demand[tech].update({fuel_rem:this_tech_fuel_ed_vals})

            """
            We can modify the *dict_energy_demand_by_fuel* and the 
            *dict_energy_demand* dictionaries to reflect the effect of the 
            measures!
            """
            
            """
            Store the list of columns for "df_scen_3_2":
            
            ['Rule',
             'Category',
             'Measure',
             'Country',
             'Scenario',
             'Share for subcategory',
             'Substituted vector',
             'Substitute vector',
             'Displaced energy per unit [MJ/unit]',
             'New energy requirement per unit [MJ/unit]',
             'Number of units',
             'Units',
             'Target years',
             'New displacement base year',
             'Displaced value before first target year',
             'Displaced value between target years',
             'Displaced value in target years',
             'Displaced value after last target year',
             'New requirement base year',
             'New requirement value before first target year',
             'New requirement value between target years',
             'New requirement value in target years',
             'New requirement value after last target year']
            
            """
            
            # 3e) Open the hydrogen efficiency
            param_related_15 = 'Green_Hydrogen_Prod_Eff'
            mask_15 = (df_scen_rc['Parameter'] == param_related_15)
            df_param_related_15 = df_scen_rc.loc[mask_15]
            df_param_related_15.reset_index(drop=True, inplace=True)
            if not df_param_related_15.empty:
                list_h2_effic = list(df_param_related_15[time_vector].iloc[0])
            
            # Here we can modify the demand based on end-use assumptions:
            if len(df_scen_3_2.index.tolist()) > 0:
                print('Here we are modifying the demands according to "df_scen_3_2".')
                
                # Iterate across each applicable modification
                for m in range(len(df_scen_3_2.index.tolist())):
                    df32_rule = df_scen_3_2["Rule"].iloc[m]
                    df32_sh_cat = df_scen_3_2["Category"].iloc[m]
                    
                    
                    df32_sh_subcat_str = str(df_scen_3_2["Share for subcategory"].iloc[m]).split(' ; ')
                    df32_sh_subcat = [float(x) if isinstance(x, str) else x for x in df32_sh_subcat_str]
                    
                    df32_substituted = df_scen_3_2["Substituted vector"].iloc[m]
                    df32_substitute = df_scen_3_2["Substitute vector"].iloc[m]
                    
                    df32_depu_str = str(df_scen_3_2["Displaced energy per unit [MJ/unit]"].iloc[m]).split(" ; ")
                    df32_depu = [float(x) if isinstance(x, str) else x for x in df32_depu_str]
                    
                    df32_nerpu = df_scen_3_2["New energy requirement per unit [MJ/unit]"].iloc[m]
                    
                    df32_numunits_str = str(df_scen_3_2["Number of units"].iloc[m]).split(" ; ")
                    df32_numunits = [float(x) if isinstance(x, str) else x for x in df32_numunits_str]
                    
                    df32_units = df_scen_3_2["Units"].iloc[m]
                    df32_target_year_str = str(df_scen_3_2["Target years"].iloc[m])
                    
                    df32_ndby = df_scen_3_2["Displacement base year"].iloc[m]
                    df32_dvbfty = df_scen_3_2["Displaced value before first target year"].iloc[m]
                    df32_dvbetty = df_scen_3_2["Displaced value between target years"].iloc[m]
                    df32_dvty = df_scen_3_2["Displaced value in target years"].iloc[m]
                    df32_dvalty = df_scen_3_2["Displaced value after last target year"].iloc[m]
                    
                    df32_nrby = df_scen_3_2["New requirement base year"].iloc[m]
                    df32_nrvbfty = df_scen_3_2["New requirement value before first target year"].iloc[m]
                    df32_nrbetty = df_scen_3_2["New requirement value between target years"].iloc[m]
                    df32_nrty = df_scen_3_2["New requirement value in target years"].iloc[m]
                    df32_nralty = df_scen_3_2["New requirement value after last target year"].iloc[m]

                    add_vector_general =  [0] * len(time_vector)

                    if df32_rule == "Substitute energy per unit":
                        avector_counter = 0
                        df32_dvty_list_str = str(df32_dvty).split(' ; ')
                        df32_dvty_list = [float(x) if isinstance(x, str) else x for x in df32_dvty_list_str]
                        
                        for avector in df32_substituted.split(' ; '):

                            base_disp_vector, base_disp_vector_agg, subtract_vector, loc_depu = process_vectors(avector, dict_energy_demand, df32_sh_cat, dict_energy_demand_by_fuel, time_vector, df32_depu, avector_counter)

                            if df32_substitute != 'none':
                                base_newreq_vector = deepcopy(dict_energy_demand[df32_sh_cat][df32_substitute])
                                base_newreq_vector_agg = deepcopy(dict_energy_demand_by_fuel[df32_substitute])
                                                      
                            '''
                            Multiple target years can be defined in this step:
                            we will work only with one target year
                            '''
                            by_idx, ty_list_str, ty_list, ty_index_list = process_target_years(time_vector, df32_ndby, df32_target_year_str, df32_dvty_list, base_disp_vector, 'first', subtract_vector)

                            # Here we fill the subtract vector with all the required targets
                            fill_subtract_vector(time_vector, by_idx, ty_list, ty_index_list, df32_dvbfty, df32_dvbetty, df32_nralty, subtract_vector)
                            avector_counter += 1

                            # After having obtained the subtract vector, we need an add vector using the substitution ratios:
                            add_vector_general, add_vector_local = create_add_vectors(avector, df32_nerpu, loc_depu, subtract_vector, add_vector_general, 'first')
                            changed_disp_vector, change_disp_vector_agg = update_disp_vectors(base_disp_vector, base_disp_vector_agg, subtract_vector)
                            update_energy_demand_dicts(df32_substituted, df32_substitute, df32_sh_cat, avector, changed_disp_vector, change_disp_vector_agg, dict_energy_demand, dict_energy_demand_by_fuel, 'first')

                            # Change the vector that needed more energy, just locally:
                            changed_newreq_vector = [a + b for a, b in zip(base_newreq_vector, add_vector_local)]
                            # Update the corresponding dictionary:
                            if df32_substitute != 'none':
                                dict_energy_demand[df32_sh_cat][df32_substitute] = deepcopy(changed_newreq_vector)

                        # Change the vector that needed more energy, just in aggregate:
                        change_newreq_vector_agg = [a + b for a, b in zip(base_newreq_vector_agg, add_vector_general)]
                        # Update the corresponding dictionary:
                        if df32_substitute != 'none':
                            dict_energy_demand_by_fuel[df32_substitute] = deepcopy(change_newreq_vector_agg)


                    if df32_rule == "Substitute energy share" and not df_param_related_15.empty:
                        avector_counter = 0
                        for avector in df32_substituted.split(' ; '):
                            base_disp_vector = deepcopy(dict_energy_demand[df32_sh_cat][avector])
                            base_disp_vector_agg = deepcopy(dict_energy_demand_by_fuel[avector])

                            base_newreq_vector, base_newreq_vector_agg, subtract_vector, loc_depu, df32_substitute_orig = process_vectors(df32_substitute, dict_energy_demand, df32_sh_cat, dict_energy_demand_by_fuel, time_vector, df32_depu, avector_counter, 'second')                            
                            sh_subcat = df32_sh_subcat[avector_counter]
                                                      
                            '''
                            Multiple target years can be defined in this step:
                            we will work only with one target year
                            '''
                            by_idx, ty_list_str, ty_list, ty_index_list, tval = process_target_years(time_vector, df32_ndby, df32_target_year_str, df32_numunits, base_disp_vector, 'second', subtract_vector, sh_subcat)

                            # Here we fill the subtract vector with all the required targets
                            fill_subtract_vector(time_vector, by_idx, ty_list, ty_index_list, df32_dvbfty, df32_dvbetty, df32_nralty, subtract_vector)
                            avector_counter += 1

                            # After having obtained the subtract vector, we need an add vector using the substitution ratios:
                            add_vector_general, add_vector_local = create_add_vectors(avector, df32_nerpu, loc_depu, subtract_vector, add_vector_general, 'second', df32_substitute_orig=df32_substitute_orig, list_h2_effic=list_h2_effic)
                            changed_disp_vector, change_disp_vector_agg = update_disp_vectors(base_disp_vector, base_disp_vector_agg, subtract_vector)
                            update_energy_demand_dicts(df32_substituted, df32_substitute, df32_sh_cat, avector, changed_disp_vector, change_disp_vector_agg, dict_energy_demand, dict_energy_demand_by_fuel, 'second')                            

                            # Change the vector that needed more energy, just locally:
                            changed_newreq_vector = [a + b for a, b in zip(base_newreq_vector, add_vector_local)]
                            # Update the corresponding dictionary:
                            if df32_substitute != 'none':
                                dict_energy_demand[df32_sh_cat][df32_substitute] = deepcopy(changed_newreq_vector)

                        # Change the vector that needed more energy, just in aggregate:
                        change_newreq_vector_agg = [a + b for a, b in zip(base_newreq_vector_agg, add_vector_general)]
                        # Update the corresponding dictionary:
                        if df32_substitute != 'none':
                            dict_energy_demand_by_fuel[df32_substitute] = deepcopy(change_newreq_vector_agg)

                print('        > The modifications are complete!')

            # Here we can modify the demand from the endogenous calculation if the electrical demand needs an override           
            df_param_related_13 = filter_dataframe(df_scen_rc, 'scenario_simple', scenario='Fixed electricity production', column='Parameter')

            if not df_param_related_13.empty:
                if df_param_related_13['projection'].iloc[0] != 'ignore':
                    dict_energy_demand_ref = deepcopy(dict_energy_demand)
                    dict_energy_demand_by_fuel_ref = deepcopy(dict_energy_demand_by_fuel)
    
                    # Grab the electricity demand from the base year:
                    dem_elec_tot_ref = deepcopy(dict_energy_demand_by_fuel_ref['Electricity'])
                    dem_elec_tot_ref_by = dem_elec_tot_ref[0]
    
                    # This means that we need to override the electrical demand by estimating the adjustment across sectors
                    if df_param_related_13['projection'].iloc[0] == 'interpolate':
                        # Iterate across all years and determine:
                        #   Years that need to be filled according to interpolation
                        #   Years with an exact value that need to be added into the list
                        #   Years that need to follow the growth of the previous trajectory
    
                        known_vals = []
                        known_yrs = []
                        find_yrs = []
                        use_vals, use_vals_ratio = [], []
    
                        if df_param_related_13['Unit'].iloc[0] == 'GWh':
                            gwh_to_pj = 0.0036
                        else:
                            gwh_to_pj = 1
    
                        for y in range(len(time_vector)):
                            this_13_val = df_param_related_13[time_vector[y]].iloc[0]
                            if y > 0:
                                last_13_val = df_param_related_13[time_vector[y-1]].iloc[0]
                                change_13_val = (dem_elec_tot_ref[y] - dem_elec_tot_ref[y-1]) / dem_elec_tot_ref[y-1]
                            
                            if y == 0:
                                known_vals.append(dem_elec_tot_ref_by)
                                known_yrs.append(time_vector[y])
                                use_vals.append(dem_elec_tot_ref_by)
                                use_vals_ratio.append(1)
                            elif isinstance(this_13_val, (float, np.floating, int)) and not np.isnan(this_13_val):
                                this_13_val *= gwh_to_pj
                                known_vals.append(this_13_val)
                                known_yrs.append(time_vector[y])
                                use_vals.append(this_13_val)
                                ratio_adjust = this_13_val/dem_elec_tot_ref[y]
                                ratio_adjust_idx = deepcopy(y)
                                use_vals_ratio.append(ratio_adjust)
                            elif type(this_13_val) is not str:
                                if math.isnan(this_13_val):
                                    find_yrs.append(time_vector[y])
                                    use_vals.append('')
                                    use_vals_ratio.append('')
                            elif this_13_val == 'follow_gdp_intensity': # assume that the last known value must be continued
                                add_13_val = (1 + change_13_val)*known_vals[-1]
                                known_vals.append(add_13_val)
                                known_yrs.append(time_vector[y])
                                use_vals.append(add_13_val)
                                use_vals_ratio.append(1)
    
                        interp_vals_linear = \
                            interpolation_to_end(time_vector, 2021, use_vals,
                                                 'last', '')
                        interp_mults_linear = \
                            interpolation_to_end(time_vector, 2021, use_vals_ratio,
                                                 'last', '')
                        interp_vals_spirit = [interp_mults_linear[y]*dem_elec_tot_ref[y] if y < ratio_adjust_idx else interp_vals_linear[y] for y in range(len(time_vector))]
                           
                        # Let's use the spirit to calculate the ratio between new and old demand
                        adj_ele_ratio_list = [ivs / det if det != 0 else 0 for ivs, det in zip(interp_vals_spirit, dem_elec_tot_ref)]
    
                        # Now we need to update the electricity demand for each sector:
                        for sec in list(dict_energy_demand.keys()):
                            ref_ele_list = deepcopy(dict_energy_demand[sec]['Electricity'])
                            result_list = [ref * adj for ref, adj in zip(ref_ele_list, adj_ele_ratio_list)]
                            dict_energy_demand[sec]['Electricity'] = deepcopy(result_list)
    
                        ref_ele_list_tot = [ref * adj for ref, adj in zip(dem_elec_tot_ref, adj_ele_ratio_list)]
                        dict_energy_demand_by_fuel['Electricity'] = deepcopy(ref_ele_list_tot)

            # We can continue modifying the demands if a specific type of projection requires to increase the ambition:
            if this_fuel_deuc_projection == 'electrify_sector_2_max':
                for sec in list(dict_energy_demand.keys()):
                    ref_afuel_list_sum = [0] * len(time_vector)
                    
                    for afuel in list(dict_energy_demand[sec].keys()):
                        if 'Total' not in afuel and 'Non-' not in afuel:
                            ref_afuel_list = deepcopy(dict_energy_demand[sec][afuel])
                            ref_afuel_list_sum = [a + b for a, b in zip(ref_afuel_list_sum, ref_afuel_list)]
                    
                    if sum(ref_afuel_list_sum) > 0:
                        # Act upon electrification, i.e., the share:
                        elec_sh = [100*a / b for a, b in zip(dict_energy_demand[sec]['Electricity'], ref_afuel_list_sum)]
                        elec_sh_orig = deepcopy(elec_sh)
                        if sum(elec_sh) == 0:
                            elec_sh = [0.01] * len(time_vector)
                            dict_energy_demand[sec]['Electricity'] = \
                                [a * b/100 for a, b in zip(elec_sh, ref_afuel_list_sum)]
                        for y in range(len(time_vector)):
                            if float(elec_sh[y]) == 0.0:
                                elec_sh[y] = 0.001
                        
                        # The rest total:
                        rest_sh = [100 - a for a in elec_sh]
                        if elec_sh[-1] < this_fuel_deuc_value_type:
                            mult_empty_norm = ['' for y in range(len(time_vector))]
                            mult_empty_norm = assign_values_for_indices(elec_sh, mult_empty_norm, [0, 1, 2])
                            mult_empty_norm = assign_values_for_indices(elec_sh, mult_empty_norm, [3, 4, 5])
                            mult_empty_norm = assign_values_for_indices(elec_sh, mult_empty_norm, [6, 7, 8])
                            mult_empty_norm = assign_values_for_indices(elec_sh, mult_empty_norm, [9])
                            mult_empty_norm[-1] = this_fuel_deuc_value_type
                            mult_interp_norm = \
                                interpolation_to_end(time_vector, ini_simu_yr, \
                                mult_empty_norm, 'last', '')
    
                            mult_interp_norm_comp = \
                                [a - b for a, b in zip([100]*len(time_vector), mult_interp_norm)]
    
                            adj_ratio_rest = \
                                [a / b for a, b in zip(mult_interp_norm_comp, rest_sh)]
    
                            # We must find a multiplier for the last year, and interpolate starting in 2023
                            adj_ratio_ele = \
                                [a / b for a, b in zip(mult_interp_norm, elec_sh)]
    
                            orig_list = dict_energy_demand[sec]['Electricity']
                            dict_energy_demand[sec]['Electricity'] = \
                                [a * b for a, b in zip(adj_ratio_ele, orig_list)]
    
                            # The rest of the share:
                            for afuel in list(dict_energy_demand[sec].keys()):
                                if 'Total' not in afuel and 'Non-' not in afuel and afuel != 'Electricity':
                                    afuel_sh = [100*a / b for a, b in zip(dict_energy_demand[sec][afuel], ref_afuel_list_sum)]  # may be unnecessary
                                    orig_list = dict_energy_demand[sec][afuel]
                                    dict_energy_demand[sec][afuel] = \
                                        [a * b for a, b in zip(adj_ratio_rest, orig_list)]
                                    
                            # Estimate the new total
                            new_total_sum_list = [0] * len(time_vector)
                            for afuel in list(dict_energy_demand[sec].keys()):
                                if 'Total' not in afuel and 'Non-' not in afuel:
                                    new_total_sum_list = \
                                        [a + b for a, b in zip(dict_energy_demand[sec][afuel], new_total_sum_list)]
                                    
                            # Calculate the new difference between totals after adjustment:
                            diff_list = [100*(a-b)/a for a, b in zip(ref_afuel_list_sum, new_total_sum_list)]
                            if sum(diff_list)/len(diff_list) > 1:
                                print('Algorithm does not work.')
                                print(diff_list)
                                sys.exit()

                # Now update the demands per fuel:
                for afuel in list(dict_energy_demand_by_fuel.keys()):
                    afuel_sum_list = [0] * len(time_vector)
                    for sec in list(dict_energy_demand.keys()):
                        afuel_sum_list = \
                            [a + b for a, b in zip(afuel_sum_list, dict_energy_demand[sec][afuel])]
                        dict_energy_demand_by_fuel[afuel] = deepcopy(afuel_sum_list)

            # Now, let's store the energy demand projections in the country dictionary:
            # parameters: 'Energy demand by sector', 'Energy intensity by sector'
            dict_local_country.update({this_country:{'Energy demand by sector': dict_energy_demand}})
            dict_local_country[this_country].update({'Energy intensity by sector': dict_energy_intensity})
            dict_local_country[this_country].update({'Energy demand by fuel': dict_energy_demand_by_fuel})

            """
            INSTRUCTIONS:
            1) Perform the agro and waste estimations
            2) Store the variables for print
            """
            if model_agro_and_waste:
                # Dataframes with sets:
                # df2_agr_sets_eq
                # df2_res_sets_eq

                # Dataframes with scenarios:
                # df3_agr_data
                # df3_res_data

                # Dataframes with emission factors:
                # df4_ef_agro_res
                # df4_ar_emi

                # Dataframes with techno-economic data:
                # d5_agr
                # d5_res

                # Now we will model according to the diagrams, and we
                # will model emissions and costs simultaneously

                # General data used:
                # Population: this_pop_vector
                # GDP per capita: this_gdp_per_cap_vals

                ### Agriculture
                #-
                # Produccion pecuaria y fermentacion enterica
                # Calculate future activity data:
                df3_agr_data_scen = filter_dataframe(df3_agr_data, 'scenario_simple', scenario=this_scen)               
                df4_ef_agro_res_scen = filter_dataframe(df4_ef_agro_res, 'scenario_simple', scenario=this_scen)
                # Project the demand:
                df3_agr_data_scen_dem = filter_dataframe(df3_agr_data_scen, 'scenario_simple', scenario='Demand', column='Parameter')
                types_livestock, types_projection, types_by_vals_dem = process_types_and_df(df3_agr_data_scen_dem, 'only_types', time_vector)

                dem_liv_proj_dict = {}
                for l in range(len(types_livestock)):
                    this_live = types_livestock[l]
                    this_proj = types_projection[l]
                    this_by_val = types_by_vals_dem[l]
                    this_val_list = []

                    if this_proj == 'grow_gdp_pc':
                        for y in range(len(time_vector)):
                            gdp_pc_gr = this_gdp_pc_growth_vals[y]/100
                            if y == 0:
                                this_val_list.append(this_by_val)
                            else:
                                next_val = this_val_list[-1] * (1 + gdp_pc_gr)
                                this_val_list.append(next_val)

                    dem_liv_proj_dict.update({this_live: this_val_list}) #Megatons

                # Include imports and exports (Megatons or Millions of tons)
                df3_agr_data_scen_imp = filter_dataframe(df3_agr_data_scen, 'two_columns_scenarios', \
                                                        scenario='Imports', scenario_2=5, \
                                                        column='Parameter', column_2='Param_ID')
                types_livestock, types_projection, types_by_vals_imp = process_types_and_df(df3_agr_data_scen_imp, 'only_types', time_vector)

                # Projections for imports
                imp_proj_dict = generate_imp_proj_dict(time_vector, df3_agr_data_scen_imp, 
                                                types_livestock, types_projection, 
                                                types_by_vals_imp, this_gdp_pc_growth_vals, 
                                                this_pop_vector)                        
                # Grabbing OPEX:
                d5_liv_imp_opex = filter_dataframe(d5_agr, 'two_columns_scenarios', \
                                                   scenario='Livestock Imports', scenario_2=this_scen, \
                                                   column='Tech', column_2='Scenario')
  
                d5_liv_imp_opex_by, d5_liv_imp_opex_proj, types_imp_projection_opex, types_by_imp_vals_opex = process_types_and_df(d5_liv_imp_opex, 'df_types_4', time_vector, 'Projection')                                    
                
                list_liv_imp_opex = []
                if d5_liv_imp_opex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_liv_imp_opex.append(types_by_imp_vals_opex) 
                        
                # Estimate the cost for livestok (MUSD):
                opex_liv_imp_proj_dict, opex_liv_imp_proj_dict_disc = generate_estimation_dicts(
                    types_livestock, imp_proj_dict, list_liv_imp_opex, time_vector, r_rate, r_year, 'third')               
                        
                #Storing livestock import costs results 
                dict_local_country[this_country].update({'OPEX de importación ganadera [MUSD]': deepcopy(opex_liv_imp_proj_dict)})
                dict_local_country[this_country].update({'OPEX de importación ganadera [MUSD] (disc)': deepcopy(opex_liv_imp_proj_dict_disc)})
                
                #Exports (Megatoneladas o Millones de tonaladas)
                df3_agr_data_scen_exp = filter_dataframe(df3_agr_data_scen, 'two_columns_scenarios', \
                                                                    scenario='Exports', scenario_2=7, \
                                                                    column='Parameter', column_2='Param_ID')
                types_livestock, types_projection, types_by_vals_exp = process_types_and_df(df3_agr_data_scen_exp, 'only_types', time_vector)
                
                # Projections for Exports
                exp_proj_dict = generate_imp_proj_dict(time_vector, df3_agr_data_scen_exp, 
                                                types_livestock, types_projection, 
                                                types_by_vals_exp, this_gdp_pc_growth_vals, 
                                                this_pop_vector)     
                        
                # Grabbing OPEX:
                d5_liv_exp_opex = filter_dataframe(d5_agr, 'two_columns_scenarios', \
                                                   scenario='Livestock Exports', scenario_2=this_scen, \
                                                   column='Tech', column_2='Scenario')
                d5_liv_exp_opex_by, d5_liv_exp_opex_proj, types_by_exp_vals_opex = process_types_and_df(d5_liv_exp_opex, 'df_types_3', time_vector)
              
                list_liv_exp_opex = []
                if d5_liv_exp_opex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_liv_exp_opex.append(types_by_exp_vals_opex)    
                # Estimate the cost for livestok (MUSD):
                opex_liv_exp_proj_dict, opex_liv_exp_proj_dict_disc = generate_estimation_dicts(
                    types_livestock, exp_proj_dict, list_liv_exp_opex, time_vector, r_rate, r_year, 'third')            
                        
                #Storing livestock import costs results 
                dict_local_country[this_country].update({'OPEX de exportación ganadera [MUSD]': deepcopy(opex_liv_exp_proj_dict)})
                dict_local_country[this_country].update({'OPEX de exportación ganadera [MUSD] (disc)': deepcopy(opex_liv_exp_proj_dict_disc)})

                # Obtain heads with average weight:
                df3_agr_data_scen_heads = filter_dataframe(df3_agr_data_scen, 'scenario_simple', scenario='Heads', column='Parameter')    
                df3_agr_data_scen_aw = filter_dataframe(df3_agr_data_scen, 'scenario_simple', scenario='Average weight', column='Parameter')
                types_livestock, types_projection, types_by_vals_aw = process_types_and_df(df3_agr_data_scen_aw, 'only_types', time_vector)

                # Projection for heads with average weight
                dem_aw_proj_dict = generate_projection_dict(
                    types_livestock, types_projection, types_by_vals_aw, 
                    df3_agr_data_scen_aw, time_vector, 'first')

                types_livestock, types_projection, types_by_vals_head = process_types_and_df(df3_agr_data_scen_heads, 'only_types', time_vector)                
               
                prod_head_proj_dict = {}
                for l in range(len(types_livestock)):
                    this_live = types_livestock[l]
                    this_proj = types_projection[l]
                    this_by_val = types_by_vals_head[l]
                    this_val_list = []

                    if this_proj == 'endogenous':
                        for y in range(len(time_vector)):
                            if y == 0:
                                this_val_list.append(this_by_val)
                            else:
                                next_val = \
                                    (dem_liv_proj_dict[this_live][y] -\
                                     imp_proj_dict[this_live][y] + \
                                         exp_proj_dict[this_live][y]) * 1e9 / \
                                     (dem_aw_proj_dict[this_live][y])      
                                this_val_list.append(next_val)

                    prod_head_proj_dict.update({this_live: this_val_list})
                                    
                # Gather the emissions factor of enteric fermentation:
                df4_ef_agro_res_fe = filter_dataframe(df4_ef_agro_res_scen, 'scenario_simple', scenario='Fermentación entérica', column='Group')
                types_livestock, types_projection, types_by_vals_fe = process_types_and_df(df4_ef_agro_res_fe, 'only_types', time_vector)
                
                # Projection for emissions factor of enteric fermentation 
                emisfac_fe_proj_dict = generate_projection_dict(
                    types_livestock, types_projection, types_by_vals_fe, 
                    df4_ef_agro_res_fe, time_vector, 'first')

                # Estimate the enteric fermentation emissions:
                out_emis_fe_proj_dict = generate_estimation_dicts(
                    types_livestock, emisfac_fe_proj_dict, prod_head_proj_dict, time_vector, r_rate, r_year, 'fourth')[0]
               
                # Calculating Costs:
                list_liv_capex = []
                list_liv_olife = []
                list_liv_opex = []

                # Grabbing CAPEX:                    
                d5_liv_capex = filter_dataframe(d5_agr, 'three_columns_scenarios', \
                                     scenario='Lifestock Production_CAPEX', scenario_2='ganado', scenario_3=this_scen, \
                                     column='Tech', column_2='Parameter', column_3='Scenario')
                d5_liv_capex_by, d5_liv_capex_proj = process_types_and_df(d5_liv_capex, 'df', time_vector)                
                if d5_liv_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_liv_capex.append(d5_liv_capex_by)
                # Grabbing operational life:
                d5_liv_ol = filter_dataframe(d5_agr, 'three_columns_scenarios', \
                                     scenario='Lifestock Production', scenario_2='Operational life', scenario_3=this_scen, \
                                     column='Tech', column_2='Parameter', column_3='Scenario')                                    
                d5_liv_ol_by, d5_liv_ol_proj = process_types_and_df(d5_liv_ol, 'df', time_vector)
                if d5_liv_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_liv_olife.append(d5_liv_ol_by)
                # Grabbing OPEX:
                d5_liv_opex = filter_dataframe(d5_agr, 'two_columns_scenarios', \
                                     scenario='Lifestock Production_OPEX', scenario_2=this_scen, \
                                     column='Tech', column_2='Scenario')                 
                d5_liv_opex_by, d5_liv_opex_proj = process_types_and_df(d5_liv_opex, 'df', time_vector)                
                types_liv_opex, types_projection_opex, types_by_vals_opex = process_types_and_df(d5_liv_opex, 'only_types', time_vector, column='Parameter')                

                for l in range(len(types_liv_opex)):
                    if d5_liv_opex_proj == 'flat':
                        for y in range(len(time_vector)):
                            list_liv_opex.append(d5_liv_opex_by)

                # Projection for OPEX
                opex_liv_proj_dict = generate_projection_dict(
                    types_liv_opex, types_projection_opex, types_by_vals_opex, 
                    d5_liv_opex, time_vector, 'first')
                        
                # Calculate investment requirements:
                livestock_heads_total = [sum(values) for values in \
                                         zip(*prod_head_proj_dict.values())]
            
                total_livestock_delta = [0]
                for y in range(1, len(time_vector)):
                    total_livestock_delta.append(livestock_heads_total[y] - livestock_heads_total[y-1])
                for y in range(int(list_liv_olife[0]), len(time_vector)):
                    total_livestock_delta[y] += total_livestock_delta[y - int(list_liv_olife[y])]
                
                liv_capex = [(ucost * act)/1e6 for ucost, act in \
                              zip(list_liv_capex, livestock_heads_total)]
                
                liv_capex_disc = deepcopy(liv_capex)
                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    liv_capex_disc[y] *= disc_constant
                    
                # Estimate the cost for livestok:
                out_opex_liv_proj_dict, out_opex_liv_proj_dict_disc = generate_estimation_dicts(
                    types_liv_opex, opex_liv_proj_dict, livestock_heads_total, time_vector, r_rate, r_year, 'second')
                                                        
                #Storing livestock import costs results 
                dict_local_country[this_country].update({'CAPEX de ganadera [MUSD]': deepcopy(liv_capex)})
                dict_local_country[this_country].update({'CAPEX de ganadera [MUSD] (disc)': deepcopy(liv_capex_disc)})
                dict_local_country[this_country].update({'OPEX de ganadera [MUSD]': deepcopy(out_opex_liv_proj_dict)})
                dict_local_country[this_country].update({'OPEX de ganadera [MUSD] (disc)': deepcopy(out_opex_liv_proj_dict_disc)})
                
                # Manure management system
                # Gather the emissions factor of enteric fermentation:
                df4_ef_agro_res_sge = filter_dataframe(df4_ef_agro_res_scen, 'scenario_simple', scenario='Sistema de Gestión de Estiércol', column='Group')
                types_livestock, types_projection, types_by_vals_sge = process_types_and_df(df4_ef_agro_res_sge, 'only_types', time_vector)

                # Projection for manure management system
                emisfac_sge_proj_dict = generate_projection_dict(
                    types_livestock, types_projection, types_by_vals_sge, 
                    df4_ef_agro_res_sge, time_vector, 'first')                

                # Estimate the manure management emissions:
                out_emis_sge_proj_dict = generate_estimation_dicts(
                    types_livestock, emisfac_sge_proj_dict, prod_head_proj_dict, time_vector, r_rate, r_year, 'fourth')[0]
                
                # Calculating Costs:
                sge_capex_out_dict = {}
                sge_capex_out_dict_disc = {}

                # Grabbing CAPEX:
                d5_sge_capex = filter_dataframe(d5_agr, 'three_columns_scenarios', \
                    scenario='SGE', scenario_2='CAPEX', scenario_3=this_scen, \
                    column='Tech', column_2='Parameter', column_3='Scenario')
                d5_sge_capex_by, d5_sge_capex_proj = process_types_and_df(d5_sge_capex, 'df', time_vector)
                types_sge_capex, types_sge_projection, types_by_vals_sge = process_types_and_df(d5_sge_capex, 'only_types', time_vector)

                sge_capex_proj_dict = generate_short_projection(
                    types_sge_capex, types_sge_projection, types_by_vals_sge, 
                    time_vector)                
                        
                # Grabbing operational life:
                d5_sge_ol = filter_dataframe(d5_agr, 'three_columns_scenarios', \
                    scenario='SGE', scenario_2='Operational life', scenario_3=this_scen, \
                    column='Tech', column_2='Parameter', column_3='Scenario')              
                d5_sge_fl_ol_by, d5_seg_ol_proj = process_types_and_df(d5_sge_ol, 'df', time_vector)                
                types_sge_ol, types_sge_ol_projection, types_by_vals_sge_ol = process_types_and_df(d5_sge_ol, 'only_types', time_vector)                

                list_sge_olife_dict = generate_short_projection(
                    types_sge_ol, types_sge_ol_projection, types_by_vals_sge_ol, 
                    time_vector)                 
                        
                # Grabbing fixed OPEX:
                d5_sge_fopex = filter_dataframe(d5_agr, 'three_columns_scenarios', \
                                     scenario='SGE', scenario_2='Fixed FOM', scenario_3=this_scen, \
                                     column='Tech', column_2='Parameter', column_3='Scenario')
                d5_sge_fopex_by, d5_sge_fopex_proj = process_types_and_df(d5_sge_fopex, 'df', time_vector)
                types_sge_fopex, types_sge_projection_fopex, types_by_vals_sge_fopex = process_types_and_df(d5_sge_fopex, 'only_types', time_vector)

                sge_fopex_proj_dict = generate_short_projection(
                    types_sge_fopex, types_sge_projection_fopex, types_by_vals_sge_fopex,
                    time_vector)                  
                            
                # Grabbing variable OPEX:
                d5_sge_vopex = filter_dataframe(d5_agr, 'three_columns_scenarios', \
                                     scenario='SGE', scenario_2='Variable FOM', scenario_3=this_scen, \
                                     column='Tech', column_2='Parameter', column_3='Scenario')
                d5_sge_vopex_by, d5_sge_vopex_proj = process_types_and_df(d5_sge_vopex, 'df', time_vector)
                types_sge_vopex, types_sge_projection_vopex, types_by_vals_sge_vopex = process_types_and_df(d5_sge_vopex, 'only_types', time_vector)                

                list_sge_olife_dict = generate_short_projection(
                    types_sge_ol, types_sge_ol_projection, types_by_vals_sge_ol, 
                    time_vector)  
                
                sge_vopex_proj_dict = {}
                for l in range(len(types_sge_vopex)):
                    this_opex = types_sge_vopex[l]
                    this_proj = types_sge_projection_vopex[l]
                    this_by_val = types_by_vals_sge_vopex[l]
                    this_val_list = []                    
                    if this_proj == 'flat':
                        this_val_list = [this_by_val]*len(time_vector)
                        sge_vopex_proj_dict.update({this_opex: this_val_list})           
                    #Calculating for investment requirement (MUSD)        
                    # Compute delta for this technology
                    tech_life_list = list_sge_olife_dict[types_sge_vopex[l]]  # This list needs to be provided for each technology type
                    this_live = prod_head_proj_dict[types_sge_vopex[l]]
                    delta_list = compute_delta_for_technology(this_live, time_vector, tech_life_list)
                    capex_sge = [(ucost * act)/1e6 for ucost, act in zip(sge_capex_proj_dict[types_sge_vopex[l]], delta_list)]
                    sge_capex_out_dict.update({this_opex: this_val_list})
                    discount_rate = r_rate/100
                    for key, values in sge_capex_out_dict.items():
                        sge_capex_out_dict_disc[key] = discounted_values(values, discount_rate)
                                        
                # Estimate the fixed opex sge (MUSD)
                out_fopex_proj_dict, out_fopex_proj_dict_disc = generate_estimation_dicts(
                    types_sge_fopex, sge_fopex_proj_dict, prod_head_proj_dict, time_vector, r_rate, r_year, 'first')
               
                # Estimate the variable opex sge (MUSD)
                out_vopex_proj_dict, out_vopex_proj_dict_disc = generate_estimation_dicts(
                    types_sge_vopex, sge_vopex_proj_dict, prod_head_proj_dict, time_vector, r_rate, r_year, 'first')                   

                #Storing SGE costs results 
                dict_local_country[this_country].update({'CAPEX por Sistema de tratemiento de estiércol de ganado [MUSD]': deepcopy(sge_capex_out_dict)})
                dict_local_country[this_country].update({'OPEX fijo por Sistema de tratemiento de estiércol de ganado [MUSD]': deepcopy(out_fopex_proj_dict)})
                dict_local_country[this_country].update({'OPEX variable por Sistema de tratemiento de estiércol de ganado [MUSD]': deepcopy(out_vopex_proj_dict)})
                dict_local_country[this_country].update({'CAPEX por Sistema de tratemiento de estiércol de ganado [MUSD] (disc)': deepcopy(sge_capex_out_dict_disc)})
                dict_local_country[this_country].update({'OPEX fijo por Sistema de tratemiento de estiércol de ganado [MUSD](disc)': deepcopy(out_fopex_proj_dict_disc)})
                dict_local_country[this_country].update({'OPEX variable por Sistema de tratemiento de estiércol de ganado [MUSD](disc)': deepcopy(out_vopex_proj_dict_disc)})
                # STORING EMISSIONS
                dict_local_country[this_country].update({'Emisiones de fermentación entérica [kt CH4]': deepcopy(out_emis_fe_proj_dict)})
                dict_local_country[this_country].update({'Sistema de gestión de estiércol [kt CH4]': deepcopy(out_emis_sge_proj_dict)})
            
                # Rice cultivation
                # Rice demand
                df3_agr_data_scen_rice = filter_dataframe(df3_agr_data_scen, 'two_columns_scenarios', \
                                     scenario='Demand', scenario_2='Rice',\
                                     column='Parameter', column_2='Type')
                types_rice, types_projection_rice, types_by_vals_rice = process_types_and_df(df3_agr_data_scen_rice, 'only_types_2', time_vector)

                all_vals_gen_pc_dict, total_rice_list = calculate_projection(
                    df3_agr_data_scen_rice, 
                    types_projection_rice, 
                    types_by_vals_rice, 
                    this_gdp_pc_growth_vals, 
                    this_pop_vector, 
                    time_vector,
                    'first'
                )                 
                    
                # Rice importation
                df3_agr_data_scen_rice_imp = filter_dataframe(df3_agr_data_scen, 'two_columns_scenarios', \
                                                             scenario='Imports', scenario_2='Rice',\
                                                             column='Parameter', column_2='Type') # unit: Mton
                types_rice_imp, types_projection_rice_imp, types_by_vals_rice_imp = process_types_and_df(df3_agr_data_scen_rice_imp, 'only_types_2', time_vector)

                all_vals_gen_pc_dict, total_rice_imp_list = calculate_projection(
                    df3_agr_data_scen_rice, 
                    types_projection_rice_imp, 
                    types_by_vals_rice_imp, 
                    this_gdp_pc_growth_vals, 
                    this_pop_vector, 
                    time_vector,
                    'first'
                )
                    
                 # Calculating Costs:
                list_rice_imp_opex = []

                # Grabbing OPEX:
                d5_rice_imp_opex_mask = (d5_agr['Tech'] == 'Rice Imports') & \
                    (d5_agr['Parameter'] == 'OPEX') &\
                        (d5_agr['Scenario'] == this_scen)
                d5_rice_imp_opex = d5_agr.loc[d5_rice_imp_opex_mask]
                d5_rice_imp_opex_by = d5_rice_imp_opex[time_vector[0]].iloc[0]
                d5_rice_imp_opex_proj = d5_rice_imp_opex['Projection'].iloc[0]
                if d5_rice_imp_opex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_rice_imp_opex.append(d5_rice_imp_opex_by)                       
                
                rice_imp_opex = [(ucost * act) for ucost, act in zip(list_rice_imp_opex, total_rice_imp_list)] 
                                
                #Estimating discounted cost
                rice_imp_opex_disc = deepcopy(rice_imp_opex)
                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    rice_imp_opex_disc[y] *= disc_constant
                
                #Exportación de arroz 
                df3_agr_data_scen_rice_exp = filter_dataframe(df3_agr_data_scen, 'two_columns_scenarios', \
                                                                    scenario='Exports', scenario_2='Rice', \
                                                                    column='Parameter', column_2='Type') # unit: Mton
                types_rice_exp, types_projection_rice_exp, types_by_vals_rice_exp = process_types_and_df(df3_agr_data_scen_rice_exp, 'only_types_2', time_vector)

                all_vals_gen_pc_dict, total_rice_exp_list = calculate_projection(
                    df3_agr_data_scen_rice, 
                    types_projection_rice_exp, 
                    types_by_vals_rice_exp, 
                    this_gdp_pc_growth_vals, 
                    this_pop_vector, 
                    time_vector,
                    'first'
                )                
                                
                 # Calculating Costs:
                list_rice_exp_opex = []

                # Grabbing OPEX:
                d5_rice_exp_opex = filter_dataframe(d5_agr, 'three_columns_scenarios', \
                     scenario='Rice Exports', scenario_2='OPEX', scenario_3=this_scen, \
                     column='Tech', column_2='Parameter', column_3='Scenario')
                d5_rice_exp_opex_by, d5_rice_exp_opex_proj = process_types_and_df(d5_rice_exp_opex, 'df', time_vector)

                if d5_rice_exp_opex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_rice_exp_opex.append(d5_rice_exp_opex_by)       
                               
                rice_exp_opex = [(ucost * act) for ucost, act in zip(list_rice_exp_opex, total_rice_exp_list)] 
                rice_exp_opex_disc = deepcopy(rice_exp_opex)
                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    rice_exp_opex_disc[y] *= disc_constant
                
                #Storing rice exports and imports costs results 
                dict_local_country[this_country].update({'OPEX para importación de arroz [MUSD]': deepcopy(rice_imp_opex)})
                dict_local_country[this_country].update({'OPEX para exportación de arroz [MUSD]': deepcopy(rice_exp_opex)})
                dict_local_country[this_country].update({'OPEX para importación de arroz [MUSD](disc)': deepcopy(rice_imp_opex_disc)})
                dict_local_country[this_country].update({'OPEX para exportación de arroz [MUSD](disc)': deepcopy(rice_exp_opex_disc)})
                                
                # Data for floated rice
                df3_rice_fl = filter_dataframe(df3_agr_data_scen, 'scenario_simple', scenario='Inundación', column='Type')
                _, types_projection_rice_fl, _ = process_types_and_df(df3_rice_fl, 'only_types_2', time_vector)
                rice_fl_by, rice_fl_proj = process_types_and_df(df3_rice_fl, 'df', time_vector)    

                # Projection growth type:
                all_vals_gen_pc_dict, total_rice_fl_list = calculate_projection(
                    df3_agr_data_scen_rice,
                    rice_fl_proj,
                    rice_fl_by,
                    this_gdp_pc_growth_vals,
                    this_pop_vector,
                    time_vector,
                    'second',
                    df3_rice_fl
                )
                   
                df4_fe_rice_fl = filter_dataframe(df4_ef_agro_res_scen, 'two_columns_scenarios', \
                                                                    scenario='Cultivo de arroz', scenario_2='Cultivo de arroz inundado', \
                                                                    column='Group', column_2='Type')                                                                                  
                _, types_projection_fe_rice_fl, types_by_vals_fe_rice_fl = process_types_and_df(df4_fe_rice_fl, 'only_types_2', time_vector)                                
                
                if types_projection_fe_rice_fl == 'flat':
                    fe_rice_fl_list = [types_by_vals_fe_rice_fl] * len(time_vector)
                elif types_projection_fe_rice_fl == 'user_defined':
                    for y in range(len(time_vector)):
                        fe_rice_fl_list.append(df4_fe_rice_fl[time_vector[y]].iloc[0])
                
                #Emission estimation (kton CH4)
                rice_fl_emis = [(ef * act)/1e6 for ef, act in zip(fe_rice_fl_list, total_rice_fl_list)]   
                
                # Calculating Costs:
                list_rice_fl_capex = []
                list_rice_fl_olife = []
                list_rice_fl_opex = []

                # Grabbing CAPEX:
                d5_rice_fl_capex = filter_dataframe(d5_agr, 'two_columns_scenarios', \
                                                    scenario='Cultivo_arroz_ CAPEX', scenario_2=this_scen, \
                                                    column='Tech', column_2='Scenario')
                d5_rice_fl_capex_by, d5_rice_fl_capex_proj = process_types_and_df(d5_rice_fl_capex, 'df', time_vector)                                                             
                types_rice_capex, types_rice_projection, types_by_vals_fl_rice = process_types_and_df(d5_rice_fl_capex, 'only_types',\
                                                                                                      time_vector, column='Parameter',\
                                                                                                      column_2='Projection')
                rice_capex_proj_dict = generate_short_projection(
                    types_rice_capex, types_rice_projection, types_by_vals_fl_rice, 
                    time_vector)                   
                
                # Grabbing operational life:
                d5_rice_fl_ol = filter_dataframe(d5_agr, 'two_columns_scenarios', \
                                                scenario='Cultivo_arroz_Op', scenario_2=this_scen, \
                                                column='Tech', column_2='Scenario')
                d5_rice_fl_ol_by, d5_rice_fl_ol_proj = process_types_and_df(d5_rice_fl_ol, 'df', time_vector)
                if d5_rice_fl_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_rice_fl_olife.append(d5_rice_fl_ol_by)
                        
                # Grabbing OPEX:
                d5_rice_fl_opex = filter_dataframe(d5_agr, 'two_columns_scenarios', \
                                                    scenario='Cultivo_arroz_OPEX', scenario_2=this_scen, \
                                                    column='Tech', column_2='Scenario')
                d5_rice_fl_opex_by, d5_rice_fl_opex_proj = process_types_and_df(d5_rice_fl_opex, 'df', time_vector)
                types_rice_opex, types_rice_projection_opex, types_by_vals_rice_opex = process_types_and_df(d5_rice_fl_opex, 'only_types',\
                                                                                                      time_vector, column='Parameter',\
                                                                                                      column_2='Projection')  

                rice_opex_proj_dict = generate_short_projection(
                    types_rice_opex, types_rice_projection_opex, types_by_vals_rice_opex, 
                    time_vector)                                                          
                
                # Calculate investment requirements (ha):
                total_rice_fl_list_delta = calculate_investment_requirements(total_rice_fl_list, list_rice_fl_olife, time_vector)
                
                #Estimate the investment for floated rice (MUSD)
                out_capex_rice_proj_dict, out_capex_rice_proj_dict_disc = generate_estimation_dicts(
                    types_rice_capex, rice_capex_proj_dict, total_rice_fl_list_delta, time_vector, r_rate, r_year, 'second')
               
                
                # Estimate the cost for floated rice (MUSD):
                out_opex_rice_fl_proj_dict, out_opex_rice_fl_proj_dict_disc = generate_estimation_dicts(
                    types_rice_opex, rice_opex_proj_dict, total_rice_fl_list, time_vector, r_rate, r_year, 'second')                

                #Storing rice floated emissions and costs results 
                dict_local_country[this_country].update({'Emisiones de cultivo de arroz por inundación [kton CH4]': deepcopy(rice_fl_emis)})
                dict_local_country[this_country].update({'OPEX para cultivo de arroz por inundación [MUSD]': deepcopy(out_opex_rice_fl_proj_dict)})
                dict_local_country[this_country].update({'CAPEX para cultivo de arroz por inundación [MUSD]': deepcopy(out_capex_rice_proj_dict)})
                dict_local_country[this_country].update({'OPEX para cultivo de arroz por inundación [MUSD](disc)': deepcopy(out_opex_rice_fl_proj_dict_disc)})
                dict_local_country[this_country].update({'CAPEX para cultivo de arroz por inundación [MUSD](disc)': deepcopy(out_capex_rice_proj_dict)})

                #data for irrigated rice
                df3_rice_ir = filter_dataframe(df3_agr_data_scen, 'scenario_simple', scenario='Irrigado', column='Type') # unit: ha
                _, types_projection_rice_ir, _ = process_types_and_df(df3_rice_ir, 'only_types_2', time_vector)
                rice_ir_by, rice_ir_proj = process_types_and_df(df3_rice_ir, 'df', time_vector)                               
                
                # Projection growth type:
                all_vals_gen_pc_dict, total_rice_ir_list = calculate_projection(
                    df3_agr_data_scen_rice,
                    rice_ir_proj,
                    rice_ir_by,
                    this_gdp_pc_growth_vals,
                    this_pop_vector,
                    time_vector,
                    'second',
                    df3_rice_ir
                )
                                        
                df4_fe_rice_ir = filter_dataframe(df4_ef_agro_res_scen, 'two_columns_scenarios', \
                                                                    scenario='Cultivo de arroz', scenario_2='Cultivo de arroz irrigado', \
                                                                    column='Group', column_2='Type')                                                                                  
                _, types_projection_fe_rice_ir, types_by_vals_fe_rice_ir = process_types_and_df(df4_fe_rice_ir, 'only_types_2', time_vector)                    

                if types_projection_fe_rice_ir == 'flat':
                    fe_rice_ir_list = [types_by_vals_fe_rice_ir] * len(time_vector)
                
                #Emission estimation (kton CH4)
                rice_ir_emis = [(ef * act)/1e6 for ef, act in zip(fe_rice_ir_list, total_rice_ir_list)]                   
                
                # Calculating Costs:
                list_rice_ir_olife = []

                # Grabbing CAPEX:
                d5_rice_ir_capex = filter_dataframe(d5_agr, 'two_columns_scenarios', \
                                                    scenario='Cultivo_arroz_ CAPEX', scenario_2=this_scen, \
                                                    column='Tech', column_2='Scenario')
                d5_rice_ir_capex_by, d5_rice_ir_capex_proj = process_types_and_df(d5_rice_ir_capex, 'df', time_vector)
                types_rice_capex, types_rice_projection, types_by_vals_rice = process_types_and_df(d5_rice_ir_capex, 'only_types',\
                                                                                                   time_vector, column='Parameter',\
                                                                                                   column_2='Projection')      
                rice_capex_proj_dict = generate_short_projection(
                    types_rice_capex, types_rice_projection, types_by_vals_rice, 
                    time_vector) 
                                
                # Grabbing operational life:                    
                d5_rice_ir_ol = filter_dataframe(d5_agr, 'two_columns_scenarios', \
                                                scenario='Cultivo_arroz_Op', scenario_2=this_scen, \
                                                column='Tech', column_2='Scenario')
                d5_rice_ir_ol_by, d5_rice_ir_ol_proj = process_types_and_df(d5_rice_ir_ol, 'df', time_vector)

                if d5_rice_ir_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_rice_ir_olife.append(d5_rice_ir_ol_by)
                                           
                # Grabbing OPEX:
                d5_rice_ir_opex = filter_dataframe(d5_agr, 'two_columns_scenarios', \
                                                    scenario='Cultivo_arroz_OPEX', scenario_2=this_scen, \
                                                    column='Tech', column_2='Scenario')
                d5_rice_ir_opex_by, d5_rice_ir_opex_proj = process_types_and_df(d5_rice_ir_opex, 'df', time_vector)
                types_rice_opex, types_rice_projection_opex, types_by_vals_rice_opex = process_types_and_df(d5_rice_ir_opex, 'only_types',\
                                                                                                            time_vector, column='Parameter',\
                                                                                                            column_2='Projection')                    
                                
                rice_opex_proj_dict = generate_short_projection(
                    types_rice_opex, types_rice_projection_opex, types_by_vals_rice_opex, 
                    time_vector)                        
               
                # Calculate investment requirements (ha):
                total_rice_ir_list_delta = calculate_investment_requirements(total_rice_ir_list, list_rice_ir_olife, time_vector)
                
                #Estimate the investment for floated rice (MUSD)
                out_capex_rice_proj_dict, out_capex_rice_proj_dict_disc = generate_estimation_dicts(
                    types_rice_capex, rice_capex_proj_dict, total_rice_ir_list_delta, time_vector, r_rate, r_year, 'second')
                  
                # Estimate the cost for floated rice (MUSD):
                out_opex_rice_proj_dict, out_opex_rice_proj_dict_disc = generate_estimation_dicts(
                    types_rice_opex, rice_opex_proj_dict, total_rice_ir_list, time_vector, r_rate, r_year, 'second')
                                 
                #Storing rice irrigated emissions and costs results 
                dict_local_country[this_country].update({'Emisiones de cultivo de arroz por irrigación [kton CH4]': deepcopy(rice_ir_emis)})
                dict_local_country[this_country].update({'OPEX para cultivo de arroz por irrigación [MUSD]': deepcopy(out_opex_rice_proj_dict)})
                dict_local_country[this_country].update({'CAPEX para cultivo de arroz por irrigación [MUSD]': deepcopy(out_capex_rice_proj_dict)})
                dict_local_country[this_country].update({'OPEX para cultivo de arroz por irrigación [MUSD](disc)': deepcopy(out_opex_rice_proj_dict_disc)})
                dict_local_country[this_country].update({'CAPEX para cultivo de arroz por irrigación [MUSD](disc)': deepcopy(out_capex_rice_proj_dict_disc)})
                                
                #data for aerated rice
                df3_rice_ar = filter_dataframe(df3_agr_data_scen, 'scenario_simple', scenario='Aireado', column='Type') # unit: ha
                _, types_projection_rice_ar, _ = process_types_and_df(df3_rice_ar, 'only_types_2', time_vector)
                rice_ar_by, rice_ar_proj = process_types_and_df(df3_rice_ar, 'df', time_vector)
                
                # Projection growth type:
                all_vals_gen_pc_dict, total_rice_ar_list = calculate_projection(
                    df3_agr_data_scen_rice,
                    rice_ar_proj,
                    rice_ar_by,
                    this_gdp_pc_growth_vals,
                    this_pop_vector,
                    time_vector,
                    'second',
                    df3_rice_ar
                )
                
                df4_fe_rice_ar = filter_dataframe(df4_ef_agro_res_scen, 'two_columns_scenarios', \
                                                                    scenario='Cultivo de arroz', scenario_2='Cultivo de arroz aereado', \
                                                                    column='Group', column_2='Type')                                                                                  
                _, types_projection_fe_rice_ar, types_by_vals_fe_rice_ar = process_types_and_df(df4_fe_rice_ar, 'only_types_2', time_vector)
               
                if types_projection_fe_rice_ar == 'flat':
                    fe_rice_ar_list = [types_by_vals_fe_rice_ar] * len(time_vector)
                                
                #Emission estimation (kton CH4)
                rice_ar_emis = [(ef * act)/1e6 for ef, act in zip(fe_rice_ar_list, total_rice_ar_list)]   
                               
                # Calculating Costs:
                list_rice_ar_olife = []

                # Grabbing CAPEX:
                d5_rice_ar_capex = filter_dataframe(d5_agr, 'two_columns_scenarios', \
                                                    scenario='Cultivo_arroz_aireado_CAPEX', scenario_2=this_scen, \
                                                    column='Tech', column_2='Scenario')
                d5_rice_ar_capex_by, d5_rice_ar_capex_proj = process_types_and_df(d5_rice_ar_capex, 'df', time_vector)
                type_rice_capex, types_rice_projection, types_by_vals_rice = process_types_and_df(d5_rice_ar_capex, 'only_types',\
                                                                                                  time_vector, column='Parameter',\
                                                                                                  column_2='Projection') 
                rice_capex_proj_dict = generate_short_projection(
                    types_rice_capex, types_rice_projection, types_by_vals_rice, 
                    time_vector) 
                                 
                # Grabbing operational life:
                d5_rice_ar_ol = filter_dataframe(d5_agr, 'two_columns_scenarios', \
                                                scenario='Cultivo_arroz_aireado', scenario_2=this_scen, \
                                                column='Tech', column_2='Scenario')
                d5_rice_ar_ol_by, d5_rice_ar_ol_proj = process_types_and_df(d5_rice_ar_ol, 'df', time_vector)                    

                if d5_rice_ar_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_rice_ar_olife.append(d5_rice_ar_ol_by)
                             
                # Grabbing OPEX:
                d5_rice_ar_opex = filter_dataframe(d5_agr, 'two_columns_scenarios', \
                                                    scenario='Cultivo_arroz_aireado_OPEX', scenario_2=this_scen, \
                                                    column='Tech', column_2='Scenario')
                d5_rice_ar_opex_by, d5_rice_ar_opex_proj = process_types_and_df(d5_rice_ar_opex, 'df', time_vector)
                types_rice_opex_2, types_rice_projection_opex, types_by_vals_rice_opex = process_types_and_df(d5_rice_ar_opex, 'only_types',\
                                                                                                              time_vector, column='Parameter',\
                                                                                                              column_2='Projection')                     
  
                rice_opex_proj_dict = generate_short_projection(
                    types_rice_opex_2, types_rice_projection_opex, types_by_vals_rice_opex, 
                    time_vector) 
                        
                # Calculate investment requirements (ha):
                total_rice_ar_list_delta = calculate_investment_requirements(total_rice_ar_list, list_rice_ar_olife, time_vector)
                                
                #Estimate the investment for aerated rice (MUSD)
                out_capex_rice_proj_dict, out_capex_rice_proj_dict_disc = generate_estimation_dicts(
                    type_rice_capex, rice_capex_proj_dict, total_rice_ar_list_delta, time_vector, r_rate, r_year, 'second')
                                     
                # Estimate the cost for aerated rice (MUSD):
                out_opex_rice_proj_dict, out_opex_rice_proj_dict_disc = generate_estimation_dicts(
                    types_rice_opex_2, rice_opex_proj_dict, total_rice_ar_list, time_vector, r_rate, r_year, 'second')
                                        
                #Storing aerated rice emissions and costs results 
                dict_local_country[this_country].update({'Emisiones de cultivo de arroz por aireado [kton CH4]': deepcopy(rice_ar_emis)})
                dict_local_country[this_country].update({'OPEX para cultivo de arroz por aireado [MUSD]': deepcopy(out_opex_rice_proj_dict)})
                dict_local_country[this_country].update({'CAPEX para cultivo de arroz por aireado [MUSD]': deepcopy(out_capex_rice_proj_dict)})
                dict_local_country[this_country].update({'OPEX para cultivo de arroz por aireado [MUSD](disc)': deepcopy(out_opex_rice_proj_dict_disc)})
                dict_local_country[this_country].update({'CAPEX para cultivo de arroz por aireado [MUSD](disc)': deepcopy(out_capex_rice_proj_dict_disc)})

                # Burning of savannas
                df3_agr_data_scen_burns = filter_dataframe(df3_agr_data, 'scenario_simple', scenario='Quema de sabanas', column='Parameter')                                                                                      
                types_burns, types_projection_burns, types_by_vals_burns = process_types_and_df(df3_agr_data_scen_burns, 'only_types_2', time_vector)
            
                # Projection growth type:
                all_vals_gen_pc_dict, total_burns_list = calculate_projection(
                    df3_agr_data_scen_burns,
                    types_projection_burns,
                    types_by_vals_burns,
                    this_gdp_pc_growth_vals,
                    this_pop_vector,
                    time_vector,
                    'third'
                )                        
                df4_ef_burns_fe = filter_dataframe(df4_ef_agro_res_scen, 'scenario_simple', scenario='Quema de sabanas', column='Group')                                                                                      
                types_burns, types_projection_burns_ef, types_by_vals_fe_burns = process_types_and_df(df4_ef_burns_fe, 'only_types_2', time_vector)            
                
                fe_burns_list = []
                if types_projection_burns_ef == 'flat':
                    fe_burns_list = [types_by_vals_fe_burns] * len(time_vector)
                elif types_projection_burns_ef == 'user_defined': 
                    for y in range(len(time_vector)):
                        fe_burns_list.append(df4_ef_burns_fe[time_vector[y]].iloc[0])
                        
                #Emission estimation (kton CH4)
                burns_emis = [(ef * act)/1e6 for ef, act in zip(fe_burns_list, total_burns_list)]
                               
                # Calculating Costs:
                list_bunrs_capex = []
                list_burns_olife = []
                list_burns_opex = []

                # Grabbing CAPEX:
                d5_agr_br_capex = filter_dataframe(d5_agr, 'three_columns_scenarios', \
                     scenario='Quema_Sabana', scenario_2='CAPEX', scenario_3=this_scen, \
                     column='Tech', column_2='Parameter', column_3='Scenario')                                                          
                d5_agr_br_capex_by, d5_agr_br_capex_proj = process_types_and_df(d5_agr_br_capex, 'df', time_vector)                    
                if d5_agr_br_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_bunrs_capex.append(d5_agr_br_capex_by)
                # Grabbing operational life:
                d5_agr_br_ol = filter_dataframe(d5_agr, 'three_columns_scenarios', \
                     scenario='Quema_Sabana', scenario_2='Operational life', scenario_3=this_scen, \
                     column='Tech', column_2='Parameter', column_3='Scenario')
                d5_agr_br_ol_by, d5_agr_br_ol_proj = process_types_and_df(d5_agr_br_ol, 'df', time_vector)                                                      
                if d5_agr_br_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_burns_olife.append(d5_agr_br_ol_by)
                # Grabbing OPEX:
                d5_agr_br_opex = filter_dataframe(d5_agr, 'three_columns_scenarios', \
                     scenario='Quema_Sabana', scenario_2='OPEX', scenario_3=this_scen, \
                     column='Tech', column_2='Parameter', column_3='Scenario')
                d5_agr_br_opex_by, d5_agr_br_opex_proj = process_types_and_df(d5_agr_br_opex, 'df', time_vector)
                if d5_agr_br_opex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_burns_opex.append(d5_agr_br_opex_by)
                elif d5_agr_br_opex_proj == 'user_defined':
                    for y in range(len(time_vector)):
                        list_burns_opex.append(d5_agr_br_opex[time_vector[y]].iloc[0])                       
               
                # Calculate investment requirements:
                total_burns_list_delta = calculate_investment_requirements(total_burns_list, list_burns_olife, time_vector)
                
                br_opex = [(ucost * act)/1e6 for ucost, act in zip(list_burns_opex, total_burns_list)]
                br_capex = [(ucost * act)/1e6 for ucost, act in zip(list_bunrs_capex, total_burns_list_delta)]
                                
                br_capex_disc, br_opex_disc = deepcopy(br_capex), deepcopy(br_opex)

                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    br_capex_disc[y] *= disc_constant
                    br_opex_disc[y] *= disc_constant

                #Storing burnt sabanas costs and emissions results 
                dict_local_country[this_country].update({'Emisiones de quema de sabanas [kton CH4]': deepcopy(burns_emis)})
                dict_local_country[this_country].update({'OPEX para quema de saban [MUSD]': deepcopy(br_opex)})
                dict_local_country[this_country].update({'CAPEX para quema de saban [MUSD]': deepcopy(br_capex)})
                dict_local_country[this_country].update({'OPEX para quema de saban [MUSD](disc)': deepcopy(br_opex_disc)})
                dict_local_country[this_country].update({'CAPEX para quema de saban [MUSD](disc)': deepcopy(br_capex_disc)}) 
                                
                # Burning of agricultural residues
                df3_agr_data_scen_burns_ag = filter_dataframe(df3_agr_data_scen, 'scenario_simple', scenario='Quema de residuos', column='Parameter')                                                                          
                types_burns_ag, types_projection_burns_ag, types_by_vals_burns_ag = process_types_and_df(df3_agr_data_scen_burns_ag, 'only_types_2', time_vector)
                
                # Data for cereals
                df3_br_ce = filter_dataframe(df3_agr_data_scen, 'scenario_simple', scenario='Cereales', column='Type')                                                                    
                _, types_projection_burn_ce, _ = process_types_and_df(df3_br_ce, 'only_types_2', time_vector)                                                                
                grs_br_ce_by, grs_br_ce_proj = process_types_and_df(df3_br_ce, 'df', time_vector)
                 
                # Projection growth type:
                all_vals_gen_pc_dict, total_burns_ce_list = calculate_projection(
                    df3_agr_data_scen_burns_ag,
                    types_projection_burn_ce,
                    grs_br_ce_by,
                    this_gdp_pc_growth_vals,
                    this_pop_vector,
                    time_vector,
                    'third'
                )
                
                df4_ef_burns_ce = filter_dataframe(df4_ef_agro_res_scen, 'two_columns_scenarios', \
                                                    scenario='Quema de residuos', scenario_2='Cereales', \
                                                    column='Group', column_2='Type')                                                                                    
                _, types_projection_burns_ef_ce, types_by_vals_burns_ef_ce = process_types_and_df(df4_ef_burns_ce, 'only_types_2', time_vector)
                
                fe_burns_ce_list = []
                if types_projection_burns_ef_ce == 'flat':
                    fe_burns_ce_list = [types_by_vals_burns_ef_ce] * len(time_vector)
                elif types_projection_burns_ef_ce == 'user_defined':
                    for y in range(len(time_vector)):
                        fe_burns_ce_list.append(df4_ef_burns_ce[time_vector[y]].iloc[0])
                                
                # Emission estimation (kton of CH4)
                ce_burns_emis = [(ef * act)/1e6 for ef, act in zip(fe_burns_ce_list, total_burns_ce_list)]
                                  
                # Data for sugarcane
                df3_br_sc = filter_dataframe(df3_agr_data_scen, 'scenario_simple', scenario='Caña de Azúcar', column='Type')                                                                       
                _, types_projection_burn_sc, _ = process_types_and_df(df3_br_sc, 'only_types_2', time_vector)                                                                
                grs_br_sc_by, grs_br_sc_proj = process_types_and_df(df3_br_sc, 'df', time_vector) 
                
                # Projection growth type:
                all_vals_gen_pc_dict, total_burns_sc_list = calculate_projection(
                    df3_agr_data_scen_burns_ag,
                    types_projection_burn_sc,
                    grs_br_sc_by,
                    this_gdp_pc_growth_vals,
                    this_pop_vector,
                    time_vector,
                    'third'
                )
                    
                df4_ef_burns_sc = filter_dataframe(df4_ef_agro_res_scen, 'two_columns_scenarios', \
                                                   scenario='Quema de residuos', scenario_2='Caña de azúcar',\
                                                   column='Group', column_2='Type')                                                                                    
                _, types_projection_burns_ef_sc, types_by_vals_burns_ef_sc = process_types_and_df(df4_ef_burns_sc, 'only_types_2', time_vector)
                 
                fe_burns_sc_list = [] 
                if types_projection_burns_ef_sc == 'flat':
                     fe_burns_sc_list = [types_by_vals_burns_ef_sc] * len(time_vector)
                elif types_projection_burns_ef_sc == 'user_defined':
                    for y in range(len(time_vector)):
                        fe_burns_sc_list.append(df4_ef_burns_sc[time_vector[y]].iloc[0])
                 
                 #Emission estimation (kton of CH4)
                sc_burns_emis = [(ef * act)/1e6 for ef, act in zip(fe_burns_sc_list, total_burns_sc_list)]
                                
                # Calculating Costs:
                list_bunrs_re_capex = []
                list_burns_re_olife = []
                list_burns_re_opex = []

                # Grabbing CAPEX:
                d5_agr_brre_capex = filter_dataframe(d5_agr, 'three_columns_scenarios', \
                     scenario='Quema_Residuos', scenario_2='CAPEX', scenario_3=this_scen, \
                     column='Tech', column_2='Parameter', column_3='Scenario')                                                                      
                d5_agr_brre_capex_by, d5_agr_brre_capex_proj = process_types_and_df(d5_agr_brre_capex, 'df', time_vector)
                if d5_agr_brre_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_bunrs_re_capex.append(d5_agr_brre_capex_by)
                # Grabbing operational life:
                d5_agr_brre_ol = filter_dataframe(d5_agr, 'three_columns_scenarios', \
                     scenario='Quema_Residuos', scenario_2='Operational life', scenario_3=this_scen, \
                     column='Tech', column_2='Parameter', column_3='Scenario')
                d5_agr_brre_ol_by, d5_agr_brre_ol_proj = process_types_and_df(d5_agr_brre_ol, 'df', time_vector)                                                                                                              
                if d5_agr_brre_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_burns_re_olife.append(d5_agr_brre_ol_by)
                # Grabbing OPEX:
                d5_agr_brre_opex = filter_dataframe(d5_agr, 'three_columns_scenarios', \
                     scenario='Quema_Residuos', scenario_2='OPEX', scenario_3=this_scen, \
                     column='Tech', column_2='Parameter', column_3='Scenario')
                d5_agr_brre_opex_by, d5_agr_brre_opex_proj = process_types_and_df(d5_agr_brre_opex, 'df', time_vector)                
                if d5_agr_brre_opex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_burns_re_opex.append(d5_agr_brre_opex_by)
                elif d5_agr_brre_opex_proj == 'user_defined':
                    for y in range(len(time_vector)):
                        list_burns_re_opex.append(d5_agr_brre_opex[time_vector[y]].iloc[0])       
                    
                # Calculate investment requirements:
                total_burns_sc_list_delta = calculate_investment_requirements(total_burns_sc_list, list_burns_re_olife, time_vector)
                
                total_burns_ce_list_delta = calculate_investment_requirements(total_burns_ce_list, list_burns_re_olife, time_vector)

                br_sc_opex = [(ucost * act)/1e6 for ucost, act in zip(list_burns_re_opex, total_burns_sc_list)]
                br_ce_opex = [(ucost * act)/1e6 for ucost, act in zip(list_burns_re_opex, total_burns_ce_list)]
                br_sc_capex = [(ucost * act)/1e6 for ucost, act in zip(list_bunrs_re_capex, total_burns_sc_list_delta)]   
                br_ce_capex = [(ucost * act)/1e6 for ucost, act in zip(list_bunrs_re_capex, total_burns_ce_list_delta)]  
                
                br_sc_opex_disc, br_ce_opex_disc, br_sc_capex_disc, br_ce_capex_disc = \
                    deepcopy(br_sc_opex), deepcopy(br_ce_opex), deepcopy(br_sc_capex), \
                    deepcopy(br_ce_capex)
                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    br_sc_opex_disc[y] *= disc_constant
                    br_ce_opex_disc[y] *= disc_constant
                    br_sc_capex_disc[y] *= disc_constant
                    br_ce_capex_disc[y] *= disc_constant
                                       
                #Storing agricultural burns emissions and costs results 
                dict_local_country[this_country].update({'Quema de residuos agrícolas de caña de azúcar [kt CH4]': deepcopy(sc_burns_emis)})
                dict_local_country[this_country].update({'Quema de residuos agrícolas de cereales [kt CH4]': deepcopy(ce_burns_emis)})
                dict_local_country[this_country].update({'CAPEX de quema de residuos agrícolas de caña de azúcar [MUSD]': deepcopy(br_sc_capex)})
                dict_local_country[this_country].update({'CAPEX de quema de residuos agrícolas de cereales [MUSD]': deepcopy(br_ce_capex)})
                dict_local_country[this_country].update({'OPEX de quema de residuos agrícolas de caña de azúcar [MUSD]': deepcopy(br_sc_opex)})
                dict_local_country[this_country].update({'OPEX de quema de residuos agrícolas de cereales [MUSD]': deepcopy(br_ce_opex)})
                dict_local_country[this_country].update({'CAPEX de quema de residuos agrícolas de caña de azúcar [MUSD] (disc)': deepcopy(br_sc_capex_disc)})
                dict_local_country[this_country].update({'CAPEX de quema de residuos agrícolas de cereales [MUSD] (disc)': deepcopy(br_ce_capex_disc)})
                dict_local_country[this_country].update({'OPEX de quema de residuos agrícolas de caña de azúcar [MUSD] (disc)': deepcopy(br_sc_opex_disc)})
                dict_local_country[this_country].update({'OPEX de quema de residuos agrícolas de cereales [MUSD] (disc)': deepcopy(br_ce_opex_disc)})

                print('Agricultural emissions have been computed!')
                            
                # Calculate future activity data:
                df3_res_data_scen = filter_dataframe(df3_res_data, 'scenario_simple', scenario=this_scen)                                                               
                df4_ef_agro_res_scen = filter_dataframe(df4_ef_agro_res, 'scenario_simple', scenario=this_scen)
                df4_ar_emi_scen = filter_dataframe(df4_ar_emi, 'scenario_simple', scenario=this_scen)
                                                           
                # General data used:
                # Population: this_pop_vector
                # GDP per capita: this_gdp_per_cap_vals

                # Solid waste:
                gen_res_sol_pc_df = filter_dataframe(df3_res_data_scen, 'scenario_simple', scenario='Generación diaria de residuos', column='Type')
                gen_res_sol_pc_by, gen_res_sol_pc_proj = process_types_and_df(gen_res_sol_pc_df, 'df', time_vector)

                # Projection growth type:
                all_vals_gen_pc_dict, total_sw_list = calculate_projection(
                    gen_res_sol_pc_df,
                    gen_res_sol_pc_proj,
                    gen_res_sol_pc_by,
                    this_gdp_pc_growth_vals,
                    this_pop_vector,
                    time_vector,
                    'fourth'
                )
                                        
                # > Emissions and costs for relleno saniario:
                grs_rs_df = filter_dataframe(df3_res_data_scen, 'scenario_simple', scenario='Relleno sanitario', column='Type') # unit: %
                grs_rs_by, grs_rs_proj = process_types_and_df(grs_rs_df, 'df', time_vector)
                grs_rs_list_delta = [0] * len(time_vector)
                if grs_rs_proj == 'user_defined':
                    grs_rs_list = [total_sw_list[0]*grs_rs_by/100]
                    for y in range(1, len(time_vector)):
                        grs_rs_list.append(total_sw_list[y] * \
                            grs_rs_df[time_vector[y]].iloc[0]/100)
                        if y > 0:
                            grs_rs_list_delta[y] = \
                                grs_rs_list[y] - grs_rs_list[y-1]
                
                grs_rs_ef_df = filter_dataframe(df4_ef_agro_res_scen, 'scenario_simple', scenario='Relleno sanitario', column='Type') # unit: %
                grs_rs_ef_by, grs_rs_ef_proj = process_types_and_df(grs_rs_ef_df, 'df', time_vector)
                if grs_rs_ef_proj == 'flat':
                    grs_rs_ef_list = [grs_rs_ef_by] * len(time_vector)
                
                # Calculating Emissions:
                grs_rs_emis = [(ef * act)/1e6 for ef, act in zip(grs_rs_ef_list, grs_rs_list)]

                # Calculating Costs:
                list_res_rs_capex = []
                list_res_rs_olife = []
                list_res_rs_opex = []
                list_res_rs_vopex = []
                list_res_rs_fopex = []
                # Grabbing CAPEX:
                d5_res_rs_capex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Relleno sanitario', scenario_3='CAPEX', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_rs_capex_by, d5_res_rs_capex_proj = process_types_and_df(d5_res_rs_capex, 'df', time_vector)                    
                if d5_res_rs_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_rs_capex.append(d5_res_rs_capex_by)
                elif d5_res_rs_capex_proj == 'user_defined':
                    for y in range(len(time_vector)):
                        list_res_rs_capex.append(d5_res_rs_capex[time_vector[y]].iloc[0])                        
                # Grabbing operational life:
                d5_res_rs_ol = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Relleno sanitario', scenario_3='Operational life', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_rs_ol_by, d5_res_rs_ol_proj = process_types_and_df(d5_res_rs_ol, 'df', time_vector)                    
                if d5_res_rs_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_rs_olife.append(d5_res_rs_ol_by)
                # Grabbing VOPEX:
                d5_res_rs_vopex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Relleno sanitario', scenario_3='Variable FOM', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_rs_vopex_by, d5_res_rs_vopex_proj = process_types_and_df(d5_res_rs_vopex, 'df', time_vector)                    
                if d5_res_rs_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_rs_vopex.append(d5_res_rs_vopex_by)
                elif d5_res_rs_vopex_proj == 'user_defined':
                    for y in range(len(time_vector)):
                        list_res_rs_vopex.append(d5_res_rs_vopex[time_vector[y]].iloc[0])
                # Grabbing FOPEX:
                d5_res_rs_fopex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Relleno sanitario', scenario_3='Fixed FOM', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_rs_fopex_by, d5_res_rs_fopex_proj = process_types_and_df(d5_res_rs_fopex, 'df', time_vector)  
                if d5_res_rs_fopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_rs_fopex.append(d5_res_rs_fopex_by)
                elif d5_res_rs_fopex_proj == 'user_defined':
                    for y in range(len(time_vector)):
                        list_res_rs_fopex.append(d5_res_rs_fopex[time_vector[y]].iloc[0])                
                # Calculate investment requirements:                
                grs_rs_opex, grs_rs_capex = calculate_investment_requirements_2(
                    grs_rs_list_delta, list_res_rs_olife, list_res_rs_vopex, list_res_rs_fopex, list_res_rs_capex, time_vector, grs_rs_list)

                # > Emissions and costs for cielo abierto:
                grs_ca_df = filter_dataframe(df3_res_data_scen, 'scenario_simple', scenario='Cielo abierto', column='Type')
                grs_ca_by, grs_ca_proj = process_types_and_df(grs_ca_df, 'df', time_vector)
                grs_ca_list_delta = [0] * len(time_vector)
                if grs_ca_proj == 'user_defined':
                    grs_ca_list = [total_sw_list[0]*grs_ca_by/100]
                    for y in range(1, len(time_vector)):
                        grs_ca_list.append(total_sw_list[y] * \
                            grs_ca_df[time_vector[y]].iloc[0]/100)
                        if y > 0:
                            grs_ca_list_delta[y] = \
                                grs_ca_list[y] - grs_ca_list[y-1]

                grs_ca_ef_df = filter_dataframe(df4_ef_agro_res_scen, 'scenario_simple', scenario='Cielo abierto', column='Type') # unit: %
                grs_ca_ef_by, grs_ca_ef_proj = process_types_and_df(grs_ca_ef_df, 'df', time_vector)
                if grs_ca_ef_proj == 'flat':
                    grs_ca_ef_list = [grs_ca_ef_by] * len(time_vector)

                # Calculating Emissions:
                grs_ca_emis = [(ef * act)/1e6 for ef, act in zip(grs_ca_ef_list, grs_ca_list)]

                # Calculating Costs:
                list_res_ca_capex = []
                list_res_ca_olife = []
                list_res_ca_opex = []
                list_res_ca_vopex = []
                list_res_ca_fopex = []
                # Grabbing CAPEX:
                d5_res_ca_capex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Cielo abierto', scenario_3='CAPEX', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_ca_capex_by, d5_res_ca_capex_proj = process_types_and_df(d5_res_ca_capex, 'df', time_vector)                    
                if d5_res_ca_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_ca_capex.append(d5_res_ca_capex_by)
                # Grabbing operational life:
                d5_res_ca_ol = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Cielo abierto', scenario_3='Operational life', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_ca_ol_by, d5_res_ca_ol_proj = process_types_and_df(d5_res_ca_ol, 'df', time_vector)                    
                if d5_res_ca_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_ca_olife.append(d5_res_ca_ol_by)
                # Grabbing VOPEX:
                d5_res_ca_vopex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Cielo abierto', scenario_3='Variable FOM', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_ca_vopex_by, d5_res_ca_vopex_proj = process_types_and_df(d5_res_ca_vopex, 'df', time_vector)
                if d5_res_ca_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_ca_vopex.append(d5_res_ca_vopex_by)
                # Grabbing FOPEX:
                d5_res_ca_fopex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Cielo abierto', scenario_3='CAPEX', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_ca_fopex_by, d5_res_ca_fopex_proj = process_types_and_df(d5_res_ca_fopex, 'df', time_vector)
                if d5_res_ca_fopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_ca_fopex.append(d5_res_ca_fopex_by)
                # Calculate investment requirements:
                grs_ca_opex, grs_ca_capex = calculate_investment_requirements_2(
                    grs_ca_list_delta, list_res_ca_olife, list_res_ca_vopex, list_res_ca_fopex, list_res_ca_capex, time_vector, grs_ca_list)

                # > Emissions and costs for reciclaje:
                grs_re_df = filter_dataframe(df3_res_data_scen, 'scenario_simple', scenario='Reciclaje', column='Type')
                grs_re_by, grs_re_proj = process_types_and_df(grs_re_df, 'df', time_vector) 
                grs_re_list_delta = [0] * len(time_vector)
                if grs_re_proj == 'user_defined':
                    grs_re_list = [total_sw_list[0]*grs_re_by/100]
                    for y in range(1, len(time_vector)):
                        grs_re_list.append(total_sw_list[y] * \
                            grs_re_df[time_vector[y]].iloc[0]/100)
                        if y > 0:
                            grs_re_list_delta[y] = \
                                grs_re_list[y] - grs_re_list[y-1]

                grs_re_ef_df = filter_dataframe(df4_ef_agro_res_scen, 'scenario_simple', scenario='Reciclaje', column='Type')
                grs_re_ef_by, grs_re_ef_proj = process_types_and_df(grs_re_ef_df, 'df', time_vector)
                if grs_re_ef_proj == 'flat':
                    grs_re_ef_list = [grs_re_ef_by] * len(time_vector)

                # Calculating Emissions:
                grs_re_emis = [(ef * act)/1e6 for ef, act in zip(grs_re_ef_list, grs_re_list)]

                # Calculating Costs:
                list_res_re_capex = []
                list_res_re_olife = []
                list_res_re_opex = []
                list_res_re_vopex = []
                list_res_re_fopex = []
                # Grabbing CAPEX:
                d5_res_re_capex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Reciclaje', scenario_3='CAPEX', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_re_capex_by, d5_res_re_capex_proj = process_types_and_df(d5_res_re_capex, 'df', time_vector)
                if d5_res_re_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_re_capex.append(d5_res_re_capex_by)
                # Grabbing operational life:
                d5_res_re_ol = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Reciclaje', scenario_3='Operational life', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_re_ol_by, d5_res_re_ol_proj = process_types_and_df(d5_res_re_ol, 'df', time_vector)                    
                if d5_res_re_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_re_olife.append(d5_res_re_ol_by)
                # Grabbing VOPEX:
                d5_res_re_vopex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Reciclaje', scenario_3='Variable FOM', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_re_vopex_by, d5_res_re_vopex_proj = process_types_and_df(d5_res_re_vopex, 'df', time_vector)
                if d5_res_re_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_re_vopex.append(d5_res_re_vopex_by)
                # Grabbing FOPEX:
                d5_res_re_fopex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Reciclaje', scenario_3='Fixed FOM', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_re_fopex_by, d5_res_re_fopex_proj = process_types_and_df(d5_res_re_fopex, 'df', time_vector)
                if d5_res_re_fopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_re_fopex.append(d5_res_re_fopex_by)
                # Calculate investment requirements:
                grs_re_opex, grs_re_capex = calculate_investment_requirements_2(
                    grs_re_list_delta, list_res_re_olife, list_res_re_vopex, list_res_re_fopex, list_res_re_capex, time_vector, grs_re_list)
                
                # > Emissions and costs for compostaje:
                grs_comp_df = filter_dataframe(df3_res_data_scen, 'scenario_simple', scenario='Compostaje', column='Type')
                grs_comp_by, grs_comp_proj = process_types_and_df(grs_comp_df, 'df', time_vector)
                grs_comp_list_delta = [0] * len(time_vector)
                if grs_comp_proj == 'user_defined':
                    grs_comp_list = [total_sw_list[0]*grs_comp_by/100]
                    for y in range(1, len(time_vector)):
                        grs_comp_list.append(total_sw_list[y] * \
                            grs_comp_df[time_vector[y]].iloc[0]/100)
                        if y > 0:
                            grs_comp_list_delta[y] = \
                                grs_comp_list[y] - grs_comp_list[y-1]

                grs_comp_ef_df = filter_dataframe(df4_ef_agro_res_scen, 'scenario_simple', scenario='Compostaje', column='Type') # unit: %
                grs_comp_ef_by, grs_comp_ef_proj = process_types_and_df(grs_comp_ef_df, 'df', time_vector)
                if grs_comp_ef_proj == 'flat':
                    grs_comp_ef_list = [grs_comp_ef_by] * len(time_vector)

                # Calculating Emissions:
                grs_comp_emis = [(ef * act)/1e6 for ef, act in zip(grs_comp_ef_list, grs_comp_list)]

                # Calculating Costs:
                list_res_comp_capex = []
                list_res_comp_olife = []
                list_res_comp_opex = []
                list_res_comp_vopex = []
                list_res_comp_fopex = []
                # Grabbing CAPEX:
                d5_res_comp_capex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Compostaje', scenario_3='CAPEX', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_comp_capex_by, d5_res_comp_capex_proj = process_types_and_df(d5_res_comp_capex, 'df', time_vector)
                if d5_res_comp_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_comp_capex.append(d5_res_comp_capex_by)
                # Grabbing operational life:
                d5_res_comp_ol = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Compostaje', scenario_3='Operational life', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_comp_ol_by, d5_res_comp_ol_proj = process_types_and_df(d5_res_comp_ol, 'df', time_vector)    
                if d5_res_comp_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_comp_olife.append(d5_res_comp_ol_by)
                # Grabbing VOPEX:
                d5_res_comp_vopex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Compostaje', scenario_3='Variable FOM', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_comp_vopex_by, d5_res_comp_vopex_proj = process_types_and_df(d5_res_comp_vopex, 'df', time_vector)
                if d5_res_comp_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_comp_vopex.append(d5_res_comp_vopex_by)
                # Grabbing FOPEX:
                d5_res_comp_fopex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Compostaje', scenario_3='Fixed FOM', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_comp_fopex_by, d5_res_comp_fopex_proj = process_types_and_df(d5_res_comp_fopex, 'df', time_vector)
                if d5_res_comp_fopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_comp_fopex.append(d5_res_comp_fopex_by)
                # Calculate investment requirements:
                grs_comp_opex, grs_comp_capex = calculate_investment_requirements_2(
                    grs_comp_list_delta, list_res_comp_olife, list_res_comp_vopex, list_res_comp_fopex, list_res_comp_capex, time_vector, grs_comp_list)

                # > Emissions and costs for dig. anaeróbica:
                grs_da_df = filter_dataframe(df3_res_data_scen, 'scenario_simple', scenario='Digestión anaeróbica para biogas', column='Type')
                grs_da_by, grs_da_proj = process_types_and_df(grs_da_df, 'df', time_vector)
                grs_da_list_delta = [0] * len(time_vector)
                if grs_da_proj == 'user_defined':
                    grs_da_list = [total_sw_list[0]*grs_da_by/100]
                    for y in range(1, len(time_vector)):
                        grs_da_list.append(total_sw_list[y] * \
                            grs_da_df[time_vector[y]].iloc[0]/100)
                        if y > 0:
                            grs_da_list_delta[y] = \
                                grs_da_list[y] - grs_da_list[y-1]

                grs_da_ef_df = filter_dataframe(df4_ef_agro_res_scen, 'scenario_simple', scenario='Digestión anaeróbica para biogas', column='Type') # unit: %
                grs_da_ef_by, grs_da_ef_proj = process_types_and_df(grs_comp_ef_df, 'df', time_vector)
                if grs_da_ef_proj == 'flat':
                    grs_da_ef_list = [grs_da_ef_by] * len(time_vector)

                # Calculating Emissions:
                grs_da_emis = [(ef * act)/1e6 for ef, act in zip(grs_da_ef_list, grs_da_list)]

                # Calculating Costs:
                list_res_da_capex = []
                list_res_da_olife = []
                list_res_da_opex = []
                list_res_da_vopex = []
                list_res_da_fopex = []
                # Grabbing CAPEX:
                d5_res_da_capex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Digestión anaeróbica para biogas', scenario_3='CAPEX', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_da_capex_by, d5_res_da_capex_proj = process_types_and_df(d5_res_da_capex, 'df', time_vector)                        
                if d5_res_da_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_da_capex.append(d5_res_da_capex_by)
                # Grabbing operational life:
                d5_res_da_ol = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Digestión anaeróbica para biogas', scenario_3='Operational life', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_da_ol_by, d5_res_da_ol_proj = process_types_and_df(d5_res_da_ol, 'df', time_vector)    
                if d5_res_da_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_da_olife.append(d5_res_da_ol_by)
                # Grabbing VOPEX:
                d5_res_da_vopex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Digestión anaeróbica para biogas', scenario_3='Variable FOM', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_da_vopex_by, d5_res_da_vopex_proj = process_types_and_df(d5_res_da_vopex, 'df', time_vector)    
                if d5_res_da_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_da_vopex.append(d5_res_da_vopex_by)
                # Grabbing FOPEX:
                d5_res_da_fopex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Digestión anaeróbica para biogas', scenario_3='Fixed FOM', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_da_fopex_by, d5_res_da_fopex_proj = process_types_and_df(d5_res_da_fopex, 'df', time_vector)    
                if d5_res_da_fopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_da_fopex.append(d5_res_da_fopex_by)
                # Calculate investment requirements:
                grs_da_opex, grs_da_capex = calculate_investment_requirements_2(
                    grs_da_list_delta, list_res_da_olife, list_res_da_vopex, list_res_da_fopex, list_res_da_capex, time_vector, grs_da_list)

                # > Emissions and costs for incineración de residuos:
                grs_ir_df = filter_dataframe(df3_res_data_scen, 'scenario_simple', scenario='Incineración de residuos', column='Type')
                grs_ir_by, grs_ir_proj = process_types_and_df(grs_ir_df, 'df', time_vector)
                grs_ir_list_delta = [0] * len(time_vector)
                if grs_ir_proj == 'user_defined':
                    grs_ir_list = [total_sw_list[0]*grs_ir_by/100]
                    for y in range(1, len(time_vector)):
                        grs_ir_list.append(total_sw_list[y] * \
                            grs_ir_df[time_vector[y]].iloc[0]/100)
                        if y > 0:
                            grs_ir_list_delta[y] = \
                                grs_ir_list[y] - grs_ir_list[y-1]

                grs_ir_ef_df = filter_dataframe(df4_ef_agro_res_scen, 'scenario_simple', scenario='Incineración de residuos', column='Type') # unit: %
                grs_ir_ef_by, grs_ir_ef_proj = process_types_and_df(grs_ir_ef_df, 'df', time_vector)
                if grs_ir_ef_proj == 'flat':
                    grs_ir_ef_list = [grs_ir_ef_by] * len(time_vector)

                # Calculating Emissions:
                grs_ir_emis = [(ef * act)/1e6 for ef, act in zip(grs_ir_ef_list, grs_ir_list)]

                # Calculating Costs:
                list_res_ir_capex = []
                list_res_ir_olife = []
                list_res_ir_opex = []
                list_res_ir_vopex = []
                list_res_ir_fopex = []
                # Grabbing CAPEX:
                d5_res_ir_capex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Incineración de residuos', scenario_3='CAPEX', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_ir_capex_by, d5_res_ir_capex_proj = process_types_and_df(d5_res_ir_capex, 'df', time_vector)    
                if d5_res_ir_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_ir_capex.append(d5_res_ir_capex_by)
                # Grabbing operational life:
                d5_res_ir_ol = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Incineración de residuos', scenario_3='Operational life', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_ir_ol_by, d5_res_ir_ol_proj = process_types_and_df(d5_res_ir_ol, 'df', time_vector)    
                if d5_res_ir_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_ir_olife.append(d5_res_ir_ol_by)
                # Grabbing VOPEX:
                d5_res_ir_vopex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Incineración de residuos', scenario_3='Variable FOM', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_ir_vopex_by, d5_res_ir_vopex_proj = process_types_and_df(d5_res_ir_vopex, 'df', time_vector)
                if d5_res_ir_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_ir_vopex.append(d5_res_ir_vopex_by)
                # Grabbing FOPEX:
                d5_res_ir_fopex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Incineración de residuos', scenario_3='Fixed FOM', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_ir_fopex_by, d5_res_ir_fopex_proj = process_types_and_df(d5_res_ir_fopex, 'df', time_vector)
                if d5_res_ir_fopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_ir_fopex.append(d5_res_ir_fopex_by)
                # Calculate investment requirements:
                grs_ir_opex, grs_ir_capex = calculate_investment_requirements_2(
                    grs_ir_list_delta, list_res_ir_olife, list_res_ir_vopex, list_res_ir_fopex, list_res_ir_capex, time_vector, grs_ir_list)

                # > Emissions and costs for incinaración abierta de residuos:
                grs_iar_df = filter_dataframe(df3_res_data_scen, 'scenario_simple', scenario='Incineración abierta de residuos', column='Type')
                grs_iar_by, grs_iar_proj = process_types_and_df(grs_iar_df, 'df', time_vector)
                grs_iar_list_delta = [0] * len(time_vector)
                if grs_iar_proj == 'user_defined':
                    grs_iar_list = [total_sw_list[0]*grs_iar_by/100]
                    for y in range(1, len(time_vector)):
                        grs_iar_list.append(total_sw_list[y] * \
                            grs_iar_df[time_vector[y]].iloc[0]/100)
                        if y > 0:
                            grs_iar_list_delta[y] = \
                                grs_iar_list[y] - grs_iar_list[y-1]

                grs_iar_ef_df = filter_dataframe(df4_ef_agro_res_scen, 'scenario_simple', scenario='Incineración abierta de residuos', column='Type') # unit: %
                grs_iar_ef_by, grs_iar_ef_proj = process_types_and_df(grs_iar_ef_df, 'df', time_vector)
                if grs_iar_ef_proj == 'flat':
                    grs_iar_ef_list = [grs_iar_ef_by] * len(time_vector)

                # Calculating Emissions:
                grs_iar_emis = [(ef * act)/1e6 for ef, act in zip(grs_iar_ef_list, grs_iar_list)]

                # Calculating Costs:
                list_res_iar_capex = []
                list_res_iar_olife = []
                list_res_iar_opex = []
                list_res_iar_vopex = []
                list_res_iar_fopex = []
                # Grabbing CAPEX:
                d5_res_iar_capex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Incineración abierta de residuos', scenario_3='CAPEX', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_iar_capex_by, d5_res_iar_capex_proj = process_types_and_df(d5_res_iar_capex, 'df', time_vector)
                if d5_res_iar_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_iar_capex.append(d5_res_iar_capex_by)
                # Grabbing operational life:
                d5_res_iar_ol = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Incineración abierta de residuos', scenario_3='CAPEX', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_iar_ol_by, d5_res_iar_ol_proj = process_types_and_df(d5_res_iar_ol, 'df', time_vector)
                if d5_res_iar_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_iar_olife.append(d5_res_iar_ol_by)
                # Grabbing VOPEX:
                d5_res_iar_vopex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Incineración abierta de residuos', scenario_3='Variable FOM', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_iar_vopex_by, d5_res_iar_vopex_proj = process_types_and_df(d5_res_iar_vopex, 'df', time_vector)
                if d5_res_iar_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_iar_vopex.append(d5_res_iar_vopex_by)
                # Grabbing FOPEX:
                d5_res_iar_fopex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Incineración abierta de residuos', scenario_3='Fixed FOM', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_iar_fopex_by, d5_res_iar_fopex_proj = process_types_and_df(d5_res_iar_fopex, 'df', time_vector)
                if d5_res_iar_fopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_iar_fopex.append(d5_res_iar_fopex_by)
                # Calculate investment requirements:
                grs_iar_opex, grs_iar_capex = calculate_investment_requirements_2(
                    grs_iar_list_delta, list_res_iar_olife, list_res_iar_vopex, list_res_iar_fopex, list_res_iar_capex, time_vector, grs_iar_list)

                # > Emissions and costs for residuos sólidos no gestionados:
                grs_rsng_df = filter_dataframe(df3_res_data_scen, 'scenario_simple', scenario='Residuos sólidos no gestionados', column='Type')
                grs_rsng_by, grs_rsng_proj = process_types_and_df(grs_rsng_df, 'df', time_vector)
                grs_rsng_list_delta = [0] * len(time_vector)
                if grs_rsng_proj == 'user_defined':
                    grs_rsng_list = [total_sw_list[0]*grs_rsng_by/100] #ton
                    for y in range(1, len(time_vector)):
                        grs_rsng_list.append(total_sw_list[y] * \
                            grs_rsng_df[time_vector[y]].iloc[0]/100)
                        if y > 0:
                            grs_rsng_list_delta[y] = \
                                grs_rsng_list[y] - grs_rsng_list[y-1]

                grs_rsng_ef_df = filter_dataframe(df4_ef_agro_res_scen, 'scenario_simple', scenario='Residuos sólidos no gestionados', column='Type') # unit: %
                grs_rsng_ef_by, grs_rsng_ef_proj = process_types_and_df(grs_rsng_ef_df, 'df', time_vector) # kg CH4/Gg waste-año
                if grs_rsng_ef_proj == 'flat':
                    grs_rsng_ef_list = [grs_rsng_ef_by] * len(time_vector)

                # Calculating Emissions:
                grs_rsng_emis = [(ef * act)/1e9 for ef, act in zip(grs_rsng_ef_list, grs_rsng_list)]

                # Calculating Costs:
                list_res_rsng_capex = []
                list_res_rsng_olife = []
                list_res_rsng_opex = []
                list_res_rsng_vopex = []
                list_res_rsng_fopex = []
                # Grabbing CAPEX:
                d5_res_rsng_capex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Residuos sólidos no gestionados', scenario_3='CAPEX', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_rsng_capex_by, d5_res_rsng_capex_proj = process_types_and_df(d5_res_rsng_capex, 'df', time_vector)
                if d5_res_rsng_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_rsng_capex.append(d5_res_rsng_capex_by)
                # Grabbing operational life:
                d5_res_rsng_ol = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Residuos sólidos no gestionados', scenario_3='Operational life', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_rsng_ol_by, d5_res_rsng_ol_proj = process_types_and_df(d5_res_rsng_ol, 'df', time_vector)    
                if d5_res_rsng_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_rsng_olife.append(d5_res_rsng_ol_by)
                # Grabbing VOPEX:
                d5_res_rsng_vopex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Residuos sólidos no gestionados', scenario_3='Variable FOM', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_rsng_vopex_by, d5_res_rsng_vopex_proj = process_types_and_df(d5_res_rsng_vopex, 'df', time_vector)    
                if d5_res_rsng_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_rsng_vopex.append(d5_res_rsng_vopex_by)
                # Grabbing FOPEX:
                d5_res_rsng_fopex = filter_dataframe(d5_res, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Residuos sólidos no gestionados', scenario_3='Fixed FOM', \
                      column='Scenario', column_2='Tech', column_3='Parameter')
                d5_res_rsng_fopex_by, d5_res_rsng_fopex_proj = process_types_and_df(d5_res_rsng_fopex, 'df', time_vector)    
                if d5_res_rsng_fopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_rsng_fopex.append(d5_res_rsng_fopex_by)
                # Calculate investment requirements:
                grs_rsng_opex, grs_rsng_capex = calculate_investment_requirements_2(
                    grs_rsng_list_delta, list_res_rsng_olife, list_res_rsng_vopex, list_res_rsng_fopex, list_res_rsng_capex, time_vector, grs_rsng_list)

                # > Emissions and costs for recuperación del metano del relleno
                mask_ch4_landfill_rec = (df3_res_data_scen['Type'] == 'Metano del relleno sanitario extraído')

                # Wastewater treatment:
                gen_dbo_pc_df = filter_dataframe(df3_res_data_scen, 'scenario_simple', scenario='DBO per cápita', column='Type') # unit: g/persona/año
                gen_dbo_pc_by, gen_dbo_pc_proj = process_types_and_df(gen_dbo_pc_df, 'df', time_vector)
                if gen_dbo_pc_proj == 'flat':
                    gen_dbo_pc_list = [gen_dbo_pc_by] * len(time_vector)
                elif gen_dbo_pc_proj == 'user_defined':
                    gen_dbo_pc_list = gen_dbo_pc_df[time_vector].iloc[0].tolist()

                # Population:
                '''
                Just call:
                this_pop_vector
                '''

                # Get treated waters
                tre_wat_df = filter_dataframe(df3_res_data_scen, 'scenario_simple', scenario='Tratamiento de aguas residuales domésticas', column='Type') # unit: %
                tre_wat_by, tre_wat_proj = process_types_and_df(tre_wat_df, 'df', time_vector)
                if tre_wat_proj == 'flat':
                    tre_wat_list = [tre_wat_by] * len(time_vector)
                elif tre_wat_proj == 'user_defined':
                    tre_wat_list = tre_wat_df[time_vector].iloc[0].tolist()

                # Multiply Pop, DBO, and share of treated water [kg DBO/año]
                tre_wat_kg = [(a * 365 * b * 1e6 * c)/(1000 * 100) for a, b, c in zip(gen_dbo_pc_list, this_pop_vector, tre_wat_list)]

                unit_capex_tre_wat = unpack_values_df_2(
                    d5_res, "Tech", "Parameter",
                    "Tratamiento de aguas residuales domésticas",
                    "CAPEX", time_vector, this_scen)  # $/persona
                unit_fopex_tre_wat = unpack_values_df_2(
                    d5_res, "Tech", "Parameter",
                    "Tratamiento de aguas residuales domésticas",
                    "Fixed FOM", time_vector, this_scen)  # % del CAPEX
                unit_vopex_tre_wat = unpack_values_df_2(
                    d5_res, "Tech", "Parameter",
                    "Tratamiento de aguas residuales domésticas",
                    "Variable FOM", time_vector, this_scen)  # $/persona
                ol_tre_wat = unpack_values_df_2(
                    d5_res, "Tech", "Parameter",
                    "Tratamiento de aguas residuales domésticas",
                    "Operational life", time_vector, this_scen)  # years

                this_pop_vector_delta = [0]
                for y in range(1, len(time_vector)):
                    this_pop_vector_delta.append(this_pop_vector[y] - this_pop_vector[y-1])
                total_capex_tre_wat = [unit_capex * pop * share/100 for unit_capex, pop, share in zip(unit_capex_tre_wat, this_pop_vector_delta, tre_wat_list)]
                for y in range(int(ol_tre_wat[0]), len(time_vector)):
                    total_capex_tre_wat[y] += total_capex_tre_wat[y - int(ol_tre_wat[y])]

                total_fopex_tre_wat = [a*b/100 for a, b in zip(unit_fopex_tre_wat, total_capex_tre_wat)]
                total_vopex_tre_wat = [unit_vopex * pop * share/100 for unit_vopex, pop, share in zip(unit_vopex_tre_wat, this_pop_vector, tre_wat_list)]
                
                total_capex_tre_wat_disc, total_fopex_tre_wat_disc, total_vopex_tre_wat_disc = \
                    deepcopy(total_capex_tre_wat), deepcopy(total_fopex_tre_wat), deepcopy(total_vopex_tre_wat)
                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    total_capex_tre_wat_disc[y] *= disc_constant
                    total_fopex_tre_wat_disc[y] *= disc_constant
                    total_vopex_tre_wat_disc[y] *= disc_constant
            
                # Get untreated waters
                unt_wat_df = filter_dataframe(df3_res_data_scen, 'scenario_simple', scenario='Aguas no tratadas', column='Type') # unit: %
                unt_wat_by, unt_wat_proj = process_types_and_df(unt_wat_df, 'df', time_vector)
                if unt_wat_proj == 'flat':
                    unt_wat_list = [unt_wat_by] * len(time_vector)
                elif unt_wat_proj == 'user_defined':
                    unt_wat_list = unt_wat_df[time_vector].iloc[0].tolist()

                # Multiply Pop, DBO, and share of untreated water [kg DBO/ año]
                unt_wat_kg = [(a * 365 * b*1e6 * c)/(1000 * 100) for a, b, c in zip(gen_dbo_pc_list, this_pop_vector, unt_wat_list)]

                # Read urbanization, emission factors, and tech utilization
                ar_urb_df = filter_dataframe(df4_ar_emi_scen, 'scenario_simple', scenario='Grado de urbanización', column='Parameter')
                ar_urb_tech, _, _ = process_types_and_df(ar_urb_df, 'only_types', time_vector, column='Urbanization')
                dict_ar_urb_tech = {}
                for aut in ar_urb_tech:
                    ar_urb_df_t = ar_urb_df.loc[(ar_urb_df['Urbanization'] == aut)]
                    ar_urb_by, ar_urb_proj = process_types_and_df(ar_urb_df_t, 'df', time_vector)
                    if ar_urb_proj == 'flat':
                        ar_urb_list = [ar_urb_by] * len(time_vector)
                    elif ar_urb_proj == 'user_defined':
                        ar_urb_list = ar_urb_df_t[time_vector].iloc[0].tolist()
                    dict_ar_urb_tech.update({aut: ar_urb_list})
                ar_ef_df = filter_dataframe(df4_ar_emi_scen, 'scenario_simple', scenario='Factor de emisión', column='Parameter')
                ar_ef_tech, _, _ = process_types_and_df(ar_ef_df, 'only_types', time_vector, column='Technology')
                dict_ar_ef_tech = {}
                for aut in ar_ef_tech:
                    ar_ef_df_t = ar_ef_df.loc[(ar_ef_df['Technology'] == aut)]
                    ar_ef_by, ar_ef_proj = process_types_and_df(ar_ef_df_t, 'df', time_vector)
                    if ar_ef_proj == 'flat':
                        ar_ef_list = [ar_ef_by] * len(time_vector)
                    elif ar_ef_proj == 'user_defined':
                        ar_ef_list = ar_ef_df_t[time_vector].iloc[0].tolist()
                    dict_ar_ef_tech.update({aut: ar_ef_list})

                # Calcualte the wieghted emissionf actor
                ef_weighted_list = [0] * len(time_vector)

                artechuti_df = filter_dataframe(df4_ar_emi_scen, 'scenario_simple', scenario='Grado de utilización de tecnología', column='Parameter')
                for auti in range(len(list(artechuti_df.index))):
                    artechuti_df_tech = artechuti_df['Technology'].iloc[auti]
                    artechuti_df_urb = artechuti_df['Urbanization'].iloc[auti]
                    artechuti_df_proj = artechuti_df['Projection'].iloc[auti]
                    artechuti_df_by = artechuti_df[time_vector[0]].iloc[auti]
                    if artechuti_df_proj == 'flat':
                        artechuti_list = [artechuti_df_by] * len(time_vector)
                    elif artechuti_df_proj == 'user_defined':
                        artechuti_list = artechuti_df[time_vector].iloc[auti].tolist()
                                            
                    ar_urb_list_call = dict_ar_urb_tech[artechuti_df_urb]
                    ar_ef_list_call = dict_ar_ef_tech[artechuti_df_tech]
                    
                    ar_factor_ef = [a*b*c for a, b, c in zip(ar_urb_list_call, artechuti_list, ar_ef_list_call)]
                    ef_weighted_list = [a + b for a, b in zip(ef_weighted_list, ar_factor_ef)]
                
                # Calculate emissions:
                all_wat_kg = [a + b for a, b in zip(tre_wat_kg, unt_wat_kg)]
                waste_water_emissions = [(a * b)/1e6 for a, b in zip(ef_weighted_list, all_wat_kg)]  # in kt CH4
                
                print('Wastewater emissions have been computed!')
                print('Wastewater costs have been computed!')
                                
                # DISCOUNTING ALL COSTS
                techs_sw = [
                    'Relleno sanitario',
                    'Cielo abierto',
                    'Reciclaje',
                    'Compostaje',
                    'Digestión anaeróbica para biogas',
                    'Incineración de residuos',
                    'Incineración abierta de residuos',
                    'Residuos sólidos no gestionados']

                opex_sw, capex_sw = {}, {}

                opex_sw.update({'Relleno sanitario': grs_rs_opex})
                capex_sw.update({'Relleno sanitario': grs_rs_capex})
                opex_sw.update({'Cielo abierto': grs_ca_opex})
                capex_sw.update({'Cielo abierto': grs_ca_capex})
                opex_sw.update({'Reciclaje': grs_re_opex})
                capex_sw.update({'Reciclaje': grs_re_capex})
                opex_sw.update({'Compostaje': grs_comp_opex})
                capex_sw.update({'Compostaje': grs_comp_capex})
                opex_sw.update({'Digestión anaeróbica para biogas': grs_da_opex})
                capex_sw.update({'Digestión anaeróbica para biogas': grs_da_capex})
                opex_sw.update({'Incineración de residuos': grs_ir_opex})
                capex_sw.update({'Incineración de residuos': grs_ir_capex})
                opex_sw.update({'Incineración abierta de residuos': grs_iar_opex})
                capex_sw.update({'Incineración abierta de residuos': grs_iar_capex})
                opex_sw.update({'Residuos sólidos no gestionados': grs_rsng_opex})
                capex_sw.update({'Residuos sólidos no gestionados': grs_rsng_capex})

                opex_sw_disc, capex_sw_disc = deepcopy(opex_sw), deepcopy(capex_sw)

                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    for tech in techs_sw:
                        opex_sw_disc[tech][y] *= disc_constant
                        capex_sw_disc[tech][y] *= disc_constant
                
                # STORING ALL COSTS
                dict_local_country[this_country].update({'Solid waste CAPEX [MUSD]': deepcopy(capex_sw)})
                dict_local_country[this_country].update({'Solid waste CAPEX [MUSD] (disc)': deepcopy(capex_sw_disc)})
                dict_local_country[this_country].update({'Solid waste OPEX [MUSD]': deepcopy(opex_sw)})
                dict_local_country[this_country].update({'Solid waste OPEX [MUSD] (disc)': deepcopy(opex_sw_disc)})
                dict_local_country[this_country].update({'CAPEX para aguas residuales tratadas [MUSD]': deepcopy(total_capex_tre_wat)})
                dict_local_country[this_country].update({'OPEX fijo para aguas residuales tratadas [MUSD]': deepcopy(total_fopex_tre_wat)})
                dict_local_country[this_country].update({'OPEX variable para aguas residuales tratadas [MUSD]': deepcopy(total_vopex_tre_wat)})
                dict_local_country[this_country].update({'CAPEX para aguas residuales tratadas [MUSD] (disc)': deepcopy(total_capex_tre_wat_disc)})
                dict_local_country[this_country].update({'OPEX fijo para aguas residuales tratadas [MUSD] (disc)': deepcopy(total_fopex_tre_wat_disc)})
                dict_local_country[this_country].update({'OPEX variable para aguas residuales tratadas [MUSD] (disc)': deepcopy(total_vopex_tre_wat_disc)})                
                
                # STORING EMISSIONS
                emis_sw = {}
                emis_sw.update({'Relleno sanitario': grs_rs_emis})
                emis_sw.update({'Cielo abierto': grs_ca_emis})
                emis_sw.update({'Reciclaje': grs_re_emis})
                emis_sw.update({'Compostaje': grs_comp_emis})
                emis_sw.update({'Digestión anaeróbica para biogas': grs_da_emis})
                emis_sw.update({'Incineración de residuos': grs_ir_emis})
                emis_sw.update({'Incineración abierta de residuos': grs_iar_emis})
                emis_sw.update({'Residuos sólidos no gestionados': grs_rsng_emis})

                dict_local_country[this_country].update({'Solid waste emissions [kt CH4]': deepcopy(emis_sw)})
                dict_local_country[this_country].update({'Emisiones de aguas residuales tratadas [kt CH4]': deepcopy(waste_water_emissions)})

                print('Waste emissions have been computed!')
                print('Waste costs have been computed!')

            """
            INSTRUCTIONS:
            1) Load inputs for RAC sector
            2) Create projections for RAC sector
            3) Estimate costs and emissions 
            4) Store the variables to print
            """
            if model_rac:
                #Load demanad for all sector
                df4_rac_data_scen = filter_dataframe(df4_rac_data, 'scenario_simple', scenario=this_scen)
                # Project the demand:
                df4_rac_data_scen_dem = filter_dataframe(df4_rac_data, 'two_columns_scenarios', \
                                    scenario=this_scen, scenario_2='Demand', \
                                    column='Scenario', column_2='Parameter')                    
                _, types_projection_dem_rac, types_by_vals_dem_rac = process_types_and_df(df4_rac_data_scen_dem, 'only_types_2', time_vector, column='Projection')
                
                # Projection growth type:
                all_vals_gen_pc_dict, total_dem_rac_list = calculate_projection(
                    df4_rac_data_scen_dem,
                    types_projection_dem_rac,
                    types_by_vals_dem_rac,
                    this_gdp_pc_growth_vals,
                    this_pop_vector,
                    time_vector,
                    'third'
                )
                                
                #Load shares by type of refrigerant for AC subsector
                df4_rac_data_scen_shares_ac = filter_dataframe(df4_rac_data, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Shares', scenario_3='AC', \
                      column='Scenario', column_2='Parameter', column_3='Subsector')
                types_shares_ac, types_projection_shares_ac, types_by_vals_shares_ac = process_types_and_df(df4_rac_data_scen_shares_ac, 'only_types', time_vector, column='Refrigerante')
                
                ac_shares_ac_out_dict = {}
                all_vals_gen_pc_dict = {}
                for y in range(len(time_vector)):
                    all_vals_gen_pc_dict.update({time_vector[y]:df4_rac_data_scen_shares_ac[time_vector[y]].tolist()})
                for l in range(len(types_shares_ac)):
                    this_rac = types_shares_ac[l]
                    this_proj = types_projection_shares_ac[l]
                    this_by_val = types_by_vals_shares_ac[l]
                    this_val_list = []
                    if this_proj == 'flat':
                        this_val_list = [this_by_val]*len(time_vector)
                        ac_shares_ac_out_dict.update({this_rac: this_val_list})
                    elif this_proj == 'user_defined':
                        total_shares_list = []
                        for y in range(len(time_vector)):
                            total_shares_list.append(all_vals_gen_pc_dict[time_vector[y]][l])
                        ac_shares_ac_out_dict.update({this_rac: total_shares_list})
                             
                # Calculate total refrigerant (ton):
                ac_dem_by_type_dict = calculate_values(types_shares_ac, ac_shares_ac_out_dict, None, total_dem_rac_list, calculation_type='demand')

                #Leakage factor (adim)
                df4_leakage_factor = filter_dataframe(df4_rac_data, 'two_columns_scenarios', \
                                    scenario=this_scen, scenario_2='Factor_fugas', \
                                    column='Scenario', column_2='Parameter')                    
                _, types_projection_leakage_factor, types_by_vals_leakage_factor = process_types_and_df(df4_leakage_factor, 'only_types_2', time_vector, column='Projection')
            
                # Projection mode for leakage factor
                all_vals_gen_pc_dict = {}
                for y in range(len(time_vector)):
                    all_vals_gen_pc_dict.update({time_vector[y]:df4_leakage_factor[time_vector[y]].iloc[0]})
                if  types_projection_leakage_factor == 'user_defined':
                    total_lf_list = []
                    gen_lf_pc = []
                    for y in range(len(time_vector)):
                        total_lf_list.append(df4_leakage_factor[time_vector[y]].iloc[0])
                        #total_lf_list.append(gen_lf_pc[-1]*this_pop_vector[y])
                elif types_projection_leakage_factor == 'flat':
                    total_lf_list = [types_by_vals_leakage_factor] * len(time_vector)
                                
                # Load Global Warming Potencial (adim)
                types_emis_GWP, types_projection_emi_GWP, types_by_vals_semis_GWP = process_types_and_df(df4_rac_emi, 'only_types', time_vector, column='Nombre común')
            
                # Projection for GWP
                rac_GWP_out_dict = generate_projection_dict(
                    types_emis_GWP, types_projection_emi_GWP, types_by_vals_semis_GWP, 
                    df4_rac_emi, time_vector, 'first')
                
                ac_dem_by_type_dict = calculate_values(types_shares_ac, ac_shares_ac_out_dict, None, total_dem_rac_list, calculation_type='demand')                
                                      
                # Estimating emisions (kton de CO2 eq)
                ac_emi_out_dict = calculate_values(types_shares_ac, rac_GWP_out_dict, ac_dem_by_type_dict, total_dem_rac_list, total_lf_list, 'emissions')

                # Load price (USD/kg)
                types_rac, types_projection_rac_price, types_by_vals_rac_price = process_types_and_df(d5_rac, 'only_types', time_vector, column='Refrigerante')                
                
                #Projection for price
                rac_price_out_dict = generate_short_projection(
                    types_rac, types_projection_rac_price, types_by_vals_rac_price, 
                    time_vector)
                                
                # Estimating prices (MUSD)
                ac_prices_ac_out_dict = {}
                ac_prices_ac_out_dict_disc = {}
                for l in range(len(types_shares_ac)):
                    ac_prices_ac = [(price*act)/1e3 for price, act in \
                                zip(rac_price_out_dict[types_shares_ac[l]], \
                                    ac_dem_by_type_dict[types_shares_ac[l]])]
                    list_price_ac_disc = []
                    for y in range(len(time_vector)):
                        his_year = int(time_vector[y])
                        disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                        local_disc = ac_prices_ac[y]*disc_constant
                        list_price_ac_disc.append(local_disc)
                    ac_prices_ac_out_dict.update({types_shares_ac[l]:ac_prices_ac})    
                    ac_prices_ac_out_dict_disc.update({types_shares_ac[l]:list_price_ac_disc})

                ac_prices_ac_out_dict = calculate_values(types_shares_ac, rac_price_out_dict, ac_dem_by_type_dict, total_dem_rac_list, calculation_type='prices')
  
                # Estimating discounted values 
                ac_prices_ac_out_dict_disc = {}
                for key, values in ac_prices_ac_out_dict.items():
                    discounted_prices = [value / ((1 + r_rate/100) ** t) for t, value in enumerate(values, 0)]
                    ac_prices_ac_out_dict_disc[key] = discounted_prices
                                    
                # Storing costs and emissions for AC subsector 
                dict_local_country[this_country].update({'Emisiones para subsector de AC [kt CO2 eq]': deepcopy(ac_emi_out_dict)})
                dict_local_country[this_country].update({'Costos para subsector de AC [MUSD]': deepcopy(ac_prices_ac_out_dict)})
                dict_local_country[this_country].update({'Costos para subsector de AC [MUSD] (disc)': deepcopy(ac_prices_ac_out_dict_disc)})
                
                #Shares for Refrigeration
                df4_rac_data_scen_shares_ref = filter_dataframe(df4_rac_data, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Shares', scenario_3='Refrigeración', \
                      column='Scenario', column_2='Parameter', column_3='Subsector')
                types_shares_ref, types_projection_shares_ref, types_by_vals_shares_ref = process_types_and_df(df4_rac_data_scen_shares_ref, 'only_types', time_vector, column='Refrigerante')
                
               # Projection for Shares for Refrigeration
                rac_shares_ref_out_dict = generate_projection_dict(
                    types_shares_ref, types_projection_shares_ref, types_by_vals_shares_ref, 
                    df4_rac_data_scen_shares_ref, time_vector, 'first')
                                         
                # Calculate total refrigerant (ton):
                ref_dem_by_type_dict = calculate_values(types_shares_ref, rac_shares_ref_out_dict, None, total_dem_rac_list, calculation_type='demand')                

                #Estimating emisions (kton de CO2 eq)
                ref_emi_out_dict = calculate_ref_values(
                    types_shares_ref, rac_GWP_out_dict, ref_dem_by_type_dict, total_lf_list, 'emissions')
                
                #Estimating prices (MUSD)
                ref_prices_out_dict = calculate_ref_values(
                    types_shares_ref, rac_price_out_dict, ref_dem_by_type_dict, calculation_type='prices')

                #Estimating discounted values 
                ref_prices_out_dict_disc = {}
                for key, values in ref_prices_out_dict.items():
                    discounted_prices = [value / ((1 + r_rate/100) ** t) for t, value in enumerate(values, 0)]
                    ref_prices_out_dict_disc[key] = discounted_prices
                    
                #Storing costs and emissions for refrigerants subsector 
                dict_local_country[this_country].update({'Emisiones para subsector de refrigeración [kt CO2 eq]': deepcopy(ref_emi_out_dict)})
                dict_local_country[this_country].update({'Costos para subsector de refrigeración [MUSD]': deepcopy(ref_prices_out_dict)})    
                dict_local_country[this_country].update({'Costos para subsector de refrigeración [MUSD] (disc)': deepcopy(ref_prices_out_dict_disc)})
                
                #Shares for extinguishers
                df4_rac_data_scen_shares_ext = filter_dataframe(df4_rac_data, 'three_columns_scenarios', \
                      scenario=this_scen, scenario_2='Shares', scenario_3='Extintores', \
                      column='Scenario', column_2='Parameter', column_3='Subsector')
                types_shares_ext, types_projection_shares_ext, types_by_vals_shares_ext = process_types_and_df(df4_rac_data_scen_shares_ext, 'only_types', time_vector, column='Refrigerante')

                # Projection for shares for extinguishers
                rac_shares_ext_out_dict = generate_projection_dict(
                    types_shares_ext, types_projection_shares_ext, types_by_vals_shares_ext, 
                    df4_rac_data_scen_shares_ext, time_vector, 'first')                      
                        
                # Calculate total refrigerant (ton):
                ext_dem_by_type_dict = calculate_values(types_shares_ext, rac_shares_ext_out_dict, None, total_dem_rac_list, calculation_type='demand')
                                    
                #Estimating emisions (kton de CO2 eq)
                ext_emi_out_dict = calculate_values(types_shares_ext, rac_GWP_out_dict, ext_dem_by_type_dict, total_dem_rac_list, total_lf_list, 'emissions')    
                                    
                #Estimating prices (MUSD)
                ext_prices_out_dict = calculate_values(types_shares_ext, rac_price_out_dict, ext_dem_by_type_dict, total_dem_rac_list, calculation_type='prices')
                
                #Estimating discounted values 
                ext_prices_out_dict_disc = {}
                for key, values in ext_prices_out_dict.items():
                    discounted_prices = [value / ((1 + r_rate/100) ** t) for t, value in enumerate(values, 0)]
                    ext_prices_out_dict_disc[key] = discounted_prices
                
                #Storing costs and emissions for AC subsector 
                dict_local_country[this_country].update({'Emisiones para subsector de extintores [kt CO2 eq]': deepcopy(ext_emi_out_dict)})
                dict_local_country[this_country].update({'Costos para subsector de extintores [MUSD]': deepcopy(ext_prices_out_dict)})
                dict_local_country[this_country].update({'Costos para subsector de extintores [MUSD] (disc)': deepcopy(ext_prices_out_dict_disc)})
                print('RAC costs and emissions have been computed!')               
            
            """
            INSTRUCTIONS:
            1) Perform the transport demand calculations
            2) Check the demand component: rewrite the transport energy demand projection
            3) Store the demand and print the energy demand difference
            """
            #print('Rewrite the demand component here')
            types_all = []  # this is defined here to avoid an undefined variable

            # Define the dictionary that stores emissions here to keep
            # emissions from transport technologies.
            # 'emission factors':
            this_df4_ef = filter_dataframe(df4_ef, 'two_columns_scenarios', \
                                scenario='Standard', scenario_2='Emission factor', \
                                column='Type', column_2='Parameter')
            this_df4_ef_fuels = this_df4_ef['Fuel'].tolist()

            emissions_fuels_list, emissions_fuels_dict = calculate_emissions(
                this_df4_ef, this_df4_ef_fuels, time_vector)

            # Include the emission factor for black carbon:
            this_df4_ef_2 = filter_dataframe(df4_ef, 'scenario_simple', scenario='Emission factor_black carbon', column='Parameter')
            this_df4_ef_2_fuels = this_df4_ef_2['Fuel'].tolist()

            emissions_2_fuels_list, emissions_2_fuels_dict = calculate_emissions(
                this_df4_ef_2, this_df4_ef_2_fuels, time_vector)   

            emissions_demand = {}  # crucial output
            emissions_demand_black_carbon = {}
            if overwrite_transport_model:
                # We must edit the "dict_energy_demand" content with the transport modeling
                dict_energy_demand_trn = deepcopy(dict_energy_demand['Transport'])
                transport_fuel_sets = \
                    df2_trans_sets_eq['Transport Fuel'].tolist()
                transport_scenario_sets = \
                    df2_trans_sets_eq['Energy Fuel'].tolist()
                dict_eq_transport_fuels = {}
                for te in range(len(transport_fuel_sets)):
                    t_fuel_set = transport_fuel_sets[te]
                    t_scen_set = transport_scenario_sets[te]
                    dict_eq_transport_fuels.update({t_fuel_set:t_scen_set})
                        
                # NOTE: we must edit the projection from "dict_energy_demand" first;
                # once that is complete, we must edit the "dict_energy_demand_by_fuel" accordingly.
                
                # Now we must follow standard transport equations and estimations:
                # TM 1) Estimate the demand projection for the country
                """
                Demand_Passenger = sum_i (km_passenger_i * fleet_passenger_i * load_factor_i)
                """
                # Store load factor and kilometers:
                dict_lf, dict_km = {}, {}
                #
                # 1.a) estimate the demand in the base year
                
                # Function #24
                def filter_transport_types(transport_list, raw_list, transport_type):
                    """
                    Filter transport types based on their classification (Passenger or Freight).
                    
                    Args:
                    - transport_list (list): List of transport types.
                    - raw_list (list): List of classifications corresponding to each transport type in transport_list.
                    - transport_type (str): The classification type to filter by (e.g., 'Passenger' or 'Freight').
                    
                    Returns:
                    - (list): A filtered list of transport types that match the specified transport_type classification.
                    """
                    return [transport_list[n] for n, v in enumerate(raw_list) if v == transport_type]
                
                # Function #25
                def get_data(df, trn_type, country, parameter, year_column):
                    """
                    Extract specific data from a DataFrame based on given conditions.
                    
                    Args:
                    - df (DataFrame): The DataFrame containing transportation data.
                    - trn_type (str): The type of transportation.
                    - country (str): Target country for data extraction.
                    - parameter (str): The parameter to filter data by (e.g., 'Residual fleet', 'Distance', 'Load Factor').
                    - year_column (str): The column name representing years.
                    
                    Returns:
                    - (float): Sum of the data values from the DataFrame that meet the specified conditions.
                    """

                    mask = (df['Type'] == trn_type) & \
                           (df['Application_Countries'].isin(['All', country])) & \
                           (df['Parameter'] == parameter)
                    return df.loc[mask][year_column].sum()
                
                # Function #26
                def update_dictionaries(transport_set, df, country, year_column, dict_lf, dict_km):
                    """
                    Update dictionaries with demand, fleet, and share values for different transport types.
                    
                    Args:
                    - transport_set (list): A set of transport types.
                    - df (DataFrame): The DataFrame containing transportation data.
                    - country (str): Target country for data extraction.
                    - year_column (str): The column name representing years.
                    - dict_lf (dict): Dictionary to update with load factor values.
                    - dict_km (dict): Dictionary to update with distance values.
                    
                    Returns:
                    - fleet_by (dict): Dictionary with residual fleet values by transport type.
                    - dem_by (dict): Dictionary with demand values by transport type.
                    - dem_sh (dict): Dictionary with demand share percentages by transport type.
                    - sum_dem_by (float): Sum of all demand values.
                    """
                    fleet_by = {}
                    dem_by = {}
                    sum_dem_by = 0
                    for trn_type in transport_set:
                        fby = get_data(df, trn_type, country, 'Residual fleet', year_column)
                        km = get_data(df, trn_type, country, 'Distance', year_column)
                        lf = get_data(df, trn_type, country, 'Load Factor', year_column)
                        
                        dict_lf[trn_type] = deepcopy(lf)
                        dict_km[trn_type] = deepcopy(km)
                        
                        fleet_by[trn_type] = fby
                        dem_by_value = fby * km * lf / 1e9
                        dem_by[trn_type] = dem_by_value
                        sum_dem_by += dem_by_value
                        
                    dem_sh = {trn_type: 100 * dem_by_value / sum_dem_by for trn_type, dem_by_value in dem_by.items()}
                    
                    return fleet_by, dem_by, dem_sh, sum_dem_by
                
                # Select the scenario:
                mask_select_trn_scen = (df3_tpt_data['Scenario'] == this_scen) | (df3_tpt_data['Scenario'] == 'ALL')
                df_trn_data = df3_tpt_data.loc[mask_select_trn_scen]
                df_trn_data.reset_index(drop=True, inplace=True)
                
                # For Passenger trains
                list_pass_trn = filter_transport_types(list_trn_type, list_trn_lvl1_u_raw, 'Passenger')
                set_pass_trn_fleet_by, set_pass_trn_dem_by, set_pass_trn_dem_sh, sum_pass_trn_dem_by = update_dictionaries(list_pass_trn, df_trn_data, this_country, per_first_yr, dict_lf, dict_km)
                
                # For Freight trains
                list_fre_trn = filter_transport_types(list_trn_type, list_trn_lvl1_u_raw, 'Freight')
                set_fre_trn_fleet_by, set_fre_trn_dem_by, set_fre_trn_dem_sh, sum_fre_trn_dem_by = update_dictionaries(list_fre_trn, df_trn_data, this_country, per_first_yr, dict_lf, dict_km)

    
                # 1.b) estimate the demand growth
                # for this we need to extract a couple of variables from
                # the "df_trn_data":

                # Function #27
                def get_dem_model_projtype(transport_type, country, param, projection_type, df):
                    """
                    Extract the projection type and corresponding mask for a given transport type and parameter.
                    
                    Args:
                    - transport_type (str): The specific transport type (e.g., road, rail).
                    - country (str): The country for which the data is being extracted.
                    - param (str): The specific parameter to be considered (e.g., fuel efficiency, emissions).
                    - projection_type (str): The type of projection to be applied (e.g., linear, exponential).
                    - df (DataFrame): The DataFrame containing the relevant data.
                    
                    Returns:
                    - projtype (str): The determined projection type applicable to the given transport type and parameter.
                    - mask (pandas.core.series.Series): The mask or criteria derived for filtering or processing the data in the DataFrame according to the projection type.
                    """
                    projtype, mask = fun_dem_model_projtype(transport_type, country, param, projection_type, df)
                    return projtype, mask
                
                # Function #28
                def project_demand(time_vector, projtype, mask, df, sum_demand_by, gdp_growth_vals, gdp_pc_growth_vals, pop_growth_vals, ela_list):
                    """
                    Project transport demand based on the given projection type and growth parameters.
                    
                    Args:
                    - time_vector (list): List of years for which the projection is to be calculated.
                    - projtype (str): The type of projection method used (e.g., 'endogenous_gdp', 'endogenous_gdp_pc', 'endogenous_pop').
                    - mask (pandas.core.series.Series): Mask or filter criteria applied to the data.
                    - df (DataFrame): The DataFrame containing the relevant data.
                    - sum_demand_by (float): Initial sum of demand to be used as a base for projection.
                    - gdp_growth_vals (list): List of annual GDP growth values.
                    - gdp_pc_growth_vals (list): List of annual GDP per capita growth values.
                    - pop_growth_vals (list): List of annual population growth values.
                    - ela_list (list): List of elasticity values corresponding to the years in the time vector.
                    
                    Returns:
                    - trn_dem (list): List of projected transport demand values for each year in the time vector.
                    """
                    trn_dem = [0 for y in range(len(time_vector))]
                    for y in range(len(time_vector)):
                        if y == 0:
                            trn_dem[y] = sum_demand_by
                        else:
                            gdp_gr = gdp_growth_vals[y] / 100
                            gdp_pc_gr = gdp_pc_growth_vals[y] / 100
                            pop_gr = pop_growth_vals[y] / 100
                
                            if projtype == 'endogenous_gdp':
                                trn_gr = 1 + (gdp_gr * ela_list[y])
                            elif projtype == 'endogenous_gdp_pc':
                                trn_gr = 1 + (gdp_pc_gr * ela_list[y])
                            elif projtype == 'endogenous_pop':
                                trn_gr = 1 + (pop_gr * ela_list[y])
                
                            if 'endogenous' in projtype:
                                trn_dem[y] = trn_gr * trn_dem[y-1]
                                
                    return trn_dem
                
                # Main execution
                # Elasticities:
                projtype_ela_pas, mask_ela_pas = get_dem_model_projtype('Passenger', this_country, 'Elasticity', 'projection', df_trn_data)
                list_ela_pas = fun_dem_proj(time_vector, projtype_ela_pas, mask_ela_pas, df_trn_data)
                
                projtype_ela_fre, mask_ela_fre = get_dem_model_projtype('Freight', this_country, 'Elasticity', 'projection', df_trn_data)
                list_ela_fre = fun_dem_proj(time_vector, projtype_ela_fre, mask_ela_fre, df_trn_data)
                
                projtype_ela_oth, mask_ela_oth = get_dem_model_projtype('Other', this_country, 'Elasticity', 'projection', df_trn_data)
                list_ela_oth = fun_dem_proj(time_vector, projtype_ela_oth, mask_ela_oth, df_trn_data)
                
                # Demands:
                projtype_dem_pas, mask_dem_pas = get_dem_model_projtype('Passenger', this_country, 'Demand', 'projection', df_trn_data)
                if 'endogenous' not in projtype_dem_pas:
                    pass_trn_dem = fun_dem_proj(time_vector, projtype_dem_pas, mask_dem_pas, df_trn_data)
                else:
                    pass_trn_dem = project_demand(time_vector, projtype_dem_pas, mask_dem_pas, df_trn_data, sum_pass_trn_dem_by, this_gdp_growth_vals, this_gdp_pc_growth_vals, this_pop_growth_vals, list_ela_pas)
                
                projtype_dem_fre, mask_dem_fre = get_dem_model_projtype('Freight', this_country, 'Demand', 'projection', df_trn_data)
                if 'endogenous' not in projtype_dem_fre:
                    fre_trn_dem = fun_dem_proj(time_vector, projtype_dem_fre, mask_dem_fre, df_trn_data)
                else:
                    fre_trn_dem = project_demand(time_vector, projtype_dem_fre, mask_dem_fre, df_trn_data, sum_fre_trn_dem_by, this_gdp_growth_vals, this_gdp_pc_growth_vals, this_pop_growth_vals, list_ela_fre)
                
                projtype_dem_oth, mask_dem_oth = get_dem_model_projtype('Other', this_country, 'Demand', 'projection', df_trn_data)
                # Note: "Other" [transport demands] is a category currently unused
                                    
                # 1.c) apply the mode shift and non-motorized parameters:

                # Function #29
                def extract_transport_types(transport_level, type_name):
                    """
                    Extract specific transport types based on the type name.
                    
                    Args:
                    - transport_level (list): The list where each element represents the level or category of transport.
                    - type_name (str): The specific type name to filter the transport levels.
                    
                    Returns:
                    - (list): A list of transport types that match the specified type_name.
                    """
                    return [
                        list_trn_type[n]
                        for n, value in enumerate(transport_level)
                        if value == type_name
                    ]

                # Function #30
                def calculate_demand_share(types, demand_share):
                    """
                    Calculate the demand share for the given types.
                    
                    Args:
                    - types (list): The list of transport types for which to calculate the demand share.
                    - demand_share (dict): Dictionary where keys are transport types and values are their respective demand shares.
                    
                    Returns:
                    - (float): The sum of demand shares for the specified transport types.
                    """
                    return sum(demand_share[t] for t in types)

                # Function #31
                def adjust_demand_share(types, demand_share, total_share):
                    """
                    Adjust the demand share for each type based on the total share and return the result as a dictionary.
                    
                    Args:
                    - types (list): The list of transport types for which to adjust the demand share.
                    - demand_share (dict): Dictionary where keys are transport types and values are their respective demand shares before adjustment.
                    - total_share (float): The total share against which to adjust the individual shares.
                    
                    Returns:
                    - adjusted_share (dict): A dictionary with transport types as keys and their adjusted demand shares as values.
                    """
                    adjusted_share = {}
                    for t in types:
                        sh_k = demand_share[t]
                        sh_k_adj = 100 * sh_k / total_share
                        adjusted_share[t] = sh_k_adj
                    return adjusted_share
                
                # Function #32
                def extract_data_from_dataframe(condition, time_vector, dataframe):
                    """
                    Extract data from a DataFrame based on a given condition and for a specific country.
                    
                    Args:
                    - condition (pandas.core.series.Series): The condition based on which the data is to be extracted from the DataFrame.
                    - time_vector (list): List of years for which the data is to be extracted.
                    - dataframe (DataFrame): The DataFrame from which the data is to be extracted.
                    
                    Returns:
                    - (list): A list of data values extracted from the DataFrame for each year in the time_vector, based on the specified condition and country.
                    """

                    mask = dataframe['Application_Countries'].isin(['All', this_country]) & condition
                    return [dataframe.loc[mask][year].iloc[0] for year in time_vector]

                # Extract transport types
                set_pass_trn_priv = extract_transport_types(list_trn_lvl2_u_raw, 'Private')
                set_pass_trn_pub = extract_transport_types(list_trn_lvl2_u_raw, 'Public')
                
                # Calculate demand shares
                pass_trn_dem_sh_private = calculate_demand_share(set_pass_trn_priv, set_pass_trn_dem_sh)
                pass_trn_dem_sh_public = calculate_demand_share(set_pass_trn_pub, set_pass_trn_dem_sh)
                
                # Adjust demand shares
                pass_trn_dem_sh_private_k = adjust_demand_share(set_pass_trn_priv, set_pass_trn_dem_sh, pass_trn_dem_sh_private)
                pass_trn_dem_sh_public_k = adjust_demand_share(set_pass_trn_pub, set_pass_trn_dem_sh, pass_trn_dem_sh_public)

                # Extract data from dataframe based on conditions
                list_mode_shift = extract_data_from_dataframe((df_trn_data['Parameter'] == 'Mode shift'), time_vector, df_trn_data)
                list_non_motorized = extract_data_from_dataframe((df_trn_data['Parameter'] == 'Non-motorized transport'), time_vector, df_trn_data)

                # Function #33
                def calculate_gpkm(this_gpkm, types, adj_share, y, gpkm_k, lp):
                    """
                    Calculate gpkm (passenger kilometers) for a specific transport type (private or public) and adjust it by type. This function updates the gpkm values for each type and maintains a list of total gpkm values.
                    
                    Parameters:
                    - this_gpkm (type): The gpkm value for the current year before adjustment. The type will be specified later.
                    - types (list): List of transport types for which gpkm values are to be calculated and adjusted.
                    - adj_share (dict): Dictionary of adjusted shares for each transport type. The keys are transport types and the values are their corresponding adjusted shares.
                    - y (int): The current year index, used to access the correct year in time-based lists or arrays.
                    - gpkm_k (dict): Dictionary with lists of gpkm values for each transport type. This dictionary is updated within the function.
                    - lp (list): List of total gpkm values for the transport type. This list is updated within the function.
                    """              
                    for t in types:
                        sh_k_adj = adj_share[t]
                        this_gpkm_k = this_gpkm * sh_k_adj / 100
                        gpkm_k[t].append(this_gpkm_k)
                    lp.append(this_gpkm_k)             

                # Function #34
                def calculate_gpkm_for_types(time_vector, demand, priv_adj_share, pub_adj_share, mode_shift, non_motorized, priv_types, pub_types):
                    """
                    Calculate gpkm (passenger or freight kilometers) for various transport types across the time vector, accounting for mode shifts and non-motorized transport.
                    
                    Args:
                    - time_vector (list): List of years for which the calculations are performed.
                    - demand (list): List of demand values for each year.
                    - priv_adj_share (dict): Dictionary of adjusted share values for private transport types.
                    - pub_adj_share (dict): Dictionary of adjusted share values for public transport types.
                    - mode_shift (list): List of mode shift values for each year.
                    - non_motorized (list): List of non-motorized transport values for each year.
                    - priv_types (list): List of private transport types.
                    - pub_types (list): List of public transport types.
                    
                    Returns:
                    - gpkm_pri_k (dict): Dictionary with computed gpkm values for private types, keyed by transport type.
                    - gpkm_pub_k (dict): Dictionary with computed gpkm values for public types, keyed by transport type.
                    - gpkm_nonmot (list): List of computed gpkm values for non-motorized transport for each year.
                    - lpriv (list): List of total gpkm values for private types for each year.
                    - lpub (list): List of total gpkm values for public types for each year.
                    """
                    gpkm_pri_k = {key: [] for key in priv_types}
                    gpkm_pub_k = {key: [] for key in pub_types}
                    gpkm_nonmot = []
                    lpub =[]
                    lpriv = []
                    
                    for y in range(len(time_vector)):
                        # For private types
                        this_gpkm_priv = demand[y] * (pass_trn_dem_sh_private - mode_shift[y] - non_motorized[y]) / 100
                        calculate_gpkm(this_gpkm_priv, priv_types, priv_adj_share, y, gpkm_pri_k, lpriv)

                        # For public types
                        this_gpkm_pub = demand[y] * (pass_trn_dem_sh_public + mode_shift[y]) / 100
                        calculate_gpkm(this_gpkm_pub, pub_types, pub_adj_share, y, gpkm_pub_k, lpub)
                
                        # For non-motorized types
                        this_gpkm_nonmot = demand[y] * (non_motorized[-1])
                        gpkm_nonmot.append(this_gpkm_nonmot)
                
                    return gpkm_pri_k, gpkm_pub_k, gpkm_nonmot, lpriv, lpub
                
                # Use the function to calculate gpkm for various transport types
                gpkm_pri_k, gpkm_pub_k, gpkm_nonmot, lpriv, lpub = calculate_gpkm_for_types(
                    time_vector,
                    pass_trn_dem,
                    pass_trn_dem_sh_private_k,
                    pass_trn_dem_sh_public_k,
                    list_mode_shift,
                    list_non_motorized,
                    set_pass_trn_priv,
                    set_pass_trn_pub
                )
    
                # 1.d) apply the logistics parameters:
                # Initialize gtkm_freight_k
                # Function #35
                def initialize_dict_of_lists(list_actual):
                    """
                    Initialize a dictionary with empty lists for each item in the provided list.
                    
                    Args:
                    - list_actual (list): The list containing the items for which the dictionary is initialized.
                    
                    Returns:
                    - (dict): A dictionary where each item from list_actual is a key and the corresponding value is an empty list.
                    """
                    return {data_actual: [] for data_actual in list_actual}
                                
                
                # Extract logistics data
                # Function #36
                def extract_logistics_data(df_trn_data, time_vector, this_country):
                    """
                    Extract logistics data based on given conditions, filtering by country and the parameter 'Logistics'.
                    
                    Args:
                    - df_trn_data (DataFrame): DataFrame with transportation data.
                    - time_vector (list): List of years for which the data is to be extracted.
                    - this_country (str): Target country for data extraction.
                    
                    Returns:
                    - (list): List of logistics values for each year in time_vector.
                    """
                    mask_logistics = (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                     (df_trn_data['Parameter'] == 'Logistics')
                
                    return [df_trn_data.loc[mask_logistics][year].iloc[0] for year in time_vector]

                # Compute freight values
                # Function #37
                def compute_freight_values(set_fre_trn, fre_trn_dem, set_fre_trn_dem_sh):
                    """
                    Compute freight values (gtkm) for each transportation type based on freight demand and share.
                    
                    Args:
                    - set_fre_trn (set): Set of freight transportation types.
                    - fre_trn_dem (list): List of total freight demands for each year.
                    - set_fre_trn_dem_sh (dict): Dictionary with freight demand shares for each transportation type.
                    
                    Returns:
                    - gtkm_freight_k (dict): Dictionary with computed gtkm freight values for each type and year, keyed by transportation type.
                    """
                    gtkm_freight_k = {key: [] for key in set_fre_trn}
                
                    for y, demand in enumerate(fre_trn_dem):
                        for freight_type in set_fre_trn:
                            this_fre_sh_k = set_fre_trn_dem_sh[freight_type]
                            this_fre_k = demand * this_fre_sh_k / 100
                            gtkm_freight_k[freight_type].append(this_fre_k)
                
                    return gtkm_freight_k

                # Using the functions:
                gtkm_freight_k = {key: [] for key in list_fre_trn}
                list_logistics = extract_logistics_data(df_trn_data, time_vector, this_country)
                gtkm_freight_k = compute_freight_values(
                    list_fre_trn, fre_trn_dem, set_fre_trn_dem_sh)

                # TM 2) Estimate the required energy for transport
                """
                Paso 1: obtener el % de flota por fuel de cada carrocería
                """
                # A dictionary with the residual fleet will come in handy:
                dict_resi_cap_trn = {}
                
                # Continue distributing the fleet:
                types_pass = list_pass_trn
                types_fre = list_fre_trn
                fuels = transport_fuel_sets
                fuels_nonelectric = [i for i in transport_fuel_sets if
                					i not in ['ELECTRICIDAD', 'HIDROGENO']]

                # Function #38
                def fetch_fleet_share_by_fuel(df, vehicle_type, country, fuel, year):
                    """
                    Fetch the fleet share by fuel type for a given vehicle type and year from the DataFrame.
                    
                    Args:
                    - df (DataFrame): DataFrame containing transportation data.
                    - vehicle_type (str): The specific vehicle type.
                    - country (str): Target country for data extraction.
                    - fuel (str): The type of fuel.
                    - year (int): The specific year for which data is extracted.
                    
                    Returns:
                    - fleet_by_type_fuel (float): The fleet share by the specified fuel type for the given vehicle type and year. Returns 0 if data is not found.
                    """
                    mask = (
                        (df['Type'] == vehicle_type) & \
                        (df['Application_Countries'].isin(['All', country])) & \
                        (df['Parameter'] == 'Residual fleet') & \
                        (df['Fuel'] == fuel)
                    )
                    try:
                        fleet_by_type_fuel = df.loc[mask][year].iloc[0]
                    except Exception:
                        fleet_by_type_fuel = 0
                
                    return fleet_by_type_fuel
                
                # Function #39
                def calculate_fleet_shares_by_vehicle_type(types, fuels, df, country, first_year, time_vector, set_trn_fleet_by):
                    """
                    Calculate the fleet shares by vehicle type for all fuels and all years in the time vector.
                    
                    Args:
                    - types (list): List of vehicle types.
                    - fuels (list): List of fuel types.
                    - df (DataFrame): DataFrame containing transportation data.
                    - country (str): Target country for data extraction.
                    - first_year (int): The initial year from the time vector.
                    - time_vector (list): List of years for which the calculations are performed.
                    - set_trn_fleet_by (dict): Dictionary containing fleet information by vehicle type.
                    
                    Returns:
                    - fleet_by_sh (dict): Dictionary containing fleet shares by vehicle type and fuel type.
                    - dict_resi_fleet (dict): Dictionary containing residual fleet data by vehicle type and fuel type.
                    """
                    dict_resi_fleet = {}
                    fleet_by_sh = {}
                
                    for vehicle_type in types:
                        
                        total_type = set_trn_fleet_by[vehicle_type]   
                
                        fleet_by_sh[vehicle_type] = {}
                        dict_resi_fleet[vehicle_type] = {fuel: [] for fuel in fuels}
                
                        for fuel in fuels:
                            ffsbyf = fetch_fleet_share_by_fuel(df, vehicle_type, country, fuel, first_year)
                            
                            if total_type > 0:
                                fleet_percentage = 100 * ffsbyf / total_type
                            else:
                                fleet_percentage = 0
                
                            fleet_by_sh[vehicle_type][fuel] = fleet_percentage
                
                            for year in time_vector:
                                a_fleet = fetch_fleet_share_by_fuel(df, vehicle_type, country, fuel, year)
                                dict_resi_fleet[vehicle_type][fuel].append(a_fleet)
                
                    return fleet_by_sh, dict_resi_fleet

                # Using the functions to replace the existing code
                set_pass_trn_fleet_by_sh, dict_resi_cap_trn_V_pass = calculate_fleet_shares_by_vehicle_type(
                    types_pass, fuels, df_trn_data, this_country, per_first_yr, time_vector, set_pass_trn_fleet_by
                )
                
                set_fre_trn_fleet_by_sh, dict_resi_cap_trn_V_fre = calculate_fleet_shares_by_vehicle_type(
                    types_fre, fuels, df_trn_data, this_country, per_first_yr, time_vector, set_fre_trn_fleet_by
                )

                # Merging the dictionaries for residual fleet capacities
                dict_resi_cap_trn = {**dict_resi_cap_trn_V_pass, **dict_resi_cap_trn_V_fre}

                """
                Paso 2: Proyectar la participación de cada fuel en cada carrocería usando el parámetro "Electrification"
                """
                dict_fuel_economy = {}
                dict_shares_fleet = {}
                types_all = types_pass + types_fre
        
                # Function #40
                def calculate_share_non_electric(set_trn_fleet_by_sh, t, fuels_nonelectric):
                    """
                    Calculate the share of non-electric fuel for a transport type.
                    
                    Args:
                    - set_trn_fleet_by_sh (dict): Dictionary of fleet shares by transport type.
                    - t (str): The specific transport type.
                    - fuels_nonelectric (list): List of non-electric fuel types.
                    
                    Returns:
                    - sh_non_electric_k (dict): Dictionary of shares for non-electric fuels, adjusted by the total non-electric share.
                    """
                    sh_non_electric = 0
                    for fne in fuels_nonelectric:
                        sh_non_electric += set_trn_fleet_by_sh[t][fne]
                
                    sh_non_electric_k = {}
                    for fne in fuels_nonelectric:
                        this_sh_k_f = set_trn_fleet_by_sh[t][fne]
                        if sh_non_electric > 0:
                            this_sh_non_ele_k = 100 * this_sh_k_f / sh_non_electric
                        else:
                            this_sh_non_ele_k = 0
                        sh_non_electric_k[fne] = this_sh_non_ele_k
                    
                    return sh_non_electric_k
                
                # Function #41
                def get_parameter_values(df, this_country, t, parameter, time_vector, af):
                    """
                    Get values for a specific parameter over the time vector from the DataFrame, optionally filtered by fuel type.
                    
                    Args:
                    - df (DataFrame): DataFrame containing transportation data.
                    - this_country (str): Target country for data extraction.
                    - t (str): The specific transport type.
                    - parameter (str): The parameter for which data is extracted.
                    - time_vector (list): List of years for which the data is extracted.
                    - af (str/int): Optional argument for the fuel type, if required for filtering.
                    
                    Returns:
                    - values (list): List of parameter values for each year in the time vector.
                    """
                    if af==0:
                        mask = (df['Application_Countries'].isin(['All', this_country])) & \
                               (df['Parameter'] == parameter) & \
                               (df['Type'] == t)
                    else:
                        mask = (df['Application_Countries'].isin(['All', this_country])) & \
                               (df['Parameter'] == parameter) & \
                               (df['Type'] == t) & \
                               (df['Fuel'] == af)
                    values = []
                    for y in time_vector:
                        value = df.loc[mask][y].iloc[0]
                        if math.isnan(value):
                            value = 0
                        values.append(value)
                    
                    return values
                
                # Function #42
                def update_fuel_economy_and_fleet(df, fuels, fuels_nonelectric, sh_non_electric_k, list_non_electric, this_country, t, time_vector):
                    """
                    Update fuel economy and fleet information for each fuel type based on non-electric shares and other factors.
                    
                    Args:
                    - df (DataFrame): DataFrame containing transportation data.
                    - fuels (list): List of all fuel types.
                    - fuels_nonelectric (list): List of non-electric fuel types.
                    - sh_non_electric_k (dict): Dictionary of shares for non-electric fuels.
                    - list_non_electric (list): List of non-electric fleet data.
                    - this_country (str): Target country for data extraction.
                    - t (str): The specific transport type.
                    - time_vector (list): List of years for which the calculations are performed.
                    
                    Returns:
                    - list_fe_k (dict): Dictionary of fuel economy values by fuel type.
                    - list_nonele_fleet_k (dict): Dictionary of non-electric fleet data by fuel type.
                    """
                    list_fe_k = {}
                    list_nonele_fleet_k = {}
                    for af in fuels:
                        list_fe_k[af] = []
                        if af in fuels_nonelectric:
                            list_nonele_fleet_k[af] = []
                
                    for y in range(len(time_vector)):
                        for af in fuels_nonelectric:
                            this_sh_ne_k = sh_non_electric_k[af]
                            this_fleet_ne_k = this_sh_ne_k * list_non_electric[y] / 100
                            list_nonele_fleet_k[af].append(this_fleet_ne_k)
                
                        for af in fuels:
                            value = get_parameter_values(df, this_country, t, 'Fuel economy', [time_vector[y]], af)[0]
                            list_fe_k[af].append(value)
                    
                    return list_fe_k, list_nonele_fleet_k
                
                # The main loop where modularized functions will be used
                dict_fuel_economy = {}
                dict_shares_fleet = {}
                
                for t in types_all:
                    # Determine which fleet share set to use
                    set_trn_fleet_by_sh = set_pass_trn_fleet_by_sh if t in types_pass else set_fre_trn_fleet_by_sh
                    
                    # Calculate the share of non-electric fuel
                    sh_non_electric_k = calculate_share_non_electric(set_trn_fleet_by_sh, t, fuels_nonelectric)
                    
                    # Get electrification, hydrogen, and non-electric values
                    list_electrification = get_parameter_values(df_trn_data, this_country, t, 'Electrification', time_vector, 0)
                    list_hydrogen = get_parameter_values(df_trn_data, this_country, t, 'Hydrogen Penetration', time_vector, 0)
                    list_non_electric = [100 - ele - h2 for ele, h2 in zip(list_electrification, list_hydrogen)]
                
                    # Update fuel economy and fleet information
                    list_fe_k, list_nonele_fleet_k = update_fuel_economy_and_fleet(df_trn_data, fuels, fuels_nonelectric, sh_non_electric_k, list_non_electric, this_country, t, time_vector)
                
                    # ... rest of the code to handle electrification and hydrogen adjustments ...
                    
                    # Store the data for this "type"
                    dict_fuel_economy[t] = {}
                    dict_shares_fleet[t] = {}
                    for af in fuels:
                        if af in fuels_nonelectric:
                            dict_shares_fleet[t][af] = deepcopy(list_nonele_fleet_k[af])
                        elif af == 'ELECTRICIDAD':
                            dict_shares_fleet[t][af] = deepcopy(list_electrification)
                        elif af == 'HIDROGENO':
                            dict_shares_fleet[t][af] = deepcopy(list_hydrogen)
                        else:
                            print('Undefined fuel set (1). Please check.')
                            sys.exit()
                        dict_fuel_economy[t][af] = deepcopy(list_fe_k[af])
                
                """
                Paso 3: calcular la energía requerida para el sector transporte
                """
                dict_trn_pj = {}
 
                # Function #43
                def initialize_nested_dict(types, fuels, time_vector_length):
                    """
                    Initialize a nested dictionary with zeroes for each combination of type and fuel, for a specified length of the time vector.
                    
                    Args:
                    - types (list): List of transport types or categories.
                    - fuels (list): List of fuel types.
                    - time_vector_length (int): Length of the time vector, indicating the number of years or time points.
                    
                    Returns:
                    - (dict): A nested dictionary where each transport type is a key, and the value is another dictionary with each fuel type as a key and a list of zeroes (of length time_vector_length) as values.
                    """
                    zero_list = [0 for _ in range(time_vector_length)]
                    return {t: {f: deepcopy(zero_list) for f in fuels} for t in types}                
                
                # Function #44
                def aggregate_gpkm(types, gpkm_pri_k, gpkm_pub_k, gtkm_freight_k):
                    """
                    Aggregate gpkm (passenger kilometers) and gtkm (goods kilometers) for all transport types.
                    
                    Args:
                    - types (list): List of transport types or categories.
                    - gpkm_pri_k (dict): Dictionary with gpkm values for private types, keyed by transport type.
                    - gpkm_pub_k (dict): Dictionary with gpkm values for public types, keyed by transport type.
                    - gtkm_freight_k (dict): Dictionary with gtkm values for freight types, keyed by transport type.
                    
                    Returns:
                    - dict_gpkm_gtkm (dict): Dictionary with aggregated gpkm and gtkm values, keyed by transport type.
                    - this_gpkm_add (list): List of gpkm or gtkm values added for the last transport type processed.
                    - t (str): The last transport type processed.
                    """
                    dict_gpkm_gtkm = {}
                    for t in types:
                        if t in list(gpkm_pri_k.keys()):
                            this_gpkm_add = gpkm_pri_k[t]
                        if t in list(gpkm_pub_k.keys()):
                            this_gpkm_add = gpkm_pub_k[t]
                        if t in list(gtkm_freight_k.keys()):
                            this_gpkm_add = gtkm_freight_k[t]
                        # Ensure that a default list is provided if none of the above keys match
                        dict_gpkm_gtkm.update({t:this_gpkm_add})
                    return dict_gpkm_gtkm, this_gpkm_add, t
                
                # Function #45
                def extract_projected_fleet(df, scenario, fuels, types, country, time_vector):
                    """
                    Extract the projected fleet data for all fuels and vehicle types based on a given scenario.
                    
                    Args:
                    - df (DataFrame): DataFrame containing transportation data.
                    - scenario (str): The specific scenario name to filter the data.
                    - fuels (list): List of fuel types.
                    - types (list): List of vehicle types or categories.
                    - country (str): Target country for data extraction.
                    - time_vector (list): List of years for which the data is extracted.
                    
                    Returns:
                    - dict_proj_fleet (dict): Nested dictionary with projected fleet data, structured by vehicle type and fuel type.
                    """
                    dict_proj_fleet = initialize_nested_dict(types, fuels, len(time_vector))
                    for fuel in fuels:
                        for vehicle_type in types:
                            mask = (
                                (df['Scenario'] == scenario) &
                                (df['Fuel'] == fuel) &
                                (df['Type'] == vehicle_type) &
                                (df['Application_Countries'].isin(['All', country])) &
                                (df['Parameter'] == 'Projected fleet')
                            )
                            # Fetch the projection type and handle empty cases
                            projection_type = df.loc[mask]['projection'].iloc[0] if not df.loc[mask]['projection'].empty else 'ignore'
                            
                            # Fetch the data for each year
                            yearly_data = [df.loc[mask][year].iloc[0] if not df.loc[mask][year].empty else 0 for year in time_vector]
                            
                            # Decision logic
                            dict_proj_fleet = handle_projection(projection_type, dict_proj_fleet, fuel, vehicle_type, yearly_data)
                
                    return dict_proj_fleet
                
                # Function #46
                def handle_projection(projection_type, dict_proj_fleet, fuel, vehicle_type, yearly_data):
                    """
                    Decide how to handle the projected fleet data based on the projection type.
                    
                    Args:
                    - projection_type (str): The type of projection (e.g., 'ignore', 'user_defined').
                    - dict_proj_fleet (dict): Nested dictionary to be updated with projected fleet data.
                    - fuel (str): The specific fuel type.
                    - vehicle_type (str): The specific vehicle type or category.
                    - yearly_data (list): List of projected fleet data for each year.
                    
                    Returns:
                    - dict_proj_fleet (dict): Updated nested dictionary with decisions or data applied to the projected fleet, based on the projection type.
                    """
                    if projection_type == 'ignore':
                        dict_proj_fleet[vehicle_type][fuel] = {'indicate': 'ignore'}
                    elif projection_type == 'user_defined':
                        dict_proj_fleet[vehicle_type][fuel] = {'indicate': 'apply', 'vals': yearly_data}
                    else:
                        print('Undefined projection type specified for the Projected fleet.')
                        sys.exit()
                    return dict_proj_fleet
                
                # Replace the existing code with function calls
                dict_trn_pj = {key: [] for key in fuels}
                dict_gpkm_gtkm, this_gpkm_add, t = aggregate_gpkm(types_all, gpkm_pri_k, gpkm_pub_k, gtkm_freight_k)
                dict_fleet_k = initialize_nested_dict(types_all, fuels, len(time_vector))
                dict_fuel_con = initialize_nested_dict(types_all, fuels, len(time_vector))
                dict_conv_cons = initialize_nested_dict(types_all, fuels, len(time_vector))
                
                # Additional dictionaries based on the provided variables
                dict_gpkm_gtkm_k = initialize_nested_dict(types_all, fuels, len(time_vector))
                dict_trn_pj_2 = initialize_nested_dict(fuels, types_all, len(time_vector))

                dict_proj_fleet = extract_projected_fleet(df_trn_data, this_scen, fuels, types_all, this_country, time_vector)

                # For all fuels, find the energy consumption:    
                dict_diffs_f_rf = {}
                emissions_demand = {}  # crucial output              
                
                # Function #47
                def calculate_fleet(distribution, load_factor, distance):
                    """
                    Calculate the fleet size based on the distribution of vehicles, load factor, and distance covered.
                    
                    Args:
                    - distribution (float): The distribution value for the specific vehicle type and fuel.
                    - load_factor (float): The load factor for the specific vehicle type and fuel.
                    - distance (float): The distance covered by the specific vehicle type and fuel.
                    - vehicle_type (str): The type of vehicle.
                    - fuel (str): The type of fuel.
                    
                    Returns:
                    - (float): The calculated fleet size for the given vehicle type and fuel.
                    """
                    if distance == 0:
                        print('review division by zero')
                        sys.exit()
                    return 1e9 * (distribution / load_factor) / distance
                
                # Function #48
                def apply_projection_overwrites(y, vehicle_type, fuel, projected_fleet_data, fleet, gpkm_gtkm, fuel_con, dict_km, fuel_efficiency, dict_lf, dict_fleet_k, dict_fuel_con):
                    """
                    Overwrite the fleet projections and associated values if the user has provided custom projections.
                    
                    Args:
                    - y (int): The current year index.
                    - vehicle_type (str): The type of vehicle.
                    - fuel (str): The type of fuel.
                    - projected_fleet_data (dict): Nested dictionary with projected fleet data.
                    - fleet (float): The current fleet size for the specific vehicle type and fuel.
                    - gpkm_gtkm (dict): Dictionary containing gpkm or gtkm values for the vehicle type.
                    - fuel_con (float): The current fuel consumption for the specific vehicle type and fuel.
                    - dict_km (dict): Dictionary containing km values for each vehicle type.
                    - fuel_efficiency (float): The fuel efficiency for the specific vehicle type and fuel.
                    - dict_lf (dict): Dictionary containing load factor values for each vehicle type.
                    - dict_fleet_k (dict): Dictionary to store the fleet size for each vehicle type and fuel.
                    - dict_fuel_con (dict): Dictionary to store the fuel consumption for each vehicle type and fuel.
                    
                    Returns:
                    - float: The updated fuel consumption after applying the projection overwrites.
                    """
                    if projected_fleet_data[vehicle_type][fuel]['indicate'] == 'apply':
                        proj_fleet_y = projected_fleet_data[fuel][vehicle_type]['vals'][y]
                
                        # Overwriting fleet:                              
                        dict_fleet_k[vehicle_type][fuel][y] = deepcopy(proj_fleet_y)
                            
                        # Overwriting gpkm_gtkm (gotta sum):
                        delta_km = (proj_fleet_y - fleet) * dict_km[vehicle_type]
                        delta_gpkm_gtkm = delta_km * dict_lf[vehicle_type] / 1e9
                        gpkm_gtkm[vehicle_type][y] += deepcopy(delta_gpkm_gtkm)
                
                        # Overwriting fuel (gotta sum):
                        delta_fuel_con = delta_km * fuel_efficiency / 1e9  # PJ
                        dict_fuel_con[vehicle_type][fuel][y] += delta_fuel_con
                        fuel_con += delta_fuel_con
                
                    return fuel_con
                
                # Function #49
                def estimate_emissions(fuel_con, fuel, year_index, emissions_fuels_dict, dict_eq_transport_fuels, time_vector):
                    """
                    Estimate emissions based on fuel consumption, fuel type, and the index of the year.
                    
                    Args:
                    - fuel_con (float): The fuel consumption for the specific fuel and year.
                    - fuel (str): The type of fuel.
                    - year_index (int): The index representing the year in the time vector.
                    - emissions_fuels_dict (dict): Dictionary containing emissions factors for different fuels.
                    - dict_eq_transport_fuels (dict): Dictionary for converting model fuel types to emission factor fuel types.
                    - time_vector (list): List of years for which the calculations are performed.
                    
                    Returns:
                    - float: The estimated emissions for the specific fuel and year.
                    - str: The fuel type in the emissions_fuels_dict corresponding to the input fuel.
                    """
                    fuel_energy_model = dict_eq_transport_fuels[fuel]
                    # Ensure that the fuel_energy_model is in the emissions_fuels_dict and the year_index is within range
                    if fuel_energy_model in emissions_fuels_dict and year_index < len(emissions_fuels_dict[fuel_energy_model]):
                        emis_fact = emissions_fuels_dict[fuel_energy_model][year_index]
                    else:
                        emis_fact = 0
                    return fuel_con * emis_fact, fuel_energy_model
    
                for this_f in fuels:
                    dict_diffs_f_rf.update({this_f: {}})
                    this_list = []
                    emis_transport_dict = {}
                    emis_transport_black_carbon_dict = {}
                    
                    for y in range(len(time_vector)):
                        this_fuel_con = 0
                        
                        for t in types_all:
                            if y == 0:
                                emis_transport_dict.update({t: []})
                                emis_transport_black_carbon_dict.update({t: []})
                                if t not in emissions_demand:
                                    emissions_demand[t] = {}
                                if t not in list(emissions_demand_black_carbon.keys()):
                                    emissions_demand_black_carbon.update({t:{}})
                
                            this_gpkm_gtkm = dict_gpkm_gtkm[t][y]
                            this_sh_fl = dict_shares_fleet[t][this_f][y] / 100
                            this_fe = dict_fuel_economy[t][this_f][y]
                            
                            # Modularized code to be inserted
                            add_fuel_con = this_gpkm_gtkm * this_sh_fl * this_fe / dict_lf[t]
                            # dict_fuel_con[t][this_f][y] = deepcopy(add_fuel_con)
                            
                            if add_fuel_con < 1e-4:
                                dict_fuel_con[t][this_f][y] = 0
                            else:
                                dict_fuel_con[t][this_f][y] = deepcopy(add_fuel_con)
                            
                            this_fuel_con += deepcopy(add_fuel_con)
                            dict_trn_pj_2[this_f][t][y] = deepcopy(add_fuel_con)
                            
                            
                
                            this_gpkm_gtkm_k = this_gpkm_gtkm * this_sh_fl
                            dict_gpkm_gtkm_k[t][this_f][y] = this_gpkm_gtkm_k
                
                            this_fleet_k = calculate_fleet(this_gpkm_gtkm_k, dict_lf[t], dict_km[t])
                            # if this_fleet_k < 10:
                            #     dict_fleet_k[t][this_f][y] = 0
                            # else:
                            #     dict_fleet_k[t][this_f][y] = this_fleet_k
                            dict_fleet_k[t][this_f][y] = this_fleet_k

                            if y == 0 and dict_resi_cap_trn[t][this_f][0] != 0:
                                dict_diffs_f_rf[this_f].update({t: this_fleet_k / dict_resi_cap_trn[t][this_f][0]})
                
                            this_fuel_con = apply_projection_overwrites(y, t, this_f, dict_proj_fleet, this_fleet_k, dict_gpkm_gtkm, this_fuel_con, dict_km, this_fe, dict_lf, dict_fleet_k, dict_fuel_con)
                            
                            year_index = time_vector.index(time_vector[y])  # Get the index of the current year in time_vector
                            emis_transport, fuel_energy_model = estimate_emissions(
                                dict_fuel_con[t][this_f][y], 
                                this_f, 
                                year_index, 
                                emissions_fuels_dict, 
                                dict_eq_transport_fuels, 
                                time_vector
                            )
                            emis_transport_dict[t].append(emis_transport)
                            
                            if fuel_energy_model in list(emissions_2_fuels_dict.keys()):
                                emis_fact_black_carbon = emissions_2_fuels_dict[fuel_energy_model][y]
                            else:
                                emis_fact_black_carbon = 0
                            
                            emis_transport_black_carbon = dict_fuel_con[t][this_f][y]*emis_fact_black_carbon
    
                            
                            emis_transport_black_carbon_dict[t].append(emis_transport_black_carbon)
    
                        this_list.append(this_fuel_con)

                    #
                    for t in types_all:
                        emissions_demand[t].update({
                            this_f:deepcopy(emis_transport_dict[t])})
                        emissions_demand_black_carbon[t].update({
                            this_f:deepcopy(emis_transport_black_carbon_dict[t])})
    
                    dict_trn_pj[this_f] = deepcopy(this_list)

                # *********************************************************
                # We can calculate the required new fleets to satisfy the demand:
                dict_new_fleet_k, dict_accum_new_fleet_k = {}, {}
    
                # We will take advantage to estimate the costs related to
                # fleet and energy; we can check the cost and tax params:
                cost_params = list(dict.fromkeys(d5_tpt['Parameter'].tolist()))
                # ['CapitalCost', 'FixedCost', 'VariableCost', 'OpLife']
                cost_units = list(dict.fromkeys(d5_tpt['Unit'].tolist()))
                # 
                tax_params = list(dict.fromkeys(d5_tax['Parameter'].tolist()))
                # ['Imports', 'IMESI_Venta', 'IVA_Venta', 'Patente',
                # 'IMESI_Combust', 'IVA_Gasoil', 'IVA_Elec', 'Impuesto_Carbono',
                # 'Otros_Gasoil', 'Rodaje']
    
                # Define the cost outputs:
                dict_capex_out = {}
                dict_fopex_out = {}
                dict_vopex_out = {}

                # Function #50
                def initialize_tax_output(tax_params, types_all, fuels):
                    """
                    Initialize the dictionary for tax outputs with the structure to accommodate various tax parameters, vehicle types, and fuel types.

                    Args:
                    - tax_params (list): List of different tax parameters.
                    - types_all (list): List of all vehicle types.
                    - fuels (list): List of all fuel types.

                    Returns:
                    - dict_tax_out (dict): Initialized nested dictionary for storing tax outputs.
                    """
                    dict_tax_out = {}
                    for atax in tax_params:
                        dict_tax_out[atax] = {t: {f: {} for f in fuels} for t in types_all}
                    return dict_tax_out
                
                #Function #51
                def calculate_new_accumulated_fleet(t, f, dict_fleet_k, dict_fuel_con, dict_resi_cap_trn, list_op_life, time_vector):
                    """
                    Calculate new and accumulated fleet lists based on the total fleet, residual capacity, operational life, and time vector.

                    Args:
                    - t (str): The specific transport type.
                    - f (str): The specific fuel type.
                    - dict_fleet_k (dict): Dictionary containing fleet data.
                    - dict_fuel_con (dict): Dictionary to store the fuel consumption for each vehicle type and fuel.
                    - dict_resi_cap_trn (dict): Dictionary containing residual capacity data.
                    - list_op_life (list): List of operational life values.
                    - time_vector (list): List of years for which the calculations are performed.

                    Returns:
                    - new_fleet_lst (list): List of new fleet values.
                    - accum_fleet_lst (list): List of accumulated fleet values.
                    - times_neg_new_fleet (int): Number of occurrences of negative new fleet values.
                    - times_neg_new_fleet_sto (list): List storing negative new fleet values.
                    - tot_fleet_lst (list): List of total fleet values.
                    - fuel_con_lst (list): List of fuel consumption values.
                    """
                    tot_fleet_lst = dict_fleet_k[t][f]
                    res_fleet_lst = dict_resi_cap_trn[t][f]
                    fuel_con_lst = dict_fuel_con[t][f]
                    accum_fleet_lst = [0 for y in range(len(time_vector))]
                    new_fleet_lst = [0 for y in range(len(time_vector))]
                    times_neg_new_fleet = 0
                    times_neg_new_fleet_sto = []
                
                    for y in range(len(time_vector)):
                        this_new_fleet = tot_fleet_lst[y] - res_fleet_lst[y] - (0 if y == 0 else accum_fleet_lst[y])
                        if this_new_fleet >= 10:
                            new_fleet_lst[y] = this_new_fleet
                            for y2 in range(y, y + int(list_op_life[y])):
                                if y2 < len(time_vector):
                                    accum_fleet_lst[y2] += this_new_fleet
                        else:
                            times_neg_new_fleet += 1
                            times_neg_new_fleet_sto.append(this_new_fleet)
                
                    return new_fleet_lst, accum_fleet_lst, times_neg_new_fleet, times_neg_new_fleet_sto, \
                        tot_fleet_lst, fuel_con_lst #, dict_fleet_new
                
                # Function #52
                def convert_variable_cost_unit(unit_var_cost, f, conv_cons):
                    """
                    Convert variable cost units to PJ based on the fuel type.

                    Args:
                    - unit_var_cost (str): The unit of the variable cost.
                    - f (str): The specific fuel type.
                    - conv_cons (float): Existing conversion constant.

                    Returns:
                    - conv_cons (float): Updated conversion constant after applying the unit conversion.
                    """
                    if unit_var_cost == 'USD/liter' and 'DIESEL' in f:
                        conv_cons = 38.6 * (1e-9)  # from liter to PJ
                    if unit_var_cost == 'USD/liter' and 'GASOLINA' in f:
                        conv_cons = 34.2 * (1e-9)  # from liter to PJ
                    if unit_var_cost == 'USD/kWh':
                        conv_cons = 3.6e-9  # from kWh to PJ
                    if unit_var_cost == 'USD/kg':
                        conv_cons = (3.6 * 33.33) * 1e-9  # from kg to PJ
                    #if unit_var_cost == 'USD/pkm':
                    #    print(conv_cons)
                    #    sys.exit()
                    return conv_cons
                
                # Function #53
                def unpack_taxes(atax, t, f, d5_tax, time_vector, apply_costs, dict_km):
                    """
                    Unpack tax parameters and calculate unit tax values based on the provided costs and depreciation.

                    Args:
                    - atax (str): The specific tax parameter.
                    - t (str): The specific transport type.
                    - f (str): The specific fuel type.
                    - d5_tax (dict): Dictionary containing tax data.
                    - time_vector (list): List of years for which the calculations are performed.
                    - apply_costs (dict): Dictionary containing costs applied for tax calculations.
                    - dict_km (dict): Dictionary containing km by tech.

                    Returns:
                    - list_atax (list): List of tax values.
                    - list_atax_unit (list): List of unit tax values.
                    - ref_atax (str): Reference for the tax parameter.
                    - list_atax_unit_without_cost (list): List of values for road tax.
                    - tech_km (float): km value of the tech use.
                    """
                    list_atax, ref_atax, mult_depr, tech_km = fun_unpack_taxes(atax, t, f, d5_tax, time_vector, dict_km)
                    list_atax_unit = []
                    list_atax_unit_without_cost = []
                    list_conv_cte_unit_tax = []
                
                    for y in range(len(time_vector)):
                        try:
                            if ref_atax == 'CapitalCost*':
                                ref_atax_call = 'CapitalCost'
                            else:
                                ref_atax_call = ref_atax
                            apply_costs_atax = apply_costs[ref_atax_call][y]
                        except Exception:
                             apply_costs_atax = 0
                        add_atax_unit = \
                            apply_costs_atax*list_atax[y]/100
                        list_atax_unit.append(add_atax_unit*mult_depr)
                        if atax == 'Rodaje':
                            list_atax_unit_without_cost.append(list_atax[y])
                            list_conv_cte_unit_tax.append(list_atax[y])
                        
                        else:
                            list_atax_unit_without_cost.append(0)
                            if apply_costs_atax !=0:
                                list_conv_cte_unit_tax.append(100/(mult_depr*apply_costs_atax)) 
                            elif apply_costs_atax ==0:                              
                                list_conv_cte_unit_tax.append(0)
                    
                    mult_depr = list_conv_cte_unit_tax
                
                    return list_atax, list_atax_unit, ref_atax, list_atax_unit_without_cost, tech_km, mult_depr
                
                # Function #54
                def calculate_tax_values(t, f, tax_params, d5_tax, time_vector, apply_costs, new_fleet_lst, tot_fleet_lst, fuel_con_lst, conv_cons, dict_tax_out, rel_tax_activity, dict_tax_unit_out,rel_tax_activity2,scenario,dict_km,dict_mult_depr):
                    """
                    Calculate and store tax values based on the provided tax parameters, vehicle type, fuel type, and costs.

                    Args:
                    - t (str): The specific transport type.
                    - f (str): The specific fuel type.
                    - tax_params (list): List of different tax parameters.
                    - d5_tax (dict): Dictionary containing tax data.
                    - time_vector (list): List of years for which the calculations are performed.
                    - apply_costs (dict): Dictionary containing costs applied for tax calculations.
                    - new_fleet_lst (list): List of new fleet values.
                    - tot_fleet_lst (list): List of total fleet values.
                    - fuel_con_lst (list): List of fuel consumption values.
                    - conv_cons (float): Conversion constant for cost calculations.
                    - dict_tax_out (dict): Initialized nested dictionary for storing tax outputs.
                    - rel_tax_activity (dict): Initialized nested dictionary for storing activity.
                    - dict_tax_unit_out (dict): Initialized nested dictionary for storing tx unit outputs.
                    - dict_km (dict): Dictionaru containing km by tech.

                    Returns:
                    - dict_tax_out (dict): Updated dictionary with calculated tax values.
                    - rel_tax_activity (dict): References to choose activity.
                    - dict_tax_unit_out (dict): Updated dictionary with calculated tax unit values.
                    """
                    

                    for atax in tax_params:
                        list_atax, list_atax_unit, ref_atax, list_atax_unit_without_cost, tech_km, mult_depr = unpack_taxes(atax, t, f, d5_tax, time_vector, apply_costs, dict_km)
                        add_atax_val_lst = []
                        
                
                        for y in range(len(time_vector)):
                            if ref_atax == 'CapitalCost':
                                add_atax_val = list_atax_unit[y]*new_fleet_lst[y]
                                rel_tax_activity[atax] = ref_atax
                                
                            elif ref_atax == 'CapitalCost*':
                                add_atax_val = list_atax_unit[y]*tot_fleet_lst[y]
                            elif ref_atax == 'KmCost':
                                add_atax_val = list_atax_unit_without_cost[y]*tot_fleet_lst[y]*tech_km

                                if atax == 'Rodaje':
                                        add_atax_val = 0.0
                                
                            else:  # Variable cost
                                add_atax_val = list_atax_unit[y]*fuel_con_lst[y]/conv_cons
                            add_atax_val_lst.append(add_atax_val)
                            
                            # if t == ('Automoviles' or 'SUV, Crossover y Rural') and time_vector[y] == 2050:
                            #     print(atax,t,f,list_atax_unit[y]*mult_depr[y],add_atax_val)
                            
                        if ref_atax != 'None':
                            rel_tax_activity.update({atax: ref_atax})
                        dict_tax_out[atax][t][f] = deepcopy(add_atax_val_lst)
                        dict_tax_unit_out[atax][t][f] = deepcopy(list_atax)
                        rel_tax_activity2[atax][t][f] = deepcopy(ref_atax)
                        dict_mult_depr[atax][t][f] = deepcopy(mult_depr)
                    return dict_tax_out, rel_tax_activity, dict_tax_unit_out, rel_tax_activity2, dict_mult_depr
                
                # Dictionary with all transport taxes
                dict_tax_out = initialize_tax_output(tax_params, types_all, fuels)
                
                # Dictionary with all transport unit taxes
                dict_tax_unit_out = initialize_tax_output(tax_params, types_all, fuels)
                rel_tax_activity2 = initialize_tax_output(tax_params, types_all, fuels)
                dict_mult_depr = initialize_tax_output(tax_params, types_all, fuels)
                
                # Dictionary with all transport activities use to calculate taxes
                activities_params = ['Fuel Consumption', 'Conversion Fuel Constant', 'Relation Tax-Activity']
                dict_activities_out = initialize_tax_output(activities_params, types_all, fuels)
                dict_activities_out.update({'Fuel Consumption': dict_fuel_con})
                rel_tax_activity = {atax:{} for atax in tax_params}

                conv_cons = 0
                for t in types_all:
                    dict_new_fleet_k[t] = {}
                    dict_capex_out[t] = {}
                    dict_fopex_out[t] = {}
                    dict_vopex_out[t] = {}
                    
                    for f in fuels:
                        # Assuming fun_unpack_costs and other necessary functions are defined
                        list_cap_cost, unit_cap_cost = fun_unpack_costs('CapitalCost', t, f, d5_tpt, time_vector)
                        list_fix_cost, unit_fix_cost = fun_unpack_costs('FixedCost', t, f, d5_tpt, time_vector)
                        list_var_cost, unit_var_cost = fun_unpack_costs('VariableCost', t, f, d5_tpt, time_vector)
                        list_op_life, unit_op_life = fun_unpack_costs('OpLife', t      , f, d5_tpt, time_vector)
                        apply_costs = {'CapitalCost': deepcopy(list_cap_cost), 'VariableCost': deepcopy(list_var_cost)}
                        
                        new_fleet_lst, accum_fleet_lst, times_neg_new_fleet, times_neg_new_fleet_sto, \
                            tot_fleet_lst, fuel_con_lst = calculate_new_accumulated_fleet(t, f, dict_fleet_k, \
                            dict_fuel_con, dict_resi_cap_trn, list_op_life, time_vector)
                        conv_cons = convert_variable_cost_unit(unit_var_cost, f, conv_cons)
                        
                        usd_capex_lst = []
                        usd_fopex_lst = []
                        usd_vopex_lst = []
                        for y in range(len(time_vector)):
                            usd_capex_lst.append(new_fleet_lst[y]*list_cap_cost[y])
                            usd_fopex_lst.append(tot_fleet_lst[y]*list_fix_cost[y]*dict_km[t]/1000)
                            usd_vopex_lst.append(fuel_con_lst[y]*list_var_cost[y]/conv_cons)
                        dict_new_fleet_k[t][f] = new_fleet_lst
                        dict_capex_out[t][f] = usd_capex_lst
                        dict_fopex_out[t][f] = usd_fopex_lst
                        dict_vopex_out[t][f] = usd_vopex_lst
                        dict_conv_cons[t][f] = conv_cons
                        # if t == 'Birodados' and f == 'ELECTRICIDAD':
                        #     print(this_scen,usd_capex_lst)

                        # print(scenario_list[s])
                        
                        # Assuming the necessary variables like new_fleet_lst, tot_fleet_lst, etc. are defined
                        
                        dict_tax_out, rel_tax_activity, dict_tax_unit_out, \
                        rel_tax_activity2, dict_mult_depr = calculate_tax_values(t, f, tax_params, d5_tax, time_vector, \
                                             apply_costs, new_fleet_lst, tot_fleet_lst, fuel_con_lst, \
                                             conv_cons, dict_tax_out, rel_tax_activity, \
                                             dict_tax_unit_out,rel_tax_activity2,scenario_list[s], \
                                             dict_km, dict_mult_depr)

                dict_activities_out.update({'Conversion Fuel Constant': dict_conv_cons})
                rel_tax_activity['IC'] = rel_tax_activity.pop('Impuesto_Carbono')
                dict_activities_out.update({'Relation Tax-Activity': rel_tax_activity})
                rel_tax_activity2['IC'] = rel_tax_activity2.pop('Impuesto_Carbono')
                dict_activities_out.update({'Relation Tax-Activity2': rel_tax_activity2})
                dict_mult_depr['IC'] = dict_mult_depr.pop('Impuesto_Carbono')
                # TM 3) Update the transport vector
                dict_eq_trn_fuels_rev = {}

                # Function #55
                def update_transport_vector(fuels, dict_eq_transport_fuels, dict_trn_pj, time_vector, \
                                            dict_energy_demand_trn,dict_eq_trn_fuels_rev):
                    """
                    Update the transport vector based on the fuel equivalence mapping, aggregating energy demand for hybrid vehicles.

                    Args:
                    - fuels (list): List of fuel types.
                    - dict_eq_transport_fuels (dict): Dictionary mapping each fuel type to its equivalent in the transport energy model.
                    - dict_trn_pj (dict): Dictionary containing transport energy demand in PJ for each fuel type.
                    - time_vector (list): List of years for which the calculations are performed.
                    - dict_energy_demand_trn (dict): Dictionary to store the updated energy demand for transport.
                    - dict_eq_trn_fuels_rev (dict): Dictionary to store the reverse mapping from transport energy model fuels to actual fuel types.

                    Returns:
                    - dict_eq_trn_fuels_rev (dict): Updated dictionary with reverse fuel equivalence mapping.
                    - dict_energy_demand_trn (dict): Updated dictionary with transport energy demand in PJ.
                    """             
                    for this_f in fuels:
                        this_equivalence = dict_eq_transport_fuels[this_f]
                        if 'HIBRIDO' not in this_f:
                            dict_energy_demand_trn[this_equivalence] = deepcopy(dict_trn_pj[this_f])
                            dict_eq_trn_fuels_rev.update({this_equivalence: this_f})
                        else:
                            for y in range(len(time_vector)):
                                if this_equivalence not in dict_energy_demand_trn:
                                    dict_energy_demand_trn[this_equivalence] = [0] * len(time_vector)
                                dict_energy_demand_trn[this_equivalence][y] += deepcopy(dict_trn_pj[this_f][y])
                
                    return dict_eq_trn_fuels_rev, dict_energy_demand_trn
                
                # Function #56
                def perform_transport_model_test(fuel_sets_all, dict_energy_demand, dict_energy_demand_trn, time_vector):
                    """
                    Perform a test to review the error in transport and simple energy modeling by comparing the original and adjusted energy demands.

                    Args:
                    - fuel_sets_all (list): List of all fuel types.
                    - dict_energy_demand (dict): Dictionary containing the original energy demand data.
                    - dict_energy_demand_trn (dict): Dictionary containing the adjusted energy demand data for transport.
                    - time_vector (list): List of years for which the calculations are performed.

                    Returns:
                    - fuel_sets_trn (list): List of fuel types with discrepancies between the original and adjusted demands.
                    - diff_dict_all (dict): Dictionary of differences in energy demand for each fuel type.
                    - error_list_all (list): List of percentage errors in total energy demand for each year.
                    """
                    fuel_sets_trn = []
                    diff_list_all = [0 for _ in range(len(time_vector))]
                    diff_dict_all = {}
                    sum_list_all = [0 for _ in range(len(time_vector))]
                
                    for fs in fuel_sets_all:
                        diff_list_fs = []
                        for y in range(len(time_vector)):
                            this_orig_dem = dict_energy_demand['Transport'][fs][y]
                            this_orig_adj = dict_energy_demand_trn[fs][y]
                            this_diff = this_orig_dem - this_orig_adj
                            diff_list_fs.append(this_diff)
                            if round(this_diff, 10) != 0:
                                fuel_sets_trn.append(fs)
                            diff_list_all[y] += this_diff
                            sum_list_all[y] += this_orig_adj
                        diff_dict_all.update({fs: diff_list_fs})
                
                    error_list_all = [100 * diff_list_all[n] / v if v != 0 else 0 for n, v in enumerate(sum_list_all)]
                    return fuel_sets_trn, diff_dict_all, error_list_all
                
                # Function #57
                def add_fuel_consumption_per_transport_tech(types_all, fuels, dict_trn_pj_2, dict_energy_demand):
                    """
                    Add fuel consumption per transport technology to the energy demand dictionary.

                    Args:
                    - types_all (list): List of all transport technology types.
                    - fuels (list): List of all fuel types.
                    - dict_trn_pj_2 (dict): Dictionary containing transport energy demand in PJ for each fuel and technology type.
                    - dict_energy_demand (dict): Dictionary to store the updated energy demand data.

                    Returns:
                    - dict_energy_demand (dict): Updated dictionary with fuel consumption data added for each transport technology type.
                    """
                    for t in types_all:
                        dict_energy_demand[t] = {}
                        for af in fuels:
                            add_dict_trn_pj_2_list = dict_trn_pj_2[af][t]
                            dict_energy_demand[t].update({af: add_dict_trn_pj_2_list})
                    return dict_energy_demand
                
                # Example usage
                dict_eq_trn_fuels_rev, dict_energy_demand_trn = update_transport_vector(fuels, dict_eq_transport_fuels, dict_trn_pj, time_vector, \
                                                                                        dict_energy_demand_trn,dict_eq_trn_fuels_rev)
                fuel_sets_all = list(dict_energy_demand['Transport'].keys())
                fuel_sets_trn, diff_dict_all, error_list_all = perform_transport_model_test(fuel_sets_all, dict_energy_demand, dict_energy_demand_trn, time_vector)
                dict_energy_demand['Transport'] = deepcopy(dict_energy_demand_trn)
                dict_energy_demand = add_fuel_consumption_per_transport_tech(types_all, fuels, dict_trn_pj_2, dict_energy_demand)

            ###########################################################

            # ... here we already have a country's demands, now...
            # 3f) open the *externality* and *emission factors*
            # 'externality':
            
            # Function #58
            def extract_externality_data(df_ext, country):
                """
                Extract externality data for different fuels from a given country from the DataFrame.

                Args:
                - df_ext (DataFrame): DataFrame containing externality data.
                - country (str): The country for which the externality data is to be extracted.

                Returns:
                - externality_fuels_dict (dict): A dictionary where keys are fuel types and values are dictionaries with 'Global warming' and 'Local pollution' costs.
                - externality_fuels_list (list): A list of fuels for which externality data is available.
                """
                externality_fuels_dict = {}
                externality_fuels_list = []
                
                df_cost_externalities = filter_dataframe(df_ext, 'two_columns_scenarios', scenario=country, scenario_2='Yes',\
                    column='Country', column_2='Use_row')
                df_cost_externalities_fuels = df_cost_externalities['Fuel'].tolist()
                # Iterate through the filtered DataFrame to populate the dictionary and list
                for i in range(len(df_cost_externalities_fuels)):
                    this_fuel = df_cost_externalities_fuels[i]
                    externality_fuels_list.append(this_fuel)
                    externality_fuels_dict[this_fuel] = {
                        'Global warming': df_cost_externalities.loc[i, 'Global warming'],
                        'Local pollution': df_cost_externalities.loc[i, 'Local pollution']
                    }
                
                return externality_fuels_dict, externality_fuels_list
            
            # Function #59
            def calculate_add_value(dict_energy_demand, externality_fuels_dict, tech, fuel, y, param):
                """
                Calculate the additional value for global warming based on the energy demand and externalities for a specific technology, fuel, and year. This function gracefully handles the absence of specified keys in the dictionaries.

                Args:
                - dict_energy_demand (dict): Dictionary containing energy demand data, structured by technology and fuel.
                - externality_fuels_dict (dict): Dictionary containing fuel externalities data, structured by fuel and externality parameters.
                - tech (str): The technology for which the calculation is performed.
                - fuel (str): The fuel for which the calculation is performed.
                - y (int): The year for which the calculation is performed.
                - param (str): The parameter for which the externality is calculated (e.g., 'Global warming').

                Returns:
                - float: The calculated additional value for the specified parameter (e.g., global warming) or 0 if the data is not available or an exception occurs.
                """
                try:
                    add_value = dict_energy_demand[tech][fuel][y] * externality_fuels_dict[fuel][param]
                except Exception:
                    add_value = 0
                return add_value

            
            # Function #60
            def calculate_emissions_externalities(demand_tech_list, dict_energy_demand, time_vector, \
            										emissions_fuels_dict, emissions_2_fuels_dict, \
            										externality_fuels_dict, emissions_demand, \
                                                        emissions_demand_black_carbon):
                """
                Calculate emissions and externalities for different demand technologies based on energy demand, fuel type, and time.

                Args:
                - demand_tech_list (list): List of demand technologies.
                - dict_energy_demand (dict): Dictionary containing energy demand data.
                - time_vector (list): List of years for which the calculations are performed.
                - emissions_fuels_dict (dict): Dictionary containing emissions factors for different fuels.
                - emissions_2_fuels_dict (dict): Dictionary containing secondary emissions factors for different fuels.
                - externality_fuels_dict (dict): Dictionary containing externality costs for different fuels.
                - emissions_demand (dict): Dictionary to store the calculated emissions data.
                - emissions_demand_black_carbon (dict): Dictionary to store the calculated secondary emissions data.

                Returns:
                - emissions_demand (dict): Updated dictionary with calculated emissions data for each technology and fuel.
                - emissions_demand_black_carbon (dict): Updated dictionary with calculated secondary emissions data for each technology and fuel.
                - externalities_globalwarming_demand (dict): Dictionary with calculated global warming externalities for each technology and fuel.
                - externalities_localpollution_demand (dict): Dictionary with calculated local pollution externalities for each technology and fuel.
                """
                externalities_globalwarming_demand = {}
                externalities_localpollution_demand = {}
            
                for tech in demand_tech_list:
                    demand_fuel_list = list(dict_energy_demand[tech].keys())
                    emissions_demand[tech] = {}
                    emissions_demand_black_carbon[tech] = {}
                    externalities_globalwarming_demand[tech] = {}
                    externalities_localpollution_demand[tech] = {}
            
                    for fuel in demand_fuel_list:
                        if fuel in emissions_fuels_list:  # store emissions
                            list_emissions_demand = []
                            for y in range(len(time_vector)):
                                add_value = \
                                    dict_energy_demand[tech][fuel][y]*emissions_fuels_dict[fuel][y]
                                list_emissions_demand.append(add_value)
                            emissions_demand[tech].update({fuel:list_emissions_demand})
            				
                        if fuel in emissions_2_fuels_dict: # Store emissions 2
                                emissions_demand_black_carbon[tech][fuel] = [
                                dict_energy_demand[tech][fuel][y] * emissions_2_fuels_dict[fuel][y]
                                for y in range(len(time_vector))]

                        if fuel in externality_fuels_list:  # store externalities
                            list_globalwarming_demand = []
                            list_localpollution_demand = []
                            for y in range(len(time_vector)):
                                add_value_globalwarming = calculate_add_value(dict_energy_demand, externality_fuels_dict, tech, fuel, y, 'Global warming')
                                list_globalwarming_demand.append(add_value_globalwarming)
                                add_value_localpollution = calculate_add_value(dict_energy_demand, externality_fuels_dict, tech, fuel, y, 'Local pollution')
                                list_localpollution_demand.append(add_value_localpollution)
    
                            externalities_globalwarming_demand[tech].update({fuel:list_globalwarming_demand})
                            externalities_localpollution_demand[tech].update({fuel:list_localpollution_demand})
            
                return emissions_demand, emissions_demand_black_carbon, externalities_globalwarming_demand, externalities_localpollution_demand

            # Example usage
            externality_fuels_dict, externality_fuels_list = extract_externality_data(df5_ext, this_country)
            demand_tech_list = [i for i in dict_energy_demand.keys() if i not in types_all]
            emissions_demand, emissions_demand_black_carbon, externalities_globalwarming_demand, \
            externalities_localpollution_demand = calculate_emissions_externalities(demand_tech_list, \
            										dict_energy_demand, time_vector, \
            										emissions_fuels_dict, emissions_2_fuels_dict, \
            										externality_fuels_dict, emissions_demand, \
                                                        emissions_demand_black_carbon)
            ext_by_country.update({this_country:deepcopy(externality_fuels_dict)})
            dict_local_country[this_country].update({'Global warming externalities by demand': \
            											externalities_globalwarming_demand})
            dict_local_country[this_country].update({'Local pollution externalities by demand': \
            											externalities_localpollution_demand})
            dict_local_country[this_country].update({'Emissions by demand': emissions_demand})
            dict_local_country[this_country].update({'Black carbon emissions by demand [ton]':  \
                                                        emissions_demand_black_carbon})

            # Select the existing capacity from the base cap dictionary:
            dict_base_caps = \
                dict_database['Cap'][this_reg][this_country]  # by pp type

            # Function #62
            def apply_fuel_price_projection(df, fuel_idx, projection, value_type, time_vector):
                """
                Apply the fuel price projection based on the projection type and value type.
            
                Args:
                df (pd.DataFrame): The DataFrame containing fuel price data.
                fuel_idx (int): The index of the fuel in the DataFrame.
                projection (str): The type of price projection.
                value_type (str): The type of value for the price projection.
                time_vector (list): List of time periods for the projection.
                """
                if projection == 'flat' and value_type == 'constant':
                    for y in time_vector:
                        df.loc[fuel_idx, y] = float(df.loc[fuel_idx, time_vector[0]])
                        
                if projection == 'Interpolate':
                    for y in range(len(time_vector)):
                        df[time_vector[y]] = pd.to_numeric(df[time_vector[y]], errors='coerce')

                    # Convert to numeric and interpolate
                    numeric_data = pd.to_numeric(df.loc[fuel_idx, time_vector], errors='coerce')
                    interpolated_data = numeric_data.interpolate(limit_direction='both')

                    # Assign interpolated values back to DataFrame
                    with warnings.catch_warnings():
                        df.loc[fuel_idx, time_vector] = deepcopy(interpolated_data)
                        
                elif projection == 'Percent growth of incomplete years':
                    growth_param = df.loc[fuel_idx, 'value']
                    for y in range(1, len(time_vector)):
                        value_field = df.loc[fuel_idx, time_vector[y]]
                        if math.isnan(value_field):
                            df.loc[fuel_idx, time_vector[y]] = round(df.loc[fuel_idx, time_vector[y-1]] * (1 + growth_param / 100), 10)
                            
            # Parameters related definition
            # Select the "param_related_4"
            param_related_4 = 'Distribution of new electrical energy generation'
            df_param_related_4 = filter_dataframe(df_scen_rc, 'scenario_simple', scenario=param_related_4, column='Parameter')
            
            list_electric_sets = \
                df_param_related_4['Tech'].tolist()

            mask_filt_techs = \
                ((d5_power_techs['Projection'] != 'none') &
                 (d5_power_techs['Parameter'] == 'Net capacity factor'))
            list_electric_sets_2 = \
                d5_power_techs.loc[mask_filt_techs]['Tech'].tolist()

            list_electric_sets_3 = list(set(list(dict_base_caps.keys())) & set(list_electric_sets_2))
            list_electric_sets_3.sort()

            # > Call other auxiliary variables:
            
            # Select the "param_related_5"
            param_related_5 = '% Imports for consumption'
            df_param_related_5 = filter_dataframe(df_scen_rc, 'scenario_simple', scenario=param_related_5, column='Parameter')

            # Select the "param_related_6"
            param_related_6 = '% Exports for production'
            df_param_related_6 = filter_dataframe(df_scen_rc, 'scenario_simple', scenario=param_related_6, column='Parameter')

            # Select the "param_related_7"
            param_related_7 = 'Fuel prices'
            df_param_related_7 = filter_dataframe(df_scen_rc, 'scenario_simple', scenario=param_related_7, column='Parameter')

            # ...proceed with the interpolation of fuel prices:
            fuel_list_local = df_param_related_7['Fuel'].tolist()

            for this_fuel in fuel_list_local:
                fuel_idx_7 = df_param_related_7['Fuel'].tolist().index(this_fuel)
                this_fuel_price_projection = df_param_related_7.loc[fuel_idx_7, 'projection']
                this_fuel_price_value_type = df_param_related_7.loc[fuel_idx_7, 'value']
            
                apply_fuel_price_projection(df_param_related_7, fuel_idx_7, this_fuel_price_projection, this_fuel_price_value_type, time_vector)

            # Select the "param_related_8"
            param_related_8 = 'Planned new capacity'
            df_param_related_8 = filter_dataframe(df_scen_rc, 'scenario_simple', scenario=param_related_8, column='Parameter')

            # Select the "param_related_9"
            param_related_9 = 'Phase-out capacity'
            df_param_related_9 = filter_dataframe(df_scen_rc, 'scenario_simple', scenario=param_related_9, column='Parameter')

            # Select the "param_related_10"
            param_related_10 = 'Capacity factor change'
            df_param_related_10 = filter_dataframe(df_scen_rc, 'scenario_simple', scenario=param_related_10, column='Parameter')

            # Select the existing transformation inputs information (receive negative values only):
            dict_base_transformation = \
                dict_database['EB'][this_reg][this_country_2]['Total transformation']['Power plants']  # by fuel (input)
            dict_base_transformation_2 = \
                dict_database['EB'][this_reg][this_country_2]['Total transformation']['Self-producers']  # by fuel (input)

            # Function #63
            def generate_base_transformation_fuels(dict_base_transformation, dict_base_transformation_2):
                """
                Generate a sorted list of base transformation fuels by combining and deduplicating fuel types from two different sources.

                Args:
                - dict_base_transformation (dict): First dictionary containing base transformation data.
                - dict_base_transformation_2 (dict): Second dictionary containing base transformation data.

                Returns:
                - base_transformation_fuels (list): Sorted list of unique fuels derived from both dictionaries.
                """
                base_transformation_fuels = list(set(dict_base_transformation.keys()) | set(dict_base_transformation_2.keys()))
                base_transformation_fuels.sort()
                return base_transformation_fuels
            
            # Function #64
            def calculate_fuel_use_production(base_transformation_fuels, dict_base_transformation, dict_base_transformation_2, base_year):
                """
                Calculate fuel use and production for electricity by analyzing the base transformation data for various fuels.

                Args:
                - base_transformation_fuels (list): List of fuels involved in base transformation.
                - dict_base_transformation (dict): Dictionary containing base transformation data for primary power stations.
                - dict_base_transformation_2 (dict): Dictionary containing base transformation data for secondary power stations.
                - base_year (int): The base year for which the calculations are performed.

                Returns:
                - base_electric_fuels_use (list): List of fuels used in electricity production.
                - base_electric_fuel_use (dict): Dictionary of fuel use quantities.
                - base_electric_production (dict): Dictionary of electricity production quantities.
                - base_electric_production_pps (dict): Dictionary of electricity production percentages for primary power stations.
                - base_electric_production_sps (dict): Dictionary of electricity production percentages for secondary power stations.
                """
                base_electric_fuels_use = []
                base_electric_fuel_use = {}
                base_electric_production = {}
                base_electric_production_pps = {}
                base_electric_production_sps = {}
            
                for btf in base_transformation_fuels:
                    btf_value_1 = dict_base_transformation.get(btf, {}).get(base_year, 0)
                    btf_value_2 = dict_base_transformation_2.get(btf, {}).get(base_year, 0)
            
                    btf_value = btf_value_1 + btf_value_2
                    if btf_value < 0:
                        base_electric_fuels_use.append(btf)
                        base_electric_fuel_use[btf] = -1 * btf_value
                    if btf_value > 0:
                        base_electric_production[btf] = btf_value
                        base_electric_production_pps[btf] = btf_value_1 / btf_value
                        base_electric_production_sps[btf] = btf_value_2 / btf_value
            
                return base_electric_fuels_use, base_electric_fuel_use, base_electric_production, base_electric_production_pps, base_electric_production_sps
            
            # Function #65
            def extract_electricity_losses_self_consumption(dict_database, this_reg, this_country_2, base_year, dict_energy_demand_by_fuel, time_vector):
                """
                Extract electricity losses, self-consumption, imports, and exports shares based on the provided database, region, and country information.

                Args:
                - dict_database (dict): The database containing electricity data.
                - this_reg (str): Region identifier.
                - this_country_2 (str): Country identifier.
                - base_year (int): The base year for calculations.
                - dict_energy_demand_by_fuel (dict): Dictionary of energy demand by fuel.
                - time_vector (list): List of time periods for the projection.

                Returns:
                - electricity_losses_list (list): List of electricity losses for each year in the time vector.
                - electricity_self_consumption_list (list): List of electricity self-consumption for each year in the time vector.
                - electricity_imports_list (list): List of electricity imports for each year in the time vector.
                - electricity_exports_list (list): List of electricity exports for each year in the time vector.
                - losses_share (float): The share of losses in the base year.
                - self_consumption_share (float): The share of self-consumption in the base year.
                - imports_share (float): The share of imports in the base year.
                - exports_share (float): The share of exports in the base year.
                """
                electricity_losses = dict_database['EB'][this_reg][this_country_2]['Losses']['none']['Electricity'][base_year]
                electricity_self_consumption = dict_database['EB'][this_reg][this_country_2]['Self-consumption']['none']['Electricity'][base_year]
                electricity_imports = dict_database['EB'][this_reg][this_country_2]['Total supply']['Imports']['Electricity'][base_year]
                electricity_exports = dict_database['EB'][this_reg][this_country_2]['Total supply']['Exports']['Electricity'][base_year]
            
                base_electricity_demand = dict_energy_demand_by_fuel['Electricity'][0]
                losses_share = electricity_losses / base_electricity_demand
                self_consumption_share = electricity_self_consumption / base_electricity_demand
                imports_share = electricity_imports / base_electricity_demand
                exports_share = electricity_exports / base_electricity_demand
            
                electricity_losses_list = [losses_share * demand for demand in dict_energy_demand_by_fuel['Electricity']]
                electricity_self_consumption_list = [self_consumption_share * demand for demand in dict_energy_demand_by_fuel['Electricity']]
                electricity_imports_list = [imports_share * demand for demand in dict_energy_demand_by_fuel['Electricity']]
                electricity_exports_list = [exports_share * demand for demand in dict_energy_demand_by_fuel['Electricity']]
            
                return electricity_losses_list, electricity_self_consumption_list, electricity_imports_list, electricity_exports_list, \
                    losses_share, self_consumption_share, imports_share, exports_share

            # Example usage
            base_transformation_fuels = generate_base_transformation_fuels(dict_base_transformation, dict_base_transformation_2)
            base_electric_fuels_use, base_electric_fuel_use, base_electric_production, base_electric_production_pps, \
                base_electric_production_sps = calculate_fuel_use_production(base_transformation_fuels, dict_base_transformation, \
                                                                             dict_base_transformation_2, base_year)
            electricity_losses_list, electricity_self_consumption_list, electricity_imports_list, electricity_exports_list, \
                losses_share, self_consumption_share, imports_share, exports_share = \
                    extract_electricity_losses_self_consumption(dict_database, this_reg, this_country_2, base_year, \
                                                                dict_energy_demand_by_fuel, time_vector)

            # ...here we must manipulate the limit to the losses!
            # Select the "param_related_11"
            param_related_11 = 'Max losses'
            df_param_related_11 = filter_dataframe(df_scen_rc, 'scenario_simple', scenario=param_related_11, column='Parameter')

            # Function #66
            def create_loss_vector(df_param, time_vector, projection, baseyear_str, losses_share):
                """
                Create a vector of losses based on the projection method and initial shares.

                Args:
                - df_param (DataFrame): DataFrame containing parameters related to losses.
                - time_vector (list): List of years for which the calculations are performed.
                - projection (str): Method used for projecting losses ('flat' or 'interpolate').
                - baseyear_str (str): Indicator of whether the base year is 'endogenous' or a different setting.
                - losses_share (float): Share of losses in the base year.

                Returns:
                - loss_vector (list): List of loss shares for each year in the time vector.
                """
                loss_vector = []
            
                if projection == 'flat' and baseyear_str == 'endogenous':
                    loss_vector = [losses_share for _ in range(len(time_vector))]
            
                elif projection == 'interpolate' and baseyear_str == 'endogenous':
                    known_loss_vals = []
                    for y in range(len(time_vector)):
                        if y == 0 or type(df_param.loc[0, time_vector[y]]) is int:
                            suggested_maxloss = df_param.loc[0, time_vector[y]] / 100 if y != 0 else losses_share
                            known_loss_vals.append(min(suggested_maxloss, losses_share))
                        else:
                            known_loss_vals.append('')
                    loss_vector = interpolation_to_end(time_vector, ini_simu_yr, known_loss_vals, 'ini', '')
                    
            
                return loss_vector
            
            # Function #67
            def apply_loss_vector(time_vector, loss_vector, dict_energy_demand_by_fuel, losses_share, self_consumption_share, imports_share, exports_share):
                """
                Apply the loss vector to calculate lists of electricity losses, self-consumption, imports, and exports for each year.

                Args:
                - time_vector (list): List of years for which the calculations are performed.
                - loss_vector (list): List of loss shares for each year.
                - dict_energy_demand_by_fuel (dict): Dictionary containing energy demand data for different fuels.
                - losses_share (float): Share of losses in the base year.
                - self_consumption_share (float): Share of self-consumption in the base year.
                - imports_share (float): Share of imports in the base year.
                - exports_share (float): Share of exports in the base year.

                Returns:
                - electricity_losses_list (list): List of electricity losses for each year.
                - electricity_self_consumption_list (list): List of electricity self-consumption for each year.
                - electricity_imports_list (list): List of electricity imports for each year.
                - electricity_exports_list (list): List of electricity exports for each year.
                """
                electricity_losses_list = [loss * dict_energy_demand_by_fuel['Electricity'][y] for y, loss in enumerate(loss_vector)]
                electricity_self_consumption_list = [self_consumption_share * dict_energy_demand_by_fuel['Electricity'][y] for y in range(len(time_vector))]
                electricity_imports_list = [imports_share * dict_energy_demand_by_fuel['Electricity'][y] for y in range(len(time_vector))]
                electricity_exports_list = [exports_share * dict_energy_demand_by_fuel['Electricity'][y] for y in range(len(time_vector))]
            
                return electricity_losses_list, electricity_self_consumption_list, electricity_imports_list, electricity_exports_list
            
            # Example usage
            maxloss_projection = df_param_related_11.iloc[0]['projection']
            maxloss_baseyear_str = df_param_related_11.loc[0, time_vector[0]]
            loss_vector = create_loss_vector(df_param_related_11, time_vector, maxloss_projection, maxloss_baseyear_str, losses_share)
            electricity_losses_list, electricity_self_consumption_list, electricity_imports_list, electricity_exports_list = apply_loss_vector(time_vector, loss_vector, dict_energy_demand_by_fuel, losses_share, self_consumption_share, imports_share, exports_share)


            # 3g) Here we must call some inputs to make the model adjust to the desired electrical demands
            param_related_12 = 'RE TAG'
            df_param_related_12 = filter_dataframe(df_scen_rc, 'scenario_simple', scenario=param_related_12, column='Parameter')

            reno_targets_exist = False
            if len(df_param_related_12.index.tolist()) > 0:
                reno_targets_exist = True
                reno_target_list = df_param_related_12[time_vector].iloc[0].tolist()

            # 3h) obtain the required *new electrical capacity by pp*, *electricity production by pp*, *fuel consumption by pp*
            #     to supply the electricity demand:
            # NOTE: we have to subtract the elements
            electrical_demand_to_supply = [0 for y in range(len(time_vector))]
            for y in range(len(time_vector)):
                electrical_demand_to_supply[y] = \
                    dict_energy_demand_by_fuel['Electricity'][y] + \
                    electricity_losses_list[y] + \
                    electricity_self_consumption_list[y] - \
                    electricity_imports_list[y] - \
                    electricity_exports_list[y]

            # ...here, 'Total' is the net energy loss in transformation
            base_electric_prod = base_electric_production['Electricity']
            base_electric_use_fuels = \
                deepcopy(base_electric_fuels_use)
            #base_electric_use_fuels.remove('Total')  # not needed anymore
            #base_electric_use_fuels.remove('Total primary sources')

            # ...we can extract the fuels we use in our technological options:
            used_fuel_list = []
            for tech in list_electric_sets:
                used_fuel = tech.split('_')[-1]
                if used_fuel not in used_fuel_list:
                    used_fuel_list.append(used_fuel)

            # ...here we need to iterate across the list of technologies and find the base distribution of electricity production:
            res_energy_shares = {}
            res_energy_sum_1 = 0
            #res_energy_sum_2 = 0
            #res_energy_sum_3 = 0
            store_percent = {}
            store_percent_rem = {}
            store_use_cap = {}
            store_res_energy = {}
            store_res_energy_all = [0 for y in range(len(time_vector))]

            # Blank capacity factors:
            cf_by_tech = {}
            forced_newcap_energy_by_tech = {}
            forced_newcap_by_tech = {}
            forced_newcap_energy_all = [0 for y in range(len(time_vector))]

            accum_forced_newcap_by_tech = {}
            # ...this is the first previous loop:
            # 1st power sector loop
            # Function #68
            def initialize_data_structures(list_electric_sets, time_vector):
                """
                Initialize data structures for capacity and energy storage for different electricity sets.

                Args:
                - list_electric_sets (list): List of different electricity sets or technologies.
                - time_vector (list): List of years for which the calculations are performed.

                Returns:
                - data_structures (dict): Dictionary with initialized lists of zeroes for each technology in the electric sets.
                """
                data_structures = {tech: [0 for _ in time_vector] for tech in list_electric_sets}
                return data_structures
            
            # Function #69
            def get_technology_indices(tech, df_param_related_8, df_param_related_9):
                """
                Get indices for a specific technology in the provided dataframes.

                Args:
                - tech (str): The specific technology for which the indices are required.
                - df_param_related_8 (DataFrame): First DataFrame containing technology data.
                - df_param_related_9 (DataFrame): Second DataFrame containing technology data.

                Returns:
                - tech_idx_8 (str): Index of the technology in the first DataFrame, or an empty string if not found.
                - tech_idx_9 (str): Index of the technology in the second DataFrame, or an empty string if not found.
                """
                tech_idx_8 = get_tech_index(df_param_related_8, tech)
                tech_idx_9 = get_tech_index(df_param_related_9, tech)
                return tech_idx_8, tech_idx_9
            
            # Function #70
            def get_tech_index(df, tech):
                """
                Get the index of a specific technology in a DataFrame.

                Args:
                - df (DataFrame): DataFrame containing technology data.
                - tech (str): The name of the technology to find.

                Returns:
                - tech_idx (int or str): The index of the technology in the DataFrame or an empty string if not found.
                """
                try:
                    return df['Tech'].tolist().index(tech)
                except Exception:
                    return ''
            
            # Function #71
            def extract_technical_characteristics(d5_power_techs, tech):
                """
                Extract technical characteristics for a specific technology from a DataFrame.

                Args:
                - d5_power_techs (DataFrame): DataFrame containing technical characteristics of different technologies.
                - tech (str): The specific technology for which the characteristics are extracted.

                Returns:
                - DataFrame: A new DataFrame containing technical characteristics of the specified technology.
                """
                mask_this_tech = (d5_power_techs['Tech'] == tech)
                return deepcopy(d5_power_techs.loc[mask_this_tech])
            
            # Function #72
            def calculate_parameter_values(df, parameter, time_vector):
                """
                Calculate parameter values based on the projection method, unit multiplier, and initial values in the DataFrame.

                Args:
                - df (DataFrame): DataFrame containing parameter data.
                - parameter (str): The specific parameter for which the values are calculated.
                - time_vector (list): List of years for which the calculations are performed.

                Returns:
                - list_param_values (list): List of calculated parameter values for each year.
                - df_mask_param (DataFrame): DataFrame filtered by the specific parameter.
                - unit_multiplier (float): The unit multiplier used for converting parameter values.
                - projection (str): The projection method used ('user_defined' or 'flat').
                """
                mask_param = df['Parameter'] == parameter
                df_mask_param = deepcopy(df.loc[mask_param])
                unit_multiplier = df_mask_param.iloc[0]['Unit multiplier']
                projection = df_mask_param.iloc[0]['Projection']
                list_param_values = []
            
                if projection == 'user_defined':
                    for y in time_vector:
                        add_value = df_mask_param.iloc[0][y] * unit_multiplier
                        list_param_values.append(add_value)
                elif projection == 'flat':
                    flat_value = df_mask_param.iloc[0][time_vector[0]] * unit_multiplier
                    list_param_values = [flat_value for _ in time_vector]
            
                return list_param_values, df_mask_param, unit_multiplier, projection
            
            # Power sector loop
            store_use_cap = initialize_data_structures(list_electric_sets_3, time_vector)
            store_res_energy = initialize_data_structures(list_electric_sets_3, time_vector)
            forced_newcap_energy_by_tech = initialize_data_structures(list_electric_sets_3, time_vector)
            forced_newcap_by_tech = initialize_data_structures(list_electric_sets_3, time_vector)
            accum_forced_newcap_by_tech = initialize_data_structures(list_electric_sets_3, time_vector)
            coco = 0
            for tech in list_electric_sets_3:
                coco+=1
                tech_idx_8, tech_idx_9 = get_technology_indices(tech, df_param_related_8, df_param_related_9)
                this_tech_df_cost_power_techs = extract_technical_characteristics(d5_power_techs, tech)
                this_tech_base_cap = dict_base_caps[tech][base_year]
                # CAU
                list_tech_cau, df_mask_cau, df_mask_cau_unitmult, df_mask_cau_proj = calculate_parameter_values(this_tech_df_cost_power_techs, 'CAU', time_vector)
                # Net capacity factor
                list_tech_cf, df_mask_cf, df_mask_cf_unitmult, df_mask_cf_proj = calculate_parameter_values(this_tech_df_cost_power_techs, 'Net capacity factor', time_vector)
                 
                mask_cf_change = (df_param_related_10['Tech'] == tech)
                df_ch_change = df_param_related_10.loc[mask_cf_change]
                if not df_ch_change.empty:
                    df_ch_change_proj = df_ch_change['projection'].iloc[0]
                    if df_ch_change_proj == 'user_defined':
                        df_ch_change_list = [df_ch_change[y].iloc[0] for y in time_vector]
                    if df_ch_change_proj == 'flat':
                        df_ch_change_list = [df_ch_change[0].iloc[0] for y in time_vector]

                # Here we call the capacity factors from an external analysis:
                if df_mask_cf_proj in ['normalize_by_endogenous', 'flat_endogenous']:
                    ###########################################################                    
                    # Function #72
                    def extract_capacity_factor_data(df, tech, year, country):
                        """
                        Extract capacity factor data for a specific technology and year from a DataFrame.

                        Args:
                        - df (DataFrame): DataFrame containing capacity factor data.
                        - tech (str): The technology for which capacity factor data is extracted.
                        - year (int): The year for which capacity factor data is extracted.
                        - country (str): The country for which capacity factor data is extracted.

                        Returns:
                        - df_filtered (DataFrame): A filtered DataFrame containing capacity factor data for the specified technology, year, and country.
                        - this_df4_cfs (DataFrame): A copy of the filtered DataFrame containing capacity factor data.
                        """
                        mask = ((df['Capacity Set'] == tech) & (df['Year'] == year) & (df['Country'] == country))
                        df_filtered = deepcopy(df.loc[mask])
                        df_filtered.reset_index(inplace=True, drop=True)
                        this_df4_cfs = deepcopy(df.loc[mask])
                        return df_filtered, this_df4_cfs
                    
                    # Function #73
                    def calculate_average_capacity_factor(df_filtered, column):
                        """
                        Calculate the average capacity factor from the filtered DataFrame.

                        Args:
                        - df_filtered (DataFrame): A filtered DataFrame containing capacity factor data.

                        Returns:
                        - cf_yearly_add (float): The sum of capacity factors for the year.
                        - cf_count (int): The count of non-zero capacity factors for the year.
                        """
                        if len(df_filtered) != 0:
                            cf_yearly = df_filtered[column].iloc[0]
                            if cf_yearly < 1:
                                cf_yearly_add = cf_yearly
                                cf_count = 1
                            else:
                                cf_yearly_add, cf_count = 0, 0
                        else:
                            cf_yearly_add, cf_count = 0, 0
                        return cf_yearly_add, cf_count
                    
                    
                    if this_country == 'Uruguay':
                        years = [2015, 2016, 2017, 2018, 2019, 2020] # The user must to change
                        column_cf = 'Yearly Capacity Factor' # The user must to change
                    else: 
                        years = [2021] # The user must to change
                        column_cf = 'Capacity factor' # The user must to change
                    cf_data = {}
                    cf_yearly_add_data = {}
                    cf_cnt_add_data = {}
                    dict_this_df4_cfs = {}
                    
                    for year in years:
                        df_cf_year, this_df4_cfs = extract_capacity_factor_data(df4_cfs, tech, year, this_country)
                        cf_year_add, cf_year_count = calculate_average_capacity_factor(df_cf_year, column_cf)
                        cf_data[year] = {'add': cf_year_add, 'count': cf_year_count}
                        cf_yearly_add_data[f"this_cf_yrl_{year}_add"] = cf_year_add
                        cf_cnt_add_data[f"this_cf_cnt_{year}_add"] = cf_year_count
                        dict_this_df4_cfs[f"this_cf_cnt_this_df4_cfs_{year}"] = this_df4_cfs

                    ###########################################################
                    if this_country == 'Uruguay':
                        # Estimate the average here:
                        this_cf_avg_num = sum(cf_yearly_add_data.get(f"this_cf_yrl_{year}_add", 0) for year in years if year != 2020)                    
                        this_cf_avg_den = sum(cf_cnt_add_data.get(f"this_cf_cnt_{year}_add", 0) for year in years if year != 2020)
                        try:
                            this_cf_avg = this_cf_avg_num / this_cf_avg_den
                        except Exception:
                            this_cf_avg = 0
                    ###########################################################
                    this_cf_list = []
                    for y in time_vector:
                        mask_cf_tech = \
                            ((df4_cfs['Capacity Set'] == tech) & \
                             (df4_cfs['Year'] == time_vector[0]) & \
                             (df4_cfs['Country'] == this_country)
                            )

                        this_df4_cfs = \
                            deepcopy(df4_cfs.loc[mask_cf_tech])
                        this_df4_cfs.reset_index(inplace=True, drop=True)

                        if len(this_df4_cfs.index.tolist()) != 0 or \
                                len(dict_this_df4_cfs[f'this_cf_cnt_this_df4_cfs_{years[-1]}'].index.tolist()) != 0:  # proceed

                            if len(this_df4_cfs.index.tolist()) != 0:
                                this_cf_yrl = this_df4_cfs[column_cf].iloc[0]
                            else:
                                this_cf_yrl = 99
                                if not this_country == 'Uruguay':
                                    print('This capacity factor condition is not possible for this version. Review inputs and structure.')
                                    sys.exit()
                            if this_country == 'Uruguay':
                                # Select the appropiate historic capacity factors:
                                if (y == 2019 or y == 2020) and this_cf_yrl < 1:
                                    this_cf = this_cf_yrl
                                elif (y == 2019 or y == 2020) and this_cf_avg < 1:
                                    this_cf = this_cf_avg
                                elif y > 2020 and this_cf_avg < 1:
                                    this_cf = this_cf_avg
                                else:
                                    this_cf = 0
                                
                            else:    
                                # Select the appropiate historic capacity factors:
                                this_cf = this_cf_yrl

                        else:
                            this_cf = 0  # this means we must use a default CF // to be selected

                        this_cf_list.append(this_cf)


                # Here we define the capacity factors endogenously:
                # Function #74
                def process_capacity_factor(time_vector, df_mask_cf, df_mask_cf_proj, df_mask_cf_unitmult, this_cf_list):
                    """
                    Process and calculate the capacity factor for a specific technology based on various conditions.

                    Parameters:
                    - time_vector (list): List of time intervals.
                    - df_mask_cf (DataFrame): DataFrame containing capacity factors.
                    - df_mask_cf_proj (str): Projection method for capacity factors.
                    - df_mask_cf_unitmult (float): Unit multiplier for the capacity factors.
                    - df_ch_change_list (list): List of change factors for each year.
                    - this_cf_list (list): List of capacity factors for each year.

                    Returns:
                    - list_tech_cf (list): A list containing the processed capacity factors for each year.
                    """
                    list_tech_cf = []
                    y_counter_i = 0

                    for y in time_vector:
                        this_cf = this_cf_list[y_counter_i]
                        if this_cf != 0:  # this applies only if energy production is existent
                            if df_mask_cf_proj == 'normalize_by_endogenous':
                                mult_value = df_mask_cf.iloc[0][y] / df_mask_cf.iloc[0][time_vector[0]] * df_mask_cf_unitmult
                                add_value = mult_value * this_cf
                            elif df_mask_cf_proj == 'flat_endogenous':
                                add_value = this_cf*df_mask_cf_unitmult
                        else:  # this applies when the technology is non-existent
                            if df_mask_cf_proj == 'normalize_by_endogenous':
                                add_value = df_mask_cf.iloc[0][y] * df_mask_cf_unitmult
                            elif df_mask_cf_proj == 'flat_endogenous':
                                add_value = df_mask_cf.iloc[0][time_vector[0]]*df_mask_cf_unitmult
                        list_tech_cf.append(add_value)
                        y_counter_i += 1
                    
                    return list_tech_cf
                

                # Function #75
                def calculate_phase_out_capacity(tech_idx, df_param, time_vector):
                    """
                    Calculate the phased-out capacity for a technology based on the provided data.

                    Args:
                    - tech_idx (int/str): Index of the technology in the DataFrame.
                    - df_param (DataFrame): DataFrame containing phase-out data for technologies.
                    - time_vector (list): List of years for which the calculations are performed.

                    Returns:
                    - phase_out_cap (list): List of phased-out capacities for each year.
                    """
                    phase_out_cap = [0 for _ in range(len(time_vector))]
                    if tech_idx != '':
                        for y in range(len(time_vector)):
                            phase_out_cap[y] = df_param.loc[tech_idx, time_vector[y]] / 1000  # Convert MW to GW
                    return phase_out_cap
                # Function #76
                def calculate_res_energy_base_and_update_cf(tech_base_cap, list_tech_cf):
                    """
                    Calculate the base residual energy for a technology and update its capacity factors.

                    Args:
                    - tech_base_cap (float): Base capacity of the technology in MW.
                    - list_tech_cf (list): List of capacity factors for the technology.

                    Returns:
                    - res_energy_base (float): Calculated base residual energy in GW.
                    """
                    # Convert MW to GW and multiply by the number of hours in a year (8760)
                    res_energy_base = tech_base_cap * 8760 * list_tech_cf / 1000
                    return res_energy_base

                # Function #77
                def calculate_forced_capacity(tech_idx, df_param, time_vector):
                    """
                    Calculate and accumulate forced new capacity for a technology based on the provided data.

                    Args:
                    - tech_idx (int/str): Index of the technology in the DataFrame.
                    - df_param (DataFrame): DataFrame containing forced capacity data for technologies.
                    - time_vector (list): List of years for which the calculations are performed.

                    Returns:
                    - forced_cap_vector (list): List of forced new capacities for each year.
                    - accum_forced_cap_vector (list): List of accumulated forced new capacities for each year.
                    """
                    forced_cap_vector = [0 for _ in range(len(time_vector))]
                    accum_forced_cap_vector = [0 for _ in range(len(time_vector))]
                    for y in range(len(time_vector)):
                        if tech_idx != '':
                            forced_cap = df_param.loc[tech_idx, time_vector[y]] / 1000  # Convert MW to GW
                        else:
                            forced_cap = 0
                        forced_cap_vector[y] = forced_cap
                        accum_forced_cap_vector[y] = sum(forced_cap_vector[:y+1])
                    return forced_cap_vector, accum_forced_cap_vector
                
                # Function #78
                def calculate_residual_capacity_and_energy(tech_base_cap, this_tech_phase_out_cap, list_tech_cau, list_tech_cf, time_vector):
                    """
                    Calculate the residual capacity and energy production for a technology.

                    Args:
                    - tech_base_cap (float): Base capacity of the technology in MW.
                    - this_tech_phase_out_cap (list): List of phased-out capacities for the technology.
                    - list_tech_cau (list): List of capacity utilization factors for the technology.
                    - list_tech_cf (list): List of capacity factors for the technology.
                    - time_vector (list): List of years for which the calculations are performed.

                    Returns:
                    - residual_cap_vector (list): List of residual capacities for each year.
                    - res_energy_vector (list): List of residual energy productions for each year.
                    """
                    residual_cap_vector = [0 for _ in range(len(time_vector))]
                    res_energy_vector = [0 for _ in range(len(time_vector))]
                
                    for y in range(len(time_vector)):
                        if y == 0:
                            residual_cap_vector[y] = tech_base_cap / 1000  # Convert MW to GW
                        else:
                            residual_cap_vector[y] = residual_cap_vector[y-1] - this_tech_phase_out_cap[y]
                
                        res_energy_vector[y] = residual_cap_vector[y] * list_tech_cau[y] * list_tech_cf[y]
                
                    return residual_cap_vector, res_energy_vector

                # Calculate the endogenous capacity factors
                # try:
                if df_mask_cf_proj == 'normalize_by_endogenous':
                    # list_tech_cf = process_capacity_factor(time_vector, df_mask_cf, df_mask_cf_proj, df_mask_cf_unitmult, df_ch_change_list, this_cf_list)
                    list_tech_cf = process_capacity_factor(time_vector, df_mask_cf, df_mask_cf_proj, df_mask_cf_unitmult, this_cf_list)
                if df_mask_cf_proj == 'flat_endogenous':
                    # list_tech_cf = process_capacity_factor(time_vector, df_mask_cf, df_mask_cf_proj, df_mask_cf_unitmult, df_ch_change_list, this_cf_list)
                    list_tech_cf = process_capacity_factor(time_vector, df_mask_cf, df_mask_cf_proj, df_mask_cf_unitmult, this_cf_list)
                    
                this_tech_phase_out_cap = calculate_phase_out_capacity(tech_idx_9, df_param_related_9, time_vector)
                res_energy_base_1 = calculate_res_energy_base_and_update_cf(this_tech_base_cap, list_tech_cf[0])
                res_energy_sum_1 += res_energy_base_1

                cf_by_tech.update({tech: deepcopy(list_tech_cf)})
                
                forced_cap_vector, accum_forced_cap_vector = calculate_forced_capacity(tech_idx_8, df_param_related_8, time_vector)
                residual_cap_vector, res_energy_vector = calculate_residual_capacity_and_energy(this_tech_base_cap, this_tech_phase_out_cap, list_tech_cau, list_tech_cf, time_vector)
                
                for y in range(len(time_vector)):
                    use_cap = residual_cap_vector[y]
                    res_energy = res_energy_vector[y]
            
                    store_use_cap[tech][y] += use_cap
                    store_res_energy[tech][y] += res_energy
                    store_res_energy_all[y] += res_energy
            
                    forced_newcap_energy = accum_forced_cap_vector[y] * list_tech_cau[y] * list_tech_cf[y]
                    forced_newcap_energy_by_tech[tech][y] += forced_newcap_energy
                    forced_newcap_by_tech[tech][y] += forced_cap_vector[y]
                    forced_newcap_energy_all[y] += forced_newcap_energy
            
                    accum_forced_newcap_by_tech[tech][y] += accum_forced_cap_vector[y]

            # Store the energy of the base year:
            store_res_energy_orig = deepcopy(store_res_energy)
            dict_store_res_energy_orig.update({this_country_2: deepcopy(store_res_energy_orig)})

            # ...this is the second previous loop:
            # 2nd power sector loop            
            # Function #79
            def calculate_energy_base_and_distribution(tech, dict_base_caps, cf_by_tech, base_year, res_energy_sum_1):
                """
                Calculate the base residual energy and its distribution for a specific technology based on the base capacity and capacity factor.

                Args:
                - tech (str): The technology for which the calculations are done.
                - dict_base_caps (dict): Dictionary containing base capacities for technologies.
                - cf_by_tech (dict): Dictionary containing capacity factors for technologies.
                - base_year (int): The base year for the calculations.
                - res_energy_sum_1 (float): Sum of residual energy from other sources.

                Returns:
                - res_energy_base_1 (float): Calculated base residual energy for the technology.
                - energy_dist_1 (float): Energy distribution for the technology.
                - this_tech_base_cap (float): Base capacity of the technology.
                """
                this_tech_base_cap = dict_base_caps[tech][base_year]
                list_tech_cf_loc = cf_by_tech[tech]
                res_energy_base_1 = calculate_res_energy_base_and_update_cf(this_tech_base_cap, list_tech_cf_loc[0])
                energy_dist_1 = res_energy_base_1 / res_energy_sum_1
                return res_energy_base_1, energy_dist_1, this_tech_base_cap
            
            # Function #81
            def process_energy_distribution(df_param_related_4, tech_idx, time_vector, check_percent, energy_dist_1):
                """
                Process the energy distribution for a technology based on the known values and interpolate if necessary.

                Args:
                - df_param_related_4 (DataFrame): DataFrame containing energy distribution data.
                - tech_idx (int): Index of the technology in the DataFrame.
                - time_vector (list): List of years for which the calculations are performed.
                - check_percent (bool): Flag to check if the distribution needs to be interpolated.
                - energy_dist_1 (float): Initial energy distribution for the technology.

                Returns:
                - this_tech_dneeg_vals (list): List of processed energy distribution values for the technology.
                """
                this_tech_dneeg_df_param_related = df_param_related_4.iloc[tech_idx]
                this_tech_dneeg_known_vals_raw = []
                this_tech_dneeg_known_vals = []
                this_tech_dneeg_known_vals_count = 0

                for y in time_vector:
                    add_value = this_tech_dneeg_df_param_related[y]
                    this_tech_dneeg_known_vals_raw.append(add_value)
                    if not this_country == 'Uruguay':
                        if isinstance(add_value, (int, float, np.floating)):
                            if not math.isnan(add_value):
                                this_tech_dneeg_known_vals.append(add_value / 100)
                                this_tech_dneeg_known_vals_count += 1
                            else:
                                this_tech_dneeg_known_vals.append('')
                        else:
                            this_tech_dneeg_known_vals.append('')
                    else:
                        if str(y) == str(base_year):
                            this_tech_dneeg_known_vals.append(energy_dist_1)  # this had been zero before
                        elif type(add_value) is int or type(add_value) is float:
                            if math.isnan(add_value) is False:
                                this_tech_dneeg_known_vals.append(add_value/100)
                            else:
                                pass
                        else:
                            this_tech_dneeg_known_vals.append('')
                            
                if check_percent and add_value != 'rem':
                    this_tech_dneeg_vals = interpolation_to_end(time_vector, ini_simu_yr, this_tech_dneeg_known_vals, 'last', 'power')
                    if this_tech_dneeg_known_vals_count == len(this_tech_dneeg_vals):
                        this_tech_dneeg_vals = deepcopy(this_tech_dneeg_known_vals)
                else:
                    this_tech_dneeg_vals = [0 for _ in range(len(time_vector))]
                    store_percent_rem.update({tech:this_tech_dneeg_vals})
                store_percent.update({tech:this_tech_dneeg_vals})
                    
                return this_tech_dneeg_vals
            


            

            # Function #82
            def calculate_energy_distribution_remainder(store_percent_rem, store_percent, list_electric_sets_3, time_vector):
                """
                Calculate the energy distribution remainder for technologies and ensure the total distribution sums up to 100%.

                Args:
                - store_percent_rem (dict): Dictionary containing percent remainder values for technologies.
                - store_percent (dict): Dictionary containing percent values for technologies.
                - list_electric_sets_3 (list): List of electric set technologies.
                - time_vector (list): List of years for which the calculations are performed.

                Returns:
                - store_percent (dict): Updated dictionary with adjusted percent values ensuring the sum is 100%.
                """
                tech_rem = list(store_percent_rem.keys())[0]
                oneminus_rem_list = store_percent_rem[tech_rem]
                for tech in list_electric_sets_3:
                    if tech != tech_rem:
                        for y in range(len(time_vector)):
                            oneminus_rem_list[y] += store_percent[tech][y]
            
                for y in range(len(time_vector)):
                    store_percent[tech_rem][y] = 1 - oneminus_rem_list[y]
            
                return store_percent
            
            # Function #83
            def apply_adjustment_factor(store_res_energy_all, base_electric_prod, list_electric_sets_3, cf_by_tech, store_res_energy, time_vector):
                """
                Apply an adjustment factor to the residual energy, capacity factors, and store residual energy for all technologies.

                Args:
                - store_res_energy_all (list): List of stored residual energy for all technologies.
                - base_electric_prod (float): Base electricity production.
                - list_electric_sets_3 (list): List of electric set technologies.
                - cf_by_tech (dict): Dictionary containing capacity factors for technologies.
                - store_res_energy (dict): Dictionary containing stored residual energy for technologies.
                - time_vector (list): List of years for which the calculations are performed.

                Returns:
                - store_res_energy_all (list): Adjusted list of stored residual energy for all technologies.
                - cf_by_tech (dict): Adjusted dictionary containing capacity factors for technologies.
                - store_res_energy (dict): Adjusted dictionary containing stored residual energy for technologies.
                """
                adjustment_factor = base_electric_prod / store_res_energy_all[0]
                for y in range(len(time_vector)):
                    store_res_energy_all[y] *= adjustment_factor
                    for tech in list_electric_sets_3:
                        store_res_energy[tech][y] *= adjustment_factor
                        cf_by_tech[tech][y] *= adjustment_factor
            
                return store_res_energy_all, cf_by_tech, store_res_energy

            for tech in list_electric_sets_3:
                tech_idx = df_param_related_4['Tech'].tolist().index(tech)
                res_energy_base_1, energy_dist_1, this_tech_base_cap = calculate_energy_base_and_distribution(tech, dict_base_caps, cf_by_tech, base_year, res_energy_sum_1)
                check_percent = list(set(df_param_related_4['value']))[0] == 'percent'
                this_tech_dneeg_vals = process_energy_distribution(df_param_related_4, tech_idx, time_vector, check_percent, energy_dist_1)
                store_percent.update({tech: this_tech_dneeg_vals})

            # Ejecutar el cálculo del remanente de la distribución de energía
            store_percent = calculate_energy_distribution_remainder(store_percent_rem, store_percent, list_electric_sets_3, time_vector)
            
            # Almacenar el enfoque de porcentaje de BAU si es necesario
            if 'BAU' in this_scen:
                store_percent_BAU.update({this_country: deepcopy(store_percent)})
            
            # Aplicar el factor de ajuste
            store_res_energy_all, cf_by_tech, store_res_energy = apply_adjustment_factor(store_res_energy_all, base_electric_prod, list_electric_sets_3, cf_by_tech, store_res_energy, time_vector)
            
            # ...here we need to iterate across the list of technologies:
            fuel_use_electricity = {}  # crucial outputs
            externalities_globalwarming_electricity = {}  # crucial output
            externalities_localpollution_electricity = {}  # crucial output
            emissions_electricity = {}  # crucial output
            total_capacity = {}  # crucial output
            residual_capacity = {}  # crucial output
            new_capacity = {}  # crucial output

            # ...capacity disaggregation:
            cap_new_unplanned = {}
            cap_new_planned = {}
            cap_phase_out = {}

            # ...debugging dictionaries:
            ele_prod_share = {}
            ele_endogenous = {}
            cap_accum = {}

            total_production = {}  # crucial output
            new_production = {}
            capex = {}  # crucial output
            fopex = {}  # crucial output
            vopex = {}  # crucial output
            gcc = {}  # crucial output

            # Create dictionaries to store data from printing
            idict_u_capex = {}
            idict_u_fopex = {}
            idict_u_vopex = {}
            idict_u_gcc = {}
            idict_cau = {}
            idict_net_cap_factor = {}
            idict_hr = {}
            idict_oplife = {}

            # ...create a variable to represent lock-in decisions
            accum_cap_energy_vector = [0 for y in range(len(time_vector))]

            # 3rd power sector loop spotting conditions of surpassing capacity potential
            # ...we copy some of the elements of the 4th power sector loop
            # ...the unique sets with restriction:
            restriction_sets = list(set(df4_caps_rest['Set'].tolist()))

            # ...define the "adjustment_fraction" to recalculate the production shares
            adjustment_fraction = {}

            unit_gen_cost_dict = {}
            
            
            # Function #84
            def process_forced_new_capacity(df, tech_idx_8, time_vector):
                """
                Process and calculate forced new capacity for a specific technology.

                Forced new capacity refers to the capacity that needs to be added due to external factors or policy decisions.

                Args:
                - df (DataFrame): DataFrame containing capacity data.
                - tech_idx_8 (int): Index of the technology in the DataFrame.
                - time_vector (list): List of years for which the calculations are performed.

                Returns:
                - forced_new_cap (list): List of forced new capacities for each year.
                """
                forced_new_cap = [0 for _ in time_vector]
                if tech_idx_8 != '':
                    for y, year in enumerate(time_vector):
                        forced_new_cap[y] = df.loc[tech_idx_8, year] / 1000
                return forced_new_cap
            
            # Function #85
            def process_phase_out_capacity(df, tech_idx_9, time_vector):
                """
                Process and calculate phase-out capacity for a specific technology.

                Phase-out capacity refers to the capacity that is retired or phased out over time.

                Args:
                - df (DataFrame): DataFrame containing phase-out data for technologies.
                - tech_idx_9 (int): Index of the technology in the DataFrame.
                - time_vector (list): List of years for which the calculations are performed.

                Returns:
                - phase_out_cap (list): List of phase-out capacities for each year.
                """
                phase_out_cap = [0 for _ in time_vector]
                if tech_idx_9 != '':
                    for y, year in enumerate(time_vector):
                        phase_out_cap[y] = df.loc[tech_idx_9, year] / 1000
                return phase_out_cap
            
            # Function #86
            def extract_capacity_restriction(df, tech, country):
                """
                Extract capacity restriction for a specific technology from a DataFrame.

                Capacity restriction refers to the maximum allowed capacity for a technology due to regulatory or other constraints.

                Args:
                - df (DataFrame): DataFrame containing capacity restriction data.
                - tech (str): The technology for which capacity restriction is extracted.
                - country (str): The country for which capacity restriction is extracted.

                Returns:
                - float: Capacity restriction value for the technology in GW, or a very high number if no restriction is found.
                """
                mask_restriction = ((df['Set'] == tech) & (df['Country'] == country))
                restriction_value_df = df.loc[mask_restriction].reset_index(drop=True)
            
                if not restriction_value_df.empty:
                    return restriction_value_df['Restriction (MW)'].iloc[0] / 1000
                return 999999999
            
            # Function #87
            def calculate_parameter(df, tech, time_vector, parameter):
                """
                Calculate a specific parameter for a given technology over time.

                This function calculates the values of a specific parameter (e.g., CAU, Variable FOM, Heat Rate) for a technology, considering the projection method and unit multiplier.

                Args:
                - df (DataFrame): DataFrame containing the power techs data.
                - tech (str): The technology name.
                - time_vector (list): List of years for calculation.
                - parameter (str): The parameter to calculate (e.g., 'CAU', 'Variable FOM', 'Heat Rate').

                Returns:
                - result_list (list): List of calculated values for the parameter for each year.
                """
                mask = (df['Parameter'] == parameter) & (df['Tech'] == tech)
                df_filtered = df.loc[mask]
                
                result_list = []
                
                if not df_filtered.empty:
                    df_filtered = df_filtered.iloc[0]
                    unit_multiplier = df_filtered['Unit multiplier']
                    projection = df_filtered['Projection']
                
                    for year in time_vector:
                        if projection == 'user_defined':
                            value = df_filtered[year] * unit_multiplier
                        elif projection == 'flat':
                            value = df_filtered[time_vector[0]] * unit_multiplier
                        result_list.append(value)
                else:
                    result_list = [0 for _ in time_vector]
                
                return result_list

            # Function #88            
            def calculate_accumulated_unplanned_energy(cau, cf, accum_cap, y):
                """
                Calculate the energy supplied by accumulated unplanned capacity for a specific technology and year.

                Args:
                - cau (list): List of capacity utilization factors for the technology.
                - cf (list): List of capacity factors for the technology.
                - accum_cap (list): List of accumulated capacities for the technology.
                - y (int): The index representing the current year in the time_vector list.

                Returns:
                - (float): The energy supplied by accumulated unplanned capacity for the specific year.
                """
                if y != 0:
                    return cau[y] * cf[y] * accum_cap[y - 1]
                return 0
            
            # Function #89
            def calculate_new_cap(new_cap_unplanned):
                """
                Calculate new capacity by ensuring that unplanned capacities are non-negative.

                Args:
                - new_cap_unplanned (float): The calculated unplanned new capacity for a specific year.

                Returns:
                - (float): The corrected new capacity ensuring non-negative values.
                """
                if new_cap_unplanned < 0:
                    new_cap_unplanned = 0
                return new_cap_unplanned
            
            # Function #90
            def estimate_new_required_energy(demand_to_supply, forced_energy_all, res_energy_all, y):
                """
                Estimate the new required energy for a specific year by considering the demand, forced energy, and residual energy.

                Args:
                - demand_to_supply (list): List of energy demands for each year.
                - forced_energy_all (list): List of energy supplied by forced capacities for each year.
                - res_energy_all (list): List of residual energies for each year.
                - y (int): The index representing the current year in the time_vector list.

                Returns:
                - new_req_energy (float): The new required energy for the specific year.
                - (bool): A boolean indicating if the new required energy is negative.
                """
                new_req_energy = demand_to_supply[y] - forced_energy_all[y] - res_energy_all[y]
                new_req_energy = calculate_new_cap(new_req_energy)
                return new_req_energy, new_req_energy < 0
            
            # Function #91
            def calculate_new_energy_assignment(y, tech, this_tech_base_cap, list_tech_cf, res_energy_sum_1, new_req_energy, store_percent, res_energy_sum):
                """
                Calculate the new energy assignment for a specific technology based on the projection method.

                Args:
                - y (int): The index representing the current year in the time_vector list.
                - tech (str): The technology for which energy assignment is calculated.
                - this_tech_base_cap (float): The base capacity of the technology.
                - list_tech_cf (list): List of capacity factors for the technology.
                - res_energy_sum_1 (float): Sum of residual energy from other sources.
                - new_req_energy (float): The new required energy calculated for the year.
                - store_percent (dict): Dictionary containing percent values for technologies.
                - res_energy_sum (float): Total sum of residual energy.

                Returns:
                - (float): The new energy assigned to the technology for the specific year.
                """
                if y == 0:
                    res_energy_base = calculate_res_energy_base_and_update_cf(this_tech_base_cap, list_tech_cf[0])
                else:
                    res_energy_base = calculate_res_energy_base_and_update_cf(this_tech_base_cap, list_tech_cf[0])
            
                energy_dist = res_energy_base / res_energy_sum  # Distribution of energy
                return new_req_energy * energy_dist if this_tech_dneeg_projection == 'keep_proportions' else new_req_energy * store_percent[tech][y]
            
            # Function #92
            def calculate_technology_production(y, res_energy, planned_energy, new_energy_assign, this_tech_total_prod):
                """
                Calculate total and new production of a technology for a specific year.

                Args:
                - y (int): The index representing the current year in the time_vector list.
                - res_energy (float): Residual energy for the technology.
                - planned_energy (float): Energy planned for the technology.
                - new_energy_assign (float): New energy assigned to the technology.
                - this_tech_total_prod (list): List of total production for the technology.

                Returns:
                - total_prod (float): Total production of the technology for the specific year.
                - (float): New production of the technology for the specific year.
                """
                total_prod = res_energy + planned_energy + new_energy_assign
                new_prod = total_prod - this_tech_total_prod[y - 1] if y != 0 else total_prod
                return total_prod, max(new_prod, 0)
            
            # Function #93
            def calculate_subtract_new_cap(y, this_tech_accum_new_cap_unplanned):
                """
                Calculate the amount to be subtracted for the new unplanned capacity for a specific year.

                Args:
                - y (int): The index representing the current year in the time_vector list.
                - this_tech_accum_new_cap_unplanned (list): List of accumulated new unplanned capacities for the technology.

                Returns:
                - (float): The amount to be subtracted from the new unplanned capacity.
                """
                if y == 0:
                    return 0
                return this_tech_accum_new_cap_unplanned[y-1]
            
            # Function #94
            def estimate_unplanned_capacity(y, new_energy_assign, list_tech_cau, list_tech_cf, subtract_new_cap):
                """
                Estimate the unplanned capacity for a specific technology and year based on the assigned energy and other parameters.

                Args:
                - y (int): The index representing the current year in the time_vector list.
                - new_energy_assign (float): New energy assigned to the technology.
                - list_tech_cau (list): List of capacity utilization factors for the technology.
                - list_tech_cf (list): List of capacity factors for the technology.
                - subtract_new_cap (float): The amount to be subtracted from the new unplanned capacity.

                Returns:
                - (float): The estimated unplanned capacity for the technology and year.
                """
                if list_tech_cau[y] * list_tech_cf[y] != 0:
                    return new_energy_assign / (list_tech_cau[y] * list_tech_cf[y]) - subtract_new_cap
                else:
                    print('division by zero', 'interpolate', 1)
                    sys.exit()
            
            # Function #95
            def update_residual_capacity(y, use_cap, this_tech_residual_cap, this_tech_phase_out_cap):
                """
                Update the residual capacity for a specific technology based on the year and used capacity.

                Args:
                - y (int): The index representing the current year in the time_vector list.
                - use_cap (float): The used capacity for the technology.
                - this_tech_residual_cap (list): List of residual capacities for the technology.
                - this_tech_phase_out_cap (list): List of phase-out capacities for the technology.

                Returns:
                - (float): The updated residual capacity for the technology and year.
                """
                if y == 0:
                    residual_cap = use_cap
                else:
                    residual_cap = this_tech_residual_cap[y-1] - this_tech_phase_out_cap[y]
                return residual_cap
            
            # Function #96
            def adjust_accumulated_capacities(y, new_cap, this_tech_accum_new_cap, new_cap_unplanned, this_tech_accum_new_cap_unplanned):
                """
                Adjust accumulated capacities for both planned and unplanned capacities for a specific year.

                Args:
                - y (int): The index representing the current year in the time_vector list.
                - new_cap (float): The new planned capacity for the technology.
                - this_tech_accum_new_cap (list): List of accumulated new planned capacities for the technology.
                - new_cap_unplanned (float): The new unplanned capacity for the technology.
                - this_tech_accum_new_cap_unplanned (list): List of accumulated new unplanned capacities for the technology.

                Returns:
                - accum_new_cap (float): The adjusted accumulated new planned capacity for the technology and year.
                - accum_new_cap_unplanned (float): The adjusted accumulated new unplanned capacity for the technology and year.
                """
                if y == 0:
                    accum_new_cap = new_cap
                    accum_new_cap_unplanned = new_cap_unplanned
                else:
                    accum_new_cap = new_cap + this_tech_accum_new_cap[y-1]
                    accum_new_cap_unplanned = new_cap_unplanned + this_tech_accum_new_cap_unplanned[y-1]
                return accum_new_cap, accum_new_cap_unplanned

            subtract_new_cap = 0
            
            for tech in list_electric_sets_3:
                adjustment_fraction.update({tech:1})
                
                # Extract index of the technology
                tech_idx = df_param_related_4['Tech'].tolist().index(tech)
                tech_idx_8, tech_idx_9 = get_technology_indices(df_param_related_8, df_param_related_9, tech)
            
                # Take details of the technolgy
                this_tech_details = df_param_related_4.iloc[tech_idx]
                this_tech_dneeg_apply_type = this_tech_details['apply_type']
                this_tech_dneeg_projection = this_tech_details['projection']
                this_tech_dneeg_value_type = this_tech_details['value']
            
                # Process new forced capacity and output phase
                this_tech_forced_new_cap = process_forced_new_capacity(df_param_related_8, tech_idx_8, time_vector)
                this_tech_phase_out_cap = process_phase_out_capacity(df_param_related_9, tech_idx_9, time_vector)


                # Extract restrictions of capacity
                restriction_value = extract_capacity_restriction(df4_caps_rest, tech, this_country)
                
                # Take list of capacity factors 
                list_tech_cf = cf_by_tech[tech]
                mask_this_tech = (d5_power_techs['Tech'] == tech)
                this_tech_df_cost_power_techs = d5_power_techs.loc[mask_this_tech]
            
                # Calculate annual unit utilization (CAU)
                list_tech_cau = calculate_parameter(d5_power_techs, tech, time_vector, 'CAU')

                # Calculate variable FOM to technology
                list_tech_vfom = calculate_parameter(this_tech_df_cost_power_techs, tech, time_vector, 'Variable FOM')

                # Calculate the heat rate for the technology
                list_tech_hr = calculate_parameter(this_tech_df_cost_power_techs, tech, time_vector, 'Heat Rate')

                # Store the unit generation cost:               
                unit_gen_cost = [a * b for a, b in zip(list_tech_hr, list_tech_vfom)]  # most likely in $/PJ
                unit_gen_cost_dict.update({tech: deepcopy(unit_gen_cost)})

                # ...define some intermediate variables; some are redefined later
                this_tech_accum_new_cap_unplanned = [0 for y in range(len(time_vector))]
                accum_cap_energy_vector = [0 for y in range(len(time_vector))]

                this_tech_accum_new_cap = [0 for y in range(len(time_vector))]
                this_tech_new_cap = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_new_cap_unplanned = [0 for y in range(len(time_vector))]  # disaggregated output (list, dict above)
                this_tech_new_cap_planned = [0 for y in range(len(time_vector))]  # disaggregated output (list, dict above)

                this_tech_residual_cap = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_total_cap = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_total_prod = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_new_prod = [0 for y in range(len(time_vector))]

                # ...apply the capacity estimation algorithm:
                if (this_tech_dneeg_projection == 'interpolate' and
                        this_tech_dneeg_value_type == 'percent' and
                        this_tech_dneeg_apply_type == 'all') or (
                        this_tech_dneeg_projection == 'keep_proportions'):

                    for y in range(len(time_vector)):

                        # Using the function for calculating energy from unplanned cumulative capacity
                        accum_energy = calculate_accumulated_unplanned_energy(list_tech_cau, list_tech_cf, this_tech_accum_new_cap_unplanned, y)
                        accum_cap_energy_vector[y] += accum_energy
                    
                        # Using the function to estimate the required energy
                        new_req_energy, is_under_zero = estimate_new_required_energy(electrical_demand_to_supply, forced_newcap_energy_all, store_res_energy_all, y)
                        if is_under_zero:
                            count_under_zero += 1
                            
                        # Variables of use and residual energy
                        use_cap = store_use_cap[tech][y]
                        res_energy = store_res_energy[tech][y]
                    
                        # Using the function to calculate the energy of the planned plants
                        planned_energy = forced_newcap_energy_by_tech[tech][y]

                        # Calculation of new energy allocation
                        res_energy_sum = res_energy_sum_1
                        new_energy_assign = calculate_new_energy_assignment(y, tech, this_tech_base_cap, list_tech_cf, res_energy_sum_1, new_req_energy, store_percent, res_energy_sum)
                    
                        # Calculation of the total and new technology production
                        this_tech_total_prod[y], this_tech_new_prod[y] = calculate_technology_production(y, res_energy, planned_energy, new_energy_assign, this_tech_total_prod)

                        # Calculation of the amount to subtract
                        subtract_new_cap = calculate_subtract_new_cap(y, this_tech_accum_new_cap_unplanned)
                    
                        # Estimation of unplanned capacity
                        new_cap_unplanned = estimate_unplanned_capacity(y, new_energy_assign, list_tech_cau, list_tech_cf, subtract_new_cap)
                    
                        # Save the calculated unplanned capacity
                        this_tech_new_cap_unplanned[y] = new_cap_unplanned

                        # Calculation of the new capacity
                        new_cap = calculate_new_cap(this_tech_new_cap_unplanned[y]) + forced_newcap_by_tech[tech][y]
 
                        # Update of the residual capacity
                        residual_cap = update_residual_capacity(y, use_cap, this_tech_residual_cap, this_tech_phase_out_cap)
                    
                        # Adjustment of the accumulated capacities
                        this_tech_accum_new_cap[y], this_tech_accum_new_cap_unplanned[y] = adjust_accumulated_capacities(y, new_cap, this_tech_accum_new_cap, this_tech_new_cap_unplanned[y], this_tech_accum_new_cap_unplanned)
                        
                        # Additional capacity updates
                        this_tech_total_cap[y] = residual_cap + this_tech_accum_new_cap[y]
                        this_tech_residual_cap[y] = residual_cap
                        this_tech_new_cap[y] += new_cap
                        this_tech_new_cap_unplanned[y] = deepcopy(this_tech_new_cap_unplanned[y])
                        this_tech_new_cap_planned[y] = deepcopy(forced_newcap_by_tech[tech][y])
                
                    # ...below we must assess if we need to recalculate the shares:
                    if this_tech_total_cap[-1] < restriction_value:
                        pass  # there is no need to do anything else
                    elif this_tech_total_cap[-1] <= 0:
                        pass  # there is no need to specify anything
                    else:  # we must re-estimate the shares
                        this_adjustment_fraction = \
                            (restriction_value-sum(this_tech_new_cap_planned)) / (this_tech_total_cap[-1] - this_tech_total_cap[0])
                        adjustment_fraction[tech] = this_adjustment_fraction
                
            # Function #97
            def calculate_new_shares(list_electric_sets_3, time_vector, adjustment_fraction, store_percent):
                """
                Calculate the new shares for each technology based on adjustment fractions and initial shares.
                This function computes the new shares for each technology in the electricity set, considering
                their initial shares and the adjustment fractions. It ensures that the shares are appropriately
                scaled based on the adjustment criteria, whether the fraction is less than, equal to, or greater
                than 1.

                Args:
                - list_electric_sets_3 (list of str): List containing the names of all technologies in the electricity set.
                - time_vector (list of int): List of years for which the calculation is done.
                - adjustment_fraction (dict of float): Dictionary containing the adjustment fraction for each technology.
                - store_percent (dict of float): Dictionary containing the initial stored percentage for each technology.

                Returns:
                - old_share (dict of list of float): Dictionary containing the old shares for each technology.
                - new_share (dict of list of float): Dictionary containing the new shares for each technology after adjustment.
                """
                sum_adjustment = [0 for y in range(len(time_vector))]
                cum_adjusted = [0 for y in range(len(time_vector))]
                old_share, new_share = {}, {}
            
                for tech in list_electric_sets_3:
                    old_share[tech] = [0 for y in range(len(time_vector))]
                    new_share[tech] = [0 for y in range(len(time_vector))]
            
                    for y in range(len(time_vector)):
                        if adjustment_fraction[tech] < 1:
                            new_share[tech][y] = adjustment_fraction[tech] * store_percent[tech][y]
                            cum_adjusted[y] += adjustment_fraction[tech] * store_percent[tech][y]
                        else:
                            sum_adjustment[y] += store_percent[tech][y]
                        old_share[tech][y] = store_percent[tech][y]
            
                # Additional loop for adjustment
                for tech in list_electric_sets_3:
                    for y in range(len(time_vector)):
                        if adjustment_fraction[tech] >= 1:
                            new_share[tech][y] = store_percent[tech][y] * ((1 - cum_adjusted[y]) / (sum_adjustment[y]))
            
                return old_share, new_share
            
            # Function #98
            def adjust_store_percent(list_electric_sets_3, time_vector, ini_simu_yr, old_share, new_share, store_percent):
                """
                Adjust the stored percentage for each technology based on new and old shares.
                This function modifies the stored percentages for each technology by comparing
                their old shares with the newly calculated shares. It ensures that the
                percentages reflect the adjusted shares, especially for years beyond the
                initial simulation year.

                Args:
                - list_electric_sets_3 (list of str): List containing the names of all technologies in the electricity set.
                - time_vector (list of int): List of years for which the adjustment is done.
                - ini_simu_yr (int): Initial year of the simulation.
                - old_share (dict of list of float): Dictionary containing the old shares for each technology.
                - new_share (dict of list of float): Dictionary containing the new shares for each technology after adjustment.
                - store_percent (dict of list of float): Dictionary containing the initial stored percentage for each technology.

                Returns:
                - store_percent (dict of list of float): Dictionary containing the adjusted stored percentage for each technology.
                """
                store_percent_freeze = deepcopy(store_percent)
                for tech in list_electric_sets_3:
                    for y in range(len(time_vector)):
                        if time_vector[y] > ini_simu_yr and old_share[tech][y] != 0:
                            store_percent[tech][y] *= new_share[tech][y] / old_share[tech][y]
                return store_percent

            # Function #99
            def sort_techs_by_cost(unit_gen_cost_dict):
                """
                Sort technologies based on their unit generation cost for each year.
                This function ranks the technologies for each year based on their
                unit generation cost. It provides a sorted list of technologies from
                the highest to the lowest cost, which can be useful for decision-making
                processes related to cost-efficiency.

                Args:
                - unit_gen_cost_dict (dict of list of float): Dictionary containing the unit generation cost for each technology.

                Returns:
                - yearly_sorted_tech_costs (dict of list of str): Dictionary with each year as the key and a list of technologies sorted by their unit generation cost as the value.
                """
                yearly_sorted_tech_costs = {}
                for year in range(len(next(iter(unit_gen_cost_dict.values())))):
                    sorted_techs = sorted(unit_gen_cost_dict.keys(), key=lambda tech: unit_gen_cost_dict[tech][year], reverse=True)
                    yearly_sorted_tech_costs[year] = sorted_techs
                return yearly_sorted_tech_costs
            
            # Calculate of new participations
            old_share, new_share = calculate_new_shares(list_electric_sets_3, time_vector, adjustment_fraction, store_percent)
            store_percent = adjust_store_percent(list_electric_sets_3, time_vector, ini_simu_yr, old_share, new_share, store_percent)
            
            # Technolgy clasification by cost
            yearly_sorted_tech_costs = sort_techs_by_cost(unit_gen_cost_dict)

            # 4th power sector loop
            list_electric_sets_3.sort()
            # thermal_filter_out = [    # This list depend of each model
            #     'PP_Thermal_Diesel',
            #     'PP_Thermal_Fuel oil',
            #     'PP_Thermal_Coal',
            #     'PP_Thermal_Crude',
            #     'PP_Thermal_Natural Gas']
            thermal_filter_out = [] # The user must to change
            list_electric_sets_3_shuffle = [item for item in list_electric_sets_3 if item not in thermal_filter_out]
            list_electric_sets_3_shuffle += thermal_filter_out

            # Remove the specified item
            item_to_move = 'PP_PV Utility+Battery_Solar'
            list_electric_sets_3_shuffle.remove(item_to_move)

            # Append it to the end
            list_electric_sets_3_shuffle.append(item_to_move)
            list_electric_sets_3_shuffle.sort()
            
            thermal_reductions_store = {}
            thermal_reductions_order = {}
            
            # Function #100
            def extract_technical_characteristics(tech, d5_power_techs):
                """
                Extract technical characteristics for a specific technology from a data source.
                This function filters the input data based on the specified technology and
                returns the filtered data. It is primarily used to gather technical parameters
                or properties related to a specific technology from a larger dataset.

                Args:
                - tech (str): The technology for which technical characteristics are to be extracted.
                - d5_power_techs (DataFrame): DataFrame containing information about various power technologies.

                Returns:
                - this_tech_df_cost_power_techs (DataFrame): A DataFrame containing the technical characteristics of the specified technology.
                """
                mask_this_tech = (d5_power_techs['Tech'] == tech)
                this_tech_df_cost_power_techs = d5_power_techs.loc[mask_this_tech]
                return this_tech_df_cost_power_techs

            # Function #101
            def calculate_capacity_changes(tech_idx, df_param, time_vector):
                """
                Calculate changes in capacity for a specific technology based on input parameters.
                This function computes the new capacity for a given technology over time. It checks
                if the index for the technology is valid and then iterates over the time vector to
                calculate new capacity values.

                Args:
                - tech_idx (int): Index of the technology in the DataFrame.
                - df_param (DataFrame): DataFrame containing the parameters relevant to capacity calculation.
                - time_vector (list of int): Time vector for the calculation, usually a list of years.

                Returns:
                - new_capacity (list): A list of new capacity values over time for the specified technology.
                """
                new_capacity = [0 for _ in time_vector]
                if tech_idx != '':
                    for y, year in enumerate(time_vector):
                        new_capacity[y] = df_param.loc[tech_idx, year] / 1000  # Convert to GW
                return new_capacity
            
            # Function #102
            def handle_thermal_reductions(tech_counter, new_req_energy, y, thermal_reductions_store):
                """
                Handle thermal reductions based on the current required energy and other input parameters.
                This function manages the thermal reductions for the energy system. It checks the position
                of the technology in the reduction sequence and accordingly updates the required energy
                and stores thermal reductions.

                Args:
                - tech_counter (int): Counter indicating the position of the technology in the reduction sequence.
                - new_req_energy (float): The current required energy before applying thermal reductions.
                - y (int): The current year index in the time vector.
                - thermal_reductions_store (dict): Dictionary storing the thermal reductions for each year.

                Returns:
                - new_req_energy (float): The updated value of required energy after considering thermal reductions.
                - thermal_reductions (float): The amount of thermal reductions for the current year.
                """
                if tech_counter == 1:
                    # Store the absolute value of new required energy
                    thermal_reductions = abs(deepcopy(new_req_energy))
                    thermal_reductions_store.update({y: deepcopy(thermal_reductions)})
                else:
                    # Retrieve stored thermal reductions, if available
                    thermal_reductions = thermal_reductions_store.get(y, 0)
            
                # Set new required energy to zero
                new_req_energy = 0
            
                return new_req_energy, thermal_reductions

            # Function #104
            def manage_thermal_reductions(this_scen, tech, y, wtb_tech_list, thermal_reductions, 
                                          max_cf_dict, cf_by_tech, ref_current_cap, list_tech_cau, 
                                          thermal_reductions_order, thermal_reductions_store,
                                          this_country):
                """
                Manage and apply thermal reductions based on the scenario, technology, and other input parameters.
                This function orchestrates the thermal reductions process by determining the sequence of reductions,
                applying reduction factors, and updating capacity factors. It is mainly used in scenarios where
                thermal reductions are a part of the energy transition strategy.

                Args:
                - this_scen (str): The current scenario being processed.
                - tech (str): The technology for which the reductions are being calculated.
                - y (int): The current year in the time vector.
                - wtb_tech_list (list of str): List of technologies sorted from worst to best in terms of variable costs.
                - thermal_reductions (float): Current thermal reductions needed.
                - max_cf_dict (dict): Dictionary containing the maximum capacity factors for reductions.
                - cf_by_tech (dict): Dictionary of capacity factors by technology.
                - ref_current_cap (float): Reference to the current capacity of the technology.
                - list_tech_cau (list of float): List of capacity utilization factors for the technology.
                - thermal_reductions_order (list of int): Order of technologies for thermal reductions.
                - thermal_reductions_store (dict): Dictionary storing thermal reductions for each year.

                Returns:
                - thermal_reductions (float): Updated thermal reductions after processing.
                - thermal_reductions_order (list): Updated order of technologies for thermal reductions.
                - cf_by_tech (dict): Updated capacity factors by technology after applying thermal reductions.
                """
                # Special handling for BAU scenario
                if this_scen == 'BAU':
                    thermal_reductions = 0
            
                # Check if the technology is in the list for potential reductions
                if tech in wtb_tech_list:
                    max_cf_for_reductions = max_cf_dict[tech]
                    list_tech_cf = deepcopy(cf_by_tech[tech])
                    list_th_tech_cf = deepcopy(cf_by_tech[tech])
                    if list_th_tech_cf[y] < max_cf_for_reductions:
                        thermal_reductions_order[y] += 1
                    
            
                enter_reduction_conditionals = False
                for idx, reduction_tech in enumerate(wtb_tech_list):
                    if tech == reduction_tech and thermal_reductions_order[y] == (idx + 1):
                        enter_reduction_conditionals = True
                        break
                    
                if enter_reduction_conditionals and list_tech_cf[y] > max_cf_for_reductions and thermal_reductions > 0:
                    # Proceed with reduction calculations
                    curr_energy_tech = list_tech_cau[y] * list_tech_cf[y] * ref_current_cap
                    min_energy_tech = list_tech_cau[y] * max_cf_for_reductions * ref_current_cap
                    max_red_tech = curr_energy_tech - min_energy_tech
            
                    if max_red_tech >= thermal_reductions:
                        new_cf = (curr_energy_tech - thermal_reductions) / (list_tech_cau[y] * ref_current_cap)
                        thermal_reductions = 0
                    else:
                        new_cf = max_cf_for_reductions
                        thermal_reductions -= max_red_tech
                        thermal_reductions_order[y] += 1
            
                    # Update capacity factors for the current and future years
                    for y_aux_cf in range(y, len(time_vector)):
                        list_tech_cf[y_aux_cf] = deepcopy(new_cf)
                        store_res_energy[tech][y_aux_cf] *= new_cf / cf_by_tech[tech][y_aux_cf]
                        forced_newcap_energy_by_tech[tech][y_aux_cf] *= new_cf / cf_by_tech[tech][y_aux_cf]
                    
                    cf_by_tech[tech] = deepcopy(list_tech_cf)
                    
                # Update thermal reductions store
                if y in list(thermal_reductions_store.keys()):
                    thermal_reductions_store[y] = deepcopy(thermal_reductions)
            
                return thermal_reductions, thermal_reductions_order, cf_by_tech

            # Function #105
            def calculate_res_energy_base(tech_base_cap, cf, year_index, base_year_cf, base_year_cf_sum):
                """
                Calculate the base residual energy for a specific technology at a given year index.
                This function computes the base residual energy, which is a crucial parameter in understanding
                the energy distribution among technologies. It uses the base capacity, capacity factor, and the
                cumulative capacity factor sum to perform this calculation.

                Args:
                - tech_base_cap (float): The base capacity of the technology in MW.
                - cf (list of float): List of capacity factors for the technology.
                - year_index (int): Index of the current year in the time vector.
                - base_year_cf (float): Capacity factor for the base year.
                - base_year_cf_sum (float): Cumulative sum of capacity factors for the base year.

                Returns:
                - res_energy_base (float): The base residual energy for the technology in the specified year.
                - res_energy_sum (float): Cumulative sum of the residual energy up to the specified year.
                """
                if year_index == 0:
                    #res_energy_base = tech_base_cap * 8760 * base_year_cf[0] / 1000  # MW to GW
                    res_energy_base = calculate_res_energy_base_and_update_cf(tech_base_cap, base_year_cf[0]) # MW to GW
                    res_energy_sum = base_year_cf_sum
                else:
                    #res_energy_base = tech_base_cap * 8760 * cf[year_index] / 1000  # MW to GW
                    res_energy_base =calculate_res_energy_base_and_update_cf(tech_base_cap, cf[year_index]) # MW to GW
                    res_energy_sum = base_year_cf_sum
            
                return res_energy_base, res_energy_sum
            
            # Function #106
            def assign_new_energy(req_energy, energy_dist, store_percent, tech, year_index):
                """
                Assign new energy to a technology for a given year based on the energy distribution or a stored percentage.
                This function determines how much new energy should be assigned to a particular technology in a specific year. 
                It either uses a direct energy distribution proportion or a stored percentage for the assignment.

                Args:
                - req_energy (float): The total required energy for the year.
                - energy_dist (float): The distribution of energy as calculated from the base residual energy.
                - store_percent (dict): A dictionary containing stored percentages for each technology.
                - tech (str): The current technology being processed.
                - year_index (int): The current year index in the time vector.

                Returns:
                - (float): The amount of new energy assigned to the technology for the year.
                """
                if energy_dist is not None:  # keep_proportions case
                    return req_energy * energy_dist
                else:  # other cases
                    return req_energy * store_percent[tech][year_index]

            # Function #107
            def update_cf_for_thermal_gas(tech, y, new_energy_assign, res_energy, planned_energy, cf_by_tech, cf_ngas_max):
                """
                Update the capacity factor for thermal natural gas technology based on new energy assignments.
                This function adjusts the capacity factor for a thermal natural gas technology considering the
                new energy assigned to it. It ensures the capacity factor does not exceed the maximum allowable
                value and updates the capacity factor for future years.

                Args:
                - tech (str): The technology (assumed to be thermal natural gas) being processed.
                - y (int): The year index for which capacity factors are being calculated.
                - new_energy_assign (float): The newly assigned energy for the technology.
                - res_energy (float): The existing residual energy for the technology.
                - planned_energy (float): The existing planned energy for the technology.
                - cf_by_tech (dict): Dictionary containing capacity factors by technology.
                - cf_ngas_max (float): Maximum allowable capacity factor for natural gas technology.

                Returns:
                - (list): Updated list of capacity factors for the technology across the time vector.
                """              
                cf_original = deepcopy(cf_by_tech[tech][y])
                cf_new_changed = deepcopy(cf_by_tech[tech][y])
        
                if (res_energy + planned_energy) > 0:
                    cf_new_changed *= (res_energy + planned_energy + new_energy_assign) / (res_energy + planned_energy)
                else:
                    cf_new_changed = deepcopy(cf_ngas_max)
        
                cf_new_changed_original = deepcopy(cf_new_changed)
                if cf_new_changed > cf_ngas_max:
                    cf_new_changed = deepcopy(cf_ngas_max)
        
                for y_aux_cf in range(y, len(time_vector)):
                    list_tech_cf[y_aux_cf] = deepcopy(cf_new_changed)
                    store_res_energy[tech][y_aux_cf] *= cf_new_changed / cf_by_tech[tech][y_aux_cf]
                    forced_newcap_energy_by_tech[tech][y_aux_cf] *= cf_new_changed / cf_by_tech[tech][y_aux_cf]
                cf_by_tech[tech] = deepcopy(list_tech_cf)
                
                return cf_by_tech[tech]            

            # Function #108
            def calculate_renewable_energy_delta(reno_target, reno_est, demand_to_supply, all_gen_target, y):
                """
                Calculate the delta of renewable energy based on the set targets and the current estimates.
                This function computes the additional renewable energy required to meet the set targets,
                considering the current estimates and the overall energy demand.

                Args:
                - reno_target (list of float): List containing renewable energy targets for each year.
                - reno_est (list of float): List containing current renewable energy estimates for each year.
                - demand_to_supply (list of float): List of electricity demand values for each year.
                - all_gen_target (list of float): List of total generation target values for each year.
                - y (int): Current year index.

                Returns:
                - reno_ene_delta_demand_based (float): The additional renewable energy required based on demand.
                - reno_ene_delta (float): The additional renewable energy required based on total production.
                """
                reno_ene_delta_demand_based = ((reno_target[y] - reno_est[y]) / 100) * demand_to_supply[y]
                reno_ene_delta = ((reno_target[y] - reno_est[y]) / 100) * all_gen_target[y]
                return reno_ene_delta_demand_based, reno_ene_delta
            
            # Function #109
            def update_energy_variables(this_tech_total_prod, reno_ene_delta, new_energy_assign, y, reno_ene_delta_add):
                """
                Update key energy variables that control the capacity expansion based on renewable energy delta.
                This function adjusts the total production and the new energy assignment for a technology based
                on the additional renewable energy required to meet set targets. It is crucial for maintaining
                the balance between demand, supply, and renewable energy objectives.

                Args:
                - this_tech_total_prod (list of float): List of total production values for the technology.
                - reno_ene_delta (float): The additional renewable energy required.
                - new_energy_assign (float): The current new energy assignment for the technology.
                - y (int): The current year index.
                - reno_ene_delta_add (list of float): List to store the additional renewable energy for each year.

                Returns:
                - this_tech_total_prod (list of float): Updated list of total production values for the technology.
                - new_energy_assign (float): Updated new energy assignment for the technology.
                - reno_ene_delta_add (list of float): Updated list storing additional renewable energy for each year.
                """
                for y_aux_cf in range(y, len(this_tech_total_prod)):
                    this_tech_total_prod[y_aux_cf] += reno_ene_delta
                    reno_ene_delta_add[y_aux_cf] += reno_ene_delta
                new_energy_assign += reno_ene_delta
                return this_tech_total_prod, new_energy_assign, reno_ene_delta_add

            # Function #110
            def update_thermal_reduction_order(thermal_tech, year, max_cf_dict, cf_by_tech, thermal_reductions_order):
                """
                Update the order of thermal reductions based on current capacity factors.
                This function adjusts the order in which thermal reductions should be
                applied by comparing the actual capacity factors of thermal technologies
                with their maximum allowed capacity factors. The order is increased if
                the current capacity factor is less than the maximum, indicating that
                reductions are still possible.

                Args:
                - thermal_tech (str): Name of the thermal technology.
                - year (int): Current year index.
                - max_cf_dict (dict): Dictionary containing the maximum capacity factors for thermal technologies.
                - cf_by_tech (dict): Dictionary containing the list of capacity factors for each technology.
                - thermal_reductions_order (list): List representing the order in which thermal reductions should be applied for each year.

                Returns:
                - thermal_reductions_order (list): Updated list representing the order in which thermal reductions should be applied for each year.
                """
                max_cf = max_cf_dict[thermal_tech]
                cf_list = deepcopy(cf_by_tech[thermal_tech])
                if cf_list[year] <= max_cf:
                    thermal_reductions_order[year] += 1
                return thermal_reductions_order
            
            # Function #111
            def apply_thermal_reduction(tech, year, max_cf_dict, cf_by_tech, thermal_reductions, total_capacity, total_production):
                """
                Apply thermal reduction to the specified technology based on the maximum allowed capacity factor.
                This function calculates the new capacity factor for the technology after applying the thermal
                reduction. It ensures that the reduction does not exceed the maximum allowed reduction and updates
                the total production of the technology accordingly.

                Args:
                - tech (str): Name of the technology.
                - year (int): Current year index.
                - max_cf_dict (dict): Dictionary containing the maximum capacity factors for thermal technologies.
                - cf_by_tech (dict): Dictionary containing the list of capacity factors for each technology.
                - thermal_reductions (float): The amount of thermal reduction to be applied.
                - total_capacity (dict): Dictionary containing the total capacity for each technology.
                - total_production (dict): Dictionary containing the total production for each technology.

                Returns:
                - thermal_reductions (float): The remaining amount of thermal reduction after applying the reduction.
                - cf_by_tech (dict): Updated dictionary containing the list of capacity factors for each technology.
                """
                max_cf = max_cf_dict[tech]
                cf_list = deepcopy(cf_by_tech[tech])
                curr_energy = cf_list[year] * total_capacity[tech][year]
                min_energy = max_cf * total_capacity[tech][year]
                max_reduction = curr_energy - min_energy
            
                if max_reduction >= thermal_reductions:
                    new_cf = (curr_energy - thermal_reductions) / total_capacity[tech][year]
                    thermal_reductions = 0
                else:
                    new_cf = max_cf
                    thermal_reductions -= max_reduction
            
                for y_aux in range(year, len(cf_by_tech[tech])):
                    cf_list[y_aux] = deepcopy(new_cf)
                    total_production[tech][y_aux] *= new_cf / cf_by_tech[tech][y_aux]
            
                cf_by_tech[tech] = deepcopy(cf_list)
                return thermal_reductions, cf_by_tech

            # Function #112
            def calculate_generation(list_tech, total_production):
                """
                Calculate renewable and total generation for each technology.
                This function aggregates the total and renewable generation for each
                technology and provides a verification mechanism to ensure that the
                calculations are correct. It distinguishes between renewable and
                non-renewable technologies based on the technology name.

                Args:
                - list_tech (list of str): List of all technologies.
                - total_production (dict): Dictionary containing the total production for each technology.

                Returns:
                - renewable_gen (list of float): List of renewable generation for each year.
                - total_gen (list of float): List of total generation for each year.
                - reno_gen_verify (list of float): List of verified renewable generation for each year.
                - all_gen_verify (list of float): List of verified total generation for each year.
                """
                renewable_gen = [0] * len(time_vector)
                total_gen = [0] * len(time_vector)
                for tech in list_tech:
                    if 'Solar' in suptech or 'Geo' in suptech or 'Sugar' in suptech or 'Wind' in suptech or 'Hydro' in suptech:
                        renewable_gen = [a + b for a, b in zip(renewable_gen, total_production[tech])]
                    total_gen = [a + b for a, b in zip(total_gen, total_production[tech])]
                reno_gen_verify = [a + b for a, b in zip(reno_gen_verify, this_tech_total_prod)]
                all_gen_verify = [a + b for a, b in zip(all_gen_verify, this_tech_total_prod)]

                return renewable_gen, total_gen, reno_gen_verify, all_gen_verify
            
            # Function #113
            def calculate_ratios(electrical_demand, this_tech_total_prod, reno_gen_verify, all_gen_verify):
                """
                Calculate the ratios of renewable generation to total generation and electrical demand.
                This function computes the renewable generation as a ratio to the total generation and
                the electrical demand for each year. It is used for validating and analyzing the
                proportion of renewable sources in the total energy mix.

                Args:
                - electrical_demand (list): List of electrical demand values for each year.
                - this_tech_total_prod (list): List of total production values for the current technology for each year.
                - reno_gen_verify (list): List of verified renewable generation values for each year.
                - all_gen_verify (list): List of verified total generation values for each year.

                Returns:
                - ratio_total_gen (list): List of ratios of renewable generation to total generation for each year.
                - ratio_electrical_demand (list): List of ratios of renewable generation to electrical demand for each year.
                - all_gen_verify (list): Updated list of verified total generation values for each year.
                """
                reno_gen_verify = [a + b for a, b in zip(reno_gen_verify, this_tech_total_prod)]
                all_gen_verify = [a + b for a, b in zip(all_gen_verify, this_tech_total_prod)]
                ratio_total_gen = [a / b for a, b in zip(reno_gen_verify, all_gen_verify)]
                ratio_electrical_demand = [a / b for a, b in zip(reno_gen_verify, electrical_demand)]
                return ratio_total_gen, ratio_electrical_demand, all_gen_verify
            
            # Function #114
            def print_verification_info(list_electric_sets_3_shuffle_rest, total_production, y):
                """
                Print verification information and perform checks for each technology.
                This function iterates through each technology and prints the total
                production for the current year. It is primarily used for debugging
                and verification purposes to ensure that the values being processed
                are within expected ranges.

                Args:
                - list_electric_sets_3_shuffle_rest (list): List of shuffled rest electric sets.
                - total_production (dict): Dictionary containing the total production for each technology.
                - y (int): Current year index.
                """
                for suptech2 in list_electric_sets_3_shuffle_rest:
                    print('>', suptech2, total_production[suptech2][y])

                print('Review elements that can be wrong')
                sys.exit()
            
            # Function #115
            def handle_negative_production(new_prod, y, tol, time_vector, scenario, tech, country, res_energy_change):
                """
                Handle cases of negative production, particularly during the transition phase of generation CAPEX.
                This function checks for negative production values and decides whether an adjustment is needed
                based on the scenario, technology, and year. It is crucial for maintaining data integrity and
                ensuring reasonable generation values.

                Args:
                - new_prod (float): New production value for the current year.
                - y (int): Current year index.
                - tol (float): Tolerance value for considering a production value as negative.
                - time_vector (list): List of years.
                - scenario (str): Current scenario being processed.
                - tech (str): Current technology being processed.
                - country (str): Current country being processed.
                - res_energy_change (float): Change in residual energy for the current year.

                Returns:
                - (bool): A boolean value indicating whether an adjustment is needed based on the conditions.
                """
                if abs(new_prod) < tol and new_prod < 0 and time_vector[y] <= 2023:
                    print('An expected negative in generation CAPEX occurred!')
                    print(scenario, tech, country)
                    return False  # No adjustment needed
                elif new_prod < -tol and time_vector[y] > 2023:
                    if res_energy_change < 0:
                        return False  # No adjustment needed
                    elif res_energy_change >= 0:
                        return True  # Adjustment needed
                return False
            
            # Function #116
            def adjust_capacity_factor(list_tech_cf, y, total_prod, total_cap, tech, list_tech_cau):
                """
                Adjust the capacity factor for non-renewable technologies.
                This function adjusts the capacity factor for the current
                year based on the total production, total capacity, and
                technology type. It ensures that the capacity factor remains
                within reasonable limits for non-renewable technologies.

                Args:
                - list_tech_cf (list): List of capacity factors for the technology.
                - y (int): Current year index.
                - total_prod (list): List of total production values for the technology.
                - total_cap (list): List of total capacities for the technology.
                - tech (str): Current technology being processed.
                - list_tech_cau (list): List of Capacity Utilization Factors (CAU) for the technology.

                Returns:
                - list_tech_cf (list): Updated list of capacity factors for the technology.
                - (float): The ratio of the updated capacity factor to the original capacity factor for the current year.
                """
                original_cf = deepcopy(list_tech_cf[y])
                if 'Solar' not in tech and 'Wind' not in tech:
                    list_tech_cf[y] = total_prod[y] / (total_cap[y - 1] * list_tech_cau[y - 1])
                return list_tech_cf, list_tech_cf[y] / original_cf

            # Function #117
            def calculate_unplanned_capacity(new_energy_assign, list_tech_cau, list_tech_cf, subtract_new_cap, y):
                """
                Calculate unplanned capacity for a given year based on the assigned new energy and other parameters.
                This function computes the unplanned capacity required to meet the new energy demands after considering
                the Capacity Utilization Factors (CAU) and Capacity Factors (CF) of the technology.

                Args:
                - new_energy_assign (float): Assigned new energy for the year.
                - list_tech_cau (list): List of Capacity Utilization Factors (CAU) for each year.
                - list_tech_cf (list): List of Capacity Factors (CF) for each year.
                - subtract_new_cap (float): Accumulated unplanned new capacity from the previous year.
                - y (int): Current year index.

                Returns:
                - (float): Calculated unplanned capacity for the year.
                """
                if list_tech_cau[y] * list_tech_cf[y] != 0:
                    return new_energy_assign / (list_tech_cau[y] * list_tech_cf[y]) - subtract_new_cap
                else:
                    print('division by zero', 'interpolate', 2)
                    sys.exit()

            # Function #118
            def update_residual_capacity_2(y, use_cap, this_tech_phase_out_cap, this_tech_residual_cap, this_tech_total_cap, this_tech_accum_new_cap):
                """
                Update the residual and total capacities for a given year, considering phase-out and new capacities.
                This function updates the values of residual capacity and total capacity for the current year based
                on used capacity, phase-out capacities, and new capacities. It is crucial for accurately tracking
                the capacity status of technologies over time.

                Args:
                - y (int): Current year index in the time vector.
                - use_cap (float): Used capacity at the start of the year.
                - this_tech_phase_out_cap (list): List of phase-out capacities for each year.
                - this_tech_residual_cap (list): List of residual capacities for each year.
                - this_tech_total_cap (list): List of total capacities for each year, updated in this function.
                - this_tech_accum_new_cap (list): List of accumulated new capacities up to the year.

                Returns:
                - residual_cap (float): Updated residual capacity for the year.
                - this_tech_total_cap (list): Updated list of total capacities for each year.
                - this_tech_residual_cap (list): Updated list of residual capacities for each year.
                """
                if y == 0:
                    residual_cap = use_cap
                    this_tech_total_cap[y] = use_cap
                    this_tech_residual_cap[y] = use_cap
                else:
                    residual_cap = this_tech_residual_cap[y-1] - this_tech_phase_out_cap[y]
                    this_tech_total_cap[y] += residual_cap + this_tech_accum_new_cap[y-1]
                    this_tech_residual_cap[y] = residual_cap
                return residual_cap, this_tech_total_cap, this_tech_residual_cap
            
            # Function #119
            def adjust_accumulated_new_cap(y, new_cap, new_cap_unplanned, this_tech_accum_new_cap, this_tech_accum_new_cap_unplanned):
                """
                Adjust accumulated new capacities for both planned and unplanned capacities for a given year.
                This function updates the accumulated new capacities, considering both the new capacities
                introduced in the current year and those accumulated from previous years. It is vital for
                tracking the cumulative effect of capacity addition over the years.

                Args:
                - y (int): Current year index.
                - new_cap (float): New capacity introduced in the current year.
                - new_cap_unplanned (float): New unplanned capacity introduced in the current year.
                - this_tech_accum_new_cap (list): List of accumulated new capacities for each year.
                - this_tech_accum_new_cap_unplanned (list): List of accumulated new unplanned capacities for each year.

                Returns:
                - this_tech_accum_new_cap (list): Updated list of accumulated new capacities for each year.
                - this_tech_accum_new_cap_unplanned (list): Updated list of accumulated new unplanned capacities for each year.
                """
                if y == 0:
                    this_tech_accum_new_cap[y] = new_cap
                    this_tech_accum_new_cap_unplanned[y] = new_cap_unplanned
                else:
                    this_tech_accum_new_cap[y] = new_cap + this_tech_accum_new_cap[y-1]
                    this_tech_accum_new_cap_unplanned[y] = new_cap_unplanned + this_tech_accum_new_cap_unplanned[y-1]
                return this_tech_accum_new_cap, this_tech_accum_new_cap_unplanned

            # Function #120
            def update_technology_data(tech, total_cap, residual_cap, new_cap, unplanned_cap, planned_cap, phase_out_cap, total_prod, new_prod, energy_dist, total_endo, accum_cap):
                """
                Update all capacity, production, and energy distribution variables for a given technology.
                This function consolidates and updates various technology-related parameters, including
                capacities, production, and energy distribution. It ensures that all relevant data for a
                specific technology is synchronized and up-to-date.

                Args:
                - tech (str): Technology name.
                - total_cap (list): Total capacity values for each year.
                - residual_cap (list): Residual capacity values for each year.
                - new_cap (list): New capacity values for each year.
                - unplanned_cap (list): Unplanned capacity values for each year.
                - planned_cap (list): Planned capacity values for each year.
                - phase_out_cap (list): Phase-out capacity values for each year.
                - total_prod (list): Total production values for each year.
                - new_prod (list): New production values for each year.
                - energy_dist (list): Energy distribution values for each year.
                - total_endo (list): Total endogenous values for each year.
                - accum_cap (list): Accumulated capacity values for each year.

                Returns:
                - (dict): Dictionary containing updated data for all parameters.
                """
                total_capacity[tech] = total_cap
                residual_capacity[tech] = residual_cap
                new_capacity[tech] = new_cap
                cap_new_unplanned[tech] = unplanned_cap
                cap_new_planned[tech] = planned_cap
                cap_phase_out[tech] = phase_out_cap
            
                total_production[tech] = total_prod
                new_production[tech] = new_prod
            
                ele_prod_share[tech] = energy_dist
                ele_endogenous[tech] = total_endo
                cap_accum[tech] = accum_cap

            # Function #121
            def update_technology_costs(tech, new_cap, total_cap, total_prod, capex_list, ffom_list, vfom_list, gcc_list, time_vector):
                """
                Update the CAPEX, FOPEX, VOPEX, and GCC for a given technology based on its capacities and production.
                This function calculates and updates the capital expenditure (CAPEX), fixed operating expenses (FOPEX),
                variable operating expenses (VOPEX), and grid connection costs (GCC) for a technology, considering its
                capacity and production values over the years.

                Args:
                - tech (str): Technology name.
                - new_cap (list): List of new capacities for each year.
                - total_cap (list): List of total capacities for each year.
                - total_prod (list): List of total productions for each year.
                - capex_list (list): List of CAPEX values for each year.
                - ffom_list (list): List of Fixed OPEX values for each year.
                - vfom_list (list): List of Variable OPEX values for each year.
                - gcc_list (list): List of Grid Connection Cost values for each year.
                - time_vector (list): List of years in the time vector.

                Returns:
                - this_tech_capex (list): Updated CAPEX list for the technology.
                - this_tech_fopex (list): Updated FOPEX list for the technology.
                - this_tech_vopex (list): Updated VOPEX list for the technology.
                - this_tech_gcc (list): Updated GCC list for the technology.
                """
                this_tech_capex = [new_cap[y] * capex_list[y] for y in range(len(time_vector))]
                this_tech_fopex = [total_cap[y] * ffom_list[y] for y in range(len(time_vector))]
                this_tech_vopex = [total_prod[y] * vfom_list[y] for y in range(len(time_vector))]
                this_tech_gcc = [new_cap[y] * gcc_list[y] for y in range(len(time_vector))]
            
                capex[tech] = this_tech_capex
                fopex[tech] = this_tech_fopex
                vopex[tech] = this_tech_vopex
                gcc[tech] = this_tech_gcc
            
                return this_tech_capex, this_tech_fopex, this_tech_vopex, this_tech_gcc

            # Function #122
            def calculate_emissions_and_update_vopex(tech, y, use_fuel, list_use_fuel, emissions_fuels_dict, df_param_related_7, time_vector, list_emissions, this_tech_vopex, idict_u_vopex):
                """
                Calculate emissions and update variable operating expenses (VOPEX) for a given technology and year based on fuel usage.
                This function computes the emissions resulting from the use of fuel by a technology and updates the VOPEX by
                incorporating the cost implications of fuel usage. It is crucial for assessing the environmental impact and
                operational costs of a technology.

                Args:
                - tech (str): Technology name.
                - y (int): Current year index.
                - use_fuel (str): Type of fuel used.
                - list_use_fuel (list): List of fuel use values for each year.
                - emissions_fuels_dict (dict): Dictionary containing emissions data for fuels.
                - df_param_related_7 (DataFrame): DataFrame containing fuel-related parameters.
                - time_vector (list): List of years in the time vector.
                - list_emissions (list): List of emissions values for each year.
                - this_tech_vopex (list): VOPEX list for the technology.
                - idict_u_vopex (dict): Dictionary of updated VOPEX.

                Returns:
                - list_emissions (list): Updated emissions list for each year.
                - this_tech_vopex (list): Updated VOPEX list for the technology.
                - idict_u_vopex (dict): Updated VOPEX dictionary.
                """
                add_value_emissions = list_use_fuel[y] * emissions_fuels_dict[use_fuel][y]
                list_emissions.append(add_value_emissions)
            
                fuel_idx_7 = df_param_related_7['Fuel'].tolist().index(use_fuel)
                this_tech_vopex[y] += \
                    deepcopy(list_use_fuel[y]*df_param_related_7.loc[fuel_idx_7, time_vector[y]])
                idict_u_vopex[tech][y] += \
                    deepcopy(df_param_related_7.loc[fuel_idx_7, time_vector[y]])
            
                return list_emissions, this_tech_vopex, idict_u_vopex

            # Function #123
            def calculate_externalities(y, use_fuel, list_use_fuel, externality_fuels_dict):
                """
                Calculate the externalities of global warming and local pollution for a given fuel and year.
                This function computes the external costs associated with the use of fuel, focusing on the
                impacts of global warming and local pollution. It is essential for understanding the broader
                environmental and societal implications of fuel usage.

                Args:
                - y (int): Current year index.
                - use_fuel (str): Type of fuel used.
                - list_use_fuel (list): List of fuel use values for each year.
                - externality_fuels_dict (dict): Dictionary containing externalities data for fuels.

                Returns:
                - add_value_globalwarming (float): Calculated externality for global warming for the year.
                - add_value_localpollution (float): Calculated externality for local pollution for the year.
                - list_externalities_globalwarming (list): Updated list of global warming externalities for each year.
                - list_externalities_localpollution (list): Updated list of local pollution externalities for each year.
                """
                add_value_globalwarming = list_use_fuel[y] * externality_fuels_dict[use_fuel]['Global warming']
                add_value_localpollution = list_use_fuel[y] * externality_fuels_dict[use_fuel]['Local pollution']
                list_externalities_globalwarming.append(add_value_globalwarming)
                list_externalities_localpollution.append(add_value_localpollution)
                return add_value_globalwarming, add_value_localpollution, list_externalities_globalwarming, list_externalities_localpollution

            # Function #124
            def update_emissions_and_externalities(tech, use_fuel, list_emissions, list_externalities_globalwarming, list_externalities_localpollution, emissions_electricity, externalities_globalwarming_electricity, externalities_localpollution_electricity):
                """
                Update dictionaries related to emissions and externalities for a given technology and fuel.

                This function ensures that the emissions and externalities data is accurately recorded and associated with the respective technology and fuel type. It aids in maintaining a structured and comprehensive record of environmental impacts.

                Args:
                - tech (str): Technology name.
                - use_fuel (str): Type of fuel used.
                - list_emissions (list): List of emissions values for each year.
                - list_externalities_globalwarming (list): List of global warming externalities for each year.
                - list_externalities_localpollution (list): List of local pollution externalities for each year.
                - emissions_electricity (dict): Dictionary to update with emissions data.
                - externalities_globalwarming_electricity (dict): Dictionary to update with global warming externalities data.
                - externalities_localpollution_electricity (dict): Dictionary to update with local pollution externalities data.
                """
                emissions_electricity.update({tech: {use_fuel: list_emissions}})
                externalities_globalwarming_electricity.update({tech: {use_fuel: list_externalities_globalwarming}})
                externalities_localpollution_electricity.update({tech: {use_fuel: list_externalities_localpollution}})
            
            tech_counter = 0
            for tech in list_electric_sets_3_shuffle:

                tech_idx = df_param_related_4['Tech'].tolist().index(tech)
                
                # Obtain technology indexes
                tech_idx_8, tech_idx_9 = get_technology_indices(tech, df_param_related_8, df_param_related_9)
            
                # Extract technical characteristics
                this_tech_df_cost_power_techs = extract_technical_characteristics(tech, d5_power_techs)
            
                tech_counter += 1

                # ...we can extract one parameter at a time:
                # CAPEX
                list_tech_capex = calculate_parameter(this_tech_df_cost_power_techs, tech, time_vector, 'CAPEX')

                # CAU
                list_tech_cau = calculate_parameter(this_tech_df_cost_power_techs, tech, time_vector, 'CAU')

                # Fixed FOM
                list_tech_ffom = calculate_parameter(this_tech_df_cost_power_techs, tech, time_vector, 'Fixed FOM')

                # Grid connection cost
                list_tech_gcc = calculate_parameter(this_tech_df_cost_power_techs, tech, time_vector, 'Grid connection cost')
                
                # Net capacity factor
                list_tech_cf = cf_by_tech[tech]
                
                # Operational life
                list_tech_ol = calculate_parameter(this_tech_df_cost_power_techs, tech, time_vector, 'Operational life')

                # Variable FOM
                list_tech_vfom = calculate_parameter(this_tech_df_cost_power_techs, tech, time_vector, 'Variable FOM')                

                # Heat Rate
                list_tech_hr = calculate_parameter(this_tech_df_cost_power_techs, tech, time_vector, 'Heat Rate')

                # ...we next need to incorporate the heat rate data into the variable opex //
                # hence, link the commodity to the technology fuel consumption

                # ...storing the power plant information for printing
                idict_u_capex.update({tech:deepcopy(list_tech_capex)})
                idict_u_fopex.update({tech:deepcopy(list_tech_ffom)})
                idict_u_vopex.update({tech:deepcopy(list_tech_vfom)})
                idict_u_gcc.update({tech:deepcopy(list_tech_gcc)})
                idict_cau.update({tech:deepcopy(list_tech_cau)})
                idict_net_cap_factor.update({tech:deepcopy(list_tech_cf)})
                idict_hr.update({tech:deepcopy(list_tech_hr)})
                idict_oplife.update({tech:deepcopy(list_tech_ol)})

                idict_net_cap_factor_by_scen_by_country[this_scen][this_country] = deepcopy(idict_net_cap_factor)

                # ...acting for "Distribution of new electrical energy generation" (_dneeg)
                this_tech_dneeg_df_param_related = df_param_related_4.iloc[tech_idx]
                this_tech_dneeg_apply_type = this_tech_dneeg_df_param_related['apply_type']
                this_tech_dneeg_projection = this_tech_dneeg_df_param_related['projection']
                this_tech_dneeg_value_type = this_tech_dneeg_df_param_related['value']
                this_tech_dneeg_known_vals_raw = []
                this_tech_dneeg_known_vals = []
                use_fuel = this_tech_dneeg_df_param_related['Fuel']

                this_tech_base_cap = dict_base_caps[tech][base_year]
                this_tech_accum_new_cap = [0 for y in range(len(time_vector))]
                this_tech_accum_new_cap_unplanned = [0 for y in range(len(time_vector))]
                this_tech_new_cap = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)

                this_tech_new_cap_unplanned = [0 for y in range(len(time_vector))]  # disaggregated output (list, dict above)
                this_tech_new_cap_planned = [0 for y in range(len(time_vector))]  # disaggregated output (list, dict above)
                this_tech_energy_dist = [0 for y in range(len(time_vector))]  # disaggregated output (list, dict above)

                this_tech_total_endo = [0 for y in range(len(time_vector))]  # disaggregated output (list, dict above)

                this_tech_residual_cap = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_total_cap = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_total_prod = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_new_prod = [0 for y in range(len(time_vector))]

                reno_ene_delta_add = [0] * len(time_vector)

                # This is equivalent to parameter 8:
                this_tech_forced_new_cap = calculate_capacity_changes(tech_idx_8, df_param_related_8, time_vector)
                # This is equivalent to parameter 9:
                this_tech_phase_out_cap = calculate_capacity_changes(tech_idx_9, df_param_related_9, time_vector)
                
                if (this_tech_dneeg_projection == 'interpolate' and
                        this_tech_dneeg_value_type == 'percent' and
                        this_tech_dneeg_apply_type == 'all') or (
                        this_tech_dneeg_projection == 'keep_proportions') or (
                        this_tech_dneeg_projection == 'user_defined'):

                    # REVIEW THIS
                    new_req_energy_list = []
                    new_req_energy_list_2 = []
                    new_ene_assign_list = []

                    for y in range(len(time_vector)):
                        # calculate the energy that the accumulated unplanned capacity supplies (this is actually unused)
                        if y != 0:
                            this_tech_accum_cap_energy = list_tech_cau[y] * list_tech_cf[y] * this_tech_accum_new_cap_unplanned[y-1]
                            accum_cap_energy_vector[y] += this_tech_accum_cap_energy
                        
                        # ...estimate the energy requirement
                        new_req_energy = electrical_demand_to_supply[y] - forced_newcap_energy_all[y] - store_res_energy_all[y]

                        if not this_country == 'Uruguay':
                            new_req_energy_list.append(new_req_energy)
    
                            # It is convenient to call the capacity here
                            if y > 0:
                                ref_current_cap = this_tech_total_cap[y-1]
                            else:
                                ref_current_cap = store_use_cap[tech][y]
                            
                            # Here we must add the capacity factor adjustment for thermal, such that the renewable target is met and no unnecessary capacity is further planned
                            # Also, considerations about renewable targets must be established
                            '''
                            There are basically 2 options to increase renewability:
                            - reduce thermal capacity factors
                            - increase renewable generation
                            
                            Furthermore, if more production than needed occurs, the thermal capacity factors can be reduced
                            '''
                     
                            # Establish technical capacity factor minimums to cope with reductions:
                            # Take the elements from the power plants:
                            max_cf_dict = {
                                'PP_Thermal_Diesel': 0.0001,
                                'PP_Thermal_Fuel oil': 0.000001,
                                'PP_Thermal_Coal': 0.0001,
                                'PP_Thermal_Crude': 0.0001,
                                'PP_Thermal_Natural Gas': 0.0001}
                        
                            # The sorted thermal technologies will come in handy
                            # Worst-to-best variable cost technologies:
                            wtb_tech_list = yearly_sorted_tech_costs[y][:5]
                            # We need to reshuffle this list so we remove nuclear and introduce Crude;
                            # this may have to change in the future when we update the model with costs.
                            wtb_tech_list = ['PP_Thermal_Diesel',
                                'PP_Thermal_Fuel oil', 'PP_Thermal_Coal',
                                'PP_Thermal_Crude', 'PP_Thermal_Natural Gas']

                            if tech_counter == 1:
                                thermal_reductions_order.update({y:1})

                        if new_req_energy < 0:
                            count_under_zero += 1
                            if not this_country == 'Uruguay':
                                new_req_energy, thermal_reductions = handle_thermal_reductions(tech_counter, new_req_energy, y, thermal_reductions_store)    
                                
                                thermal_reductions, thermal_reductions_order, cf_by_tech = manage_thermal_reductions(
                                    this_scen, tech, y, wtb_tech_list, thermal_reductions, max_cf_dict, cf_by_tech, 
                                    ref_current_cap, list_tech_cau, thermal_reductions_order, thermal_reductions_store,
                                    this_country)
                            else:
                                new_req_energy = 0
                        else:
                            if not this_country == 'Uruguay':
                                thermal_reductions_store.update({y:0})
                        
                        new_req_energy_list_2.append(new_req_energy)

                        '''
                        NOTE:
                        This means that the excess energy must be covered.
                        We design for an excess considering the additional planned capacity out of the excess.
                        '''                  
                        # NOTE: "store_res_energy_all" has the production of the residual capacity

                        # Instead of distribution to keep proportion, we need to proceed with the interpolation
                        # of the shares that this source will produce:
                        use_cap = store_use_cap[tech][y]  # cap with calibrated values
                        res_energy = store_res_energy[tech][y]  # energy that calibrated value produces (not considering new capacity)
                        
                        res_energy_change = 0
                        if y > 0:
                            res_energy_change = store_res_energy[tech][y] - store_res_energy[tech][y-1]
                    
                        planned_energy = forced_newcap_energy_by_tech[tech][y]  # energy from planned plants

                        
                        res_energy_base, res_energy_sum = calculate_res_energy_base(
                            this_tech_base_cap, 
                            cf_by_tech[tech], 
                            y, 
                            list_tech_cf, 
                            res_energy_sum_1
                        )
                        
                        energy_dist = None
                        if this_tech_dneeg_projection == 'keep_proportions':
                            energy_dist = res_energy_base / res_energy_sum
                
                        new_energy_assign = assign_new_energy(
                            new_req_energy, 
                            energy_dist, 
                            store_percent, 
                            tech, 
                            y
                        )
                        
                        new_ene_assign_list.append(new_energy_assign)

                        if tech_counter <= len(list_electric_sets_3_shuffle):
                            this_tech_total_prod[y] = deepcopy(res_energy + planned_energy + new_energy_assign)
                        
                        if this_scen == 'BAU':
                            cf_ngas_max = 0.8
                        else:
                            cf_ngas_max = 0.8  # more backup operation

                        if not this_country == 'Uruguay' and tech == 'PP_Thermal_Natural Gas' and new_energy_assign > 0 and cf_by_tech[tech][y] < cf_ngas_max and y > 0:
                            cf_by_tech['PP_Thermal_Natural Gas'] = update_cf_for_thermal_gas('PP_Thermal_Natural Gas', y, new_energy_assign, res_energy, planned_energy, cf_by_tech, cf_ngas_max)
                        
                        # Here the new energy assign of renewables can increase to meet the renewable targets:
                        if tech_counter == len(list_electric_sets_3_shuffle) and reno_targets_exist:

                            
                            list_electric_sets_3_shuffle_rest = list_electric_sets_3_shuffle[:-1]
                            # Here we must find out about the renewable generation
                            reno_gen = [0] * len(time_vector)
                            all_gen_reno_target = [0] * len(time_vector)                        
                            for suptech in list_electric_sets_3_shuffle_rest:
                                if 'Solar' in suptech or 'Geo' in suptech or 'Sugar' in suptech or 'Wind' in suptech or 'Hydro' in suptech:
                                    reno_gen = [a + b for a, b in zip(reno_gen, total_production[suptech])]
                                all_gen_reno_target = [a + b for a, b in zip(all_gen_reno_target, total_production[suptech])]
                        
                            reno_gen = [a + b for a, b in zip(reno_gen, this_tech_total_prod)]                          
                            reno_est = [100 * a / b if b != 0 else 0 for a, b in zip(reno_gen, all_gen_reno_target)]
                                
                            # We can compare the percentage of renewables
                            if isinstance(reno_target_list[y], (float, np.floating, int)):
                                if reno_est[y] < reno_target_list[y] and not np.isnan(reno_target_list[y]):
                                    print('We need to increase renewability! Replace thermal generation with renewable generation for THIS tech.')
                                    # First let's calculate the energy swap required, which is similar to "thermal_reductions":
                                    reno_ene_delta_demand_based, reno_ene_delta = \
                                        calculate_renewable_energy_delta(reno_target_list, reno_est,\
                                                                         electrical_demand_to_supply, all_gen_reno_target, y)                                    
                                    this_tech_total_prod, new_energy_assign, reno_ene_delta_add = \
                                        update_energy_variables(this_tech_total_prod, reno_ene_delta, new_energy_assign, y, reno_ene_delta_add)
                                    thermal_reductions_2 = deepcopy(reno_ene_delta)                             

                                    # Main logic
                                    for th_tech in wtb_tech_list:
                                        thermal_reductions_order = update_thermal_reduction_order(th_tech, y, max_cf_dict, cf_by_tech, thermal_reductions_order)
                                        enter_reduction_conditional = th_tech == wtb_tech_list[thermal_reductions_order[y] - 1]
                                        
                                        if enter_reduction_conditional and cf_by_tech[th_tech][y] > max_cf_dict[th_tech] and thermal_reductions_2 >= 0:
                                            thermal_reductions_2, cf_by_tech = apply_thermal_reduction(th_tech, y, max_cf_dict, cf_by_tech, thermal_reductions_2, total_capacity, total_production)
                                            
                                    print_reno_test = False
                                    if print_reno_test:
                                        print('Writing a test verifying the the renewability has been reached according to the RE Target parameter')
                                        # Here we must write a test to check the renewability of the system
                                        reno_gen_verify = [0] * len(time_vector)
                                        all_gen_verify = [0] * len(time_vector)
                                        for tech in list_electric_sets_3_shuffle_rest:
                                            if 'Solar' in suptech or 'Geo' in suptech or 'Sugar' in suptech or 'Wind' in suptech or 'Hydro' in suptech:
                                                renewable_gen = [a + b for a, b in zip(reno_gen_verify, total_production[suptech])]
                                            total_gen = [a + b for a, b in zip(all_gen_verify, total_production[suptech])]                                        
                                        ratio_all_gen, ratio_electrical_demand, all_gen_verify = \
                                            calculate_ratios(electrical_demand_to_supply, this_tech_total_prod, \
                                                             reno_gen_verify, all_gen_verify)
                                        print_verification_info(list_electric_sets_3_shuffle_rest, total_production, y)

                        if y != 0 and tech not in thermal_filter_out:
                            this_tech_new_prod[y] = this_tech_total_prod[y] - this_tech_total_prod[y - 1]
                            tol_min_neg_capex_pj_ele = 0.1
                            adjustment_needed = handle_negative_production(
                                this_tech_new_prod[y], y, tol_min_neg_capex_pj_ele, time_vector, 
                                this_scen, tech, this_country, res_energy_change
                            )
                            if adjustment_needed:
                                list_tech_cf, mult_factor_cf_reno = adjust_capacity_factor(
                                    list_tech_cf, y, this_tech_total_prod, this_tech_total_cap, tech, list_tech_cau
                                )
    
                        # Remembering how much should be subtracted
                        subtract_new_cap = calculate_subtract_new_cap(y, this_tech_accum_new_cap_unplanned)                            
                        
                        # Estimating unplanned capacity
                        new_cap_unplanned = calculate_unplanned_capacity(new_energy_assign, list_tech_cau, list_tech_cf, subtract_new_cap, y)

                        # Filter to avoid inconsistencies
                        new_cap = calculate_new_cap(new_cap_unplanned) + forced_newcap_by_tech[tech][y]
                        
                        # Update residual capacity
                        residual_cap, this_tech_total_cap, this_tech_residual_cap = update_residual_capacity_2(y, use_cap, this_tech_phase_out_cap, this_tech_residual_cap, this_tech_total_cap, this_tech_accum_new_cap)
                        
                        # Adjust accumulated new capacities
                        this_tech_accum_new_cap, this_tech_accum_new_cap_unplanned = \
                            adjust_accumulated_new_cap(y, new_cap, new_cap_unplanned, this_tech_accum_new_cap, this_tech_accum_new_cap_unplanned)
                                               
                        this_tech_new_cap[y] += new_cap
                        this_tech_total_cap[y] += new_cap
                    
                        this_tech_new_cap_unplanned[y] = deepcopy(new_cap_unplanned)
                        this_tech_new_cap_planned[y] = deepcopy(forced_newcap_by_tech[tech][y])
                        this_tech_energy_dist[y] = deepcopy(store_percent[tech][y])

                        # ...these are further debugging capacity/energy variables
                        for aux_y in range(y, len(time_vector)):
                            this_tech_total_endo[aux_y] = deepcopy(new_energy_assign)

                # ...we must now see the additional energy requirements of primary or secondary carriers because of total capacity
                if sum(list_tech_hr) != 0:  # means there is fuel a requirement:
                    list_use_fuel = []
                    for y in range(len(time_vector)):
                        add_value = \
                            this_tech_total_cap[y]*list_tech_cau[y]*list_tech_cf[y]*list_tech_hr[y]
                        list_use_fuel.append(add_value)
                    fuel_use_electricity.update({tech:{use_fuel:list_use_fuel}})
                else:
                    fuel_use_electricity.update({tech:'none'})
                
                # ...here we store the correspoding physical variables and compute debugging variables:
                update_technology_data(
                    tech, this_tech_total_cap, this_tech_residual_cap, this_tech_new_cap, this_tech_new_cap_unplanned, 
                    this_tech_new_cap_planned, this_tech_phase_out_cap, this_tech_total_prod, this_tech_new_prod, 
                    this_tech_energy_dist, this_tech_total_endo, this_tech_accum_new_cap)


                # ...here we compute the costs by multiplying capacities times unit costs:
                this_tech_capex, this_tech_fopex, this_tech_vopex, this_tech_gcc = update_technology_costs(
                    tech, 
                    this_tech_new_cap, 
                    this_tech_total_cap, 
                    this_tech_total_prod, 
                    list_tech_capex, 
                    list_tech_ffom, 
                    list_tech_vfom, 
                    list_tech_gcc, 
                    time_vector
                )

                # ...here we compute the externalities and emissions by multiplying fuel use times unit values:
                if sum(list_tech_hr) != 0:  # means there is fuel a requirement:
                    list_emissions = []
                    list_externalities_globalwarming = []
                    list_externalities_localpollution = []
                    for y in range(len(time_vector)):
                        if use_fuel in emissions_fuels_list:  # ...store emissions here
                            list_emissions, this_tech_vopex, idict_u_vopex = calculate_emissions_and_update_vopex(
                                tech, 
                                y, 
                                use_fuel, 
                                list_use_fuel, 
                                emissions_fuels_dict, 
                                df_param_related_7, 
                                time_vector, 
                                list_emissions, 
                                this_tech_vopex, 
                                idict_u_vopex
                            )                            

                        if use_fuel in externality_fuels_list:  # ...store externalities                        

                            add_value_globalwarming, add_value_localpollution, \
                                list_externalities_globalwarming, list_externalities_localpollution \
                                    = calculate_externalities(y, use_fuel, list_use_fuel, externality_fuels_dict)
                        
                    update_emissions_and_externalities(tech, use_fuel, list_emissions, list_externalities_globalwarming, list_externalities_localpollution, emissions_electricity, externalities_globalwarming_electricity, externalities_localpollution_electricity)

            # 3i) Store the transport calculations:
            '''
            *Use these variables:*
            dict_fleet_k
            dict_new_fleet_k
            dict_capex_out
            dict_fopex_out
            dict_vopex_out
            
            Remember to apply this: dict_eq_transport_fuels
            '''           
            if overwrite_transport_model:               
                # Update local country taxes and fiancial data
                dict_tax_out_t1 = dict_tax_out['Imports']
                dict_tax_out_t2 = dict_tax_out['IMESI_Venta']
                dict_tax_out_t3 = dict_tax_out['IVA_Venta']
                dict_tax_out_t4 = dict_tax_out['Patente']
                dict_tax_out_t5 = dict_tax_out['IMESI_Combust']
                dict_tax_out_t6 = dict_tax_out['IVA_Gasoil']
                dict_tax_out_t7 = dict_tax_out['IVA_Elec']
                dict_tax_out_t8 = dict_tax_out['Impuesto_Carbono']
                dict_tax_out_t9 = dict_tax_out['Otros_Gasoil']
                dict_tax_out_t10 = dict_tax_out['Tasa_Consular']
                dict_tax_out_t11 = dict_tax_out['Rodaje']
                
                dict_fuel_consump= dict_activities_out['Fuel Consumption']
                dict_conv_fuel_cts= dict_activities_out['Conversion Fuel Constant']

                dict_local_country[this_country].update({'Fleet': deepcopy(dict_fleet_k)})
                dict_local_country[this_country].update({'New Fleet': deepcopy(dict_new_fleet_k)})
                dict_local_country[this_country].update({'Fuel Consumption': deepcopy(dict_fuel_consump)})
                dict_local_country[this_country].update({'Conversion Fuel Constant': deepcopy(dict_conv_fuel_cts)})
                dict_local_country[this_country].update({'Transport CAPEX [$]': deepcopy(dict_capex_out)})
                dict_local_country[this_country].update({'Transport Fixed OPEX [$]': deepcopy(dict_fopex_out)})
                dict_local_country[this_country].update({'Transport Variable OPEX [$]': deepcopy(dict_vopex_out)})                
                dict_local_country[this_country].update({'Transport Tax Imports [$]': deepcopy(dict_tax_out_t1)})
                dict_local_country[this_country].update({'Transport Tax IMESI_Venta [$]': deepcopy(dict_tax_out_t2)})
                dict_local_country[this_country].update({'Transport Tax IVA_Venta [$]': deepcopy(dict_tax_out_t3)})                
                dict_local_country[this_country].update({'Transport Tax Patente [$]': deepcopy(dict_tax_out_t4)})
                dict_local_country[this_country].update({'Transport Tax IMESI_Combust [$]': deepcopy(dict_tax_out_t5)})
                dict_local_country[this_country].update({'Transport Tax IVA_Gasoil [$]': deepcopy(dict_tax_out_t6)})
                dict_local_country[this_country].update({'Transport Tax IVA_Elec [$]': deepcopy(dict_tax_out_t7)})
                dict_local_country[this_country].update({'Transport Tax IC [$]': deepcopy(dict_tax_out_t8)})
                dict_local_country[this_country].update({'Transport Tax Otros_Gasoil [$]': deepcopy(dict_tax_out_t9)})
                dict_local_country[this_country].update({'Transport Tax Tasa_Consular [$]': deepcopy(dict_tax_out_t10)})
                dict_local_country[this_country].update({'Transport Tax Rodaje [$]': deepcopy(dict_tax_out_t11)})

            # 3j) Store the data for printing:
            # Function #125
            def discount_all_costs(time_vector, r_rate, r_year, tech_list, cost_dicts, externality_fuel_lists, disc_externalities):
                """
                Apply discounting to all costs and externalities for each technology.
                This function calculates the present value of costs (CAPEX, FOPEX,
                VOPEX, and GCC) and externalities (global warming and local pollution)
                for each technology over the years in the time vector. It takes into account
                the discount rate to reflect the time value of money, ensuring that future costs
                and benefits are accurately represented in today's terms.

                Args:
                - time_vector (list(type)): List of years for which the costs and externalities are calculated.
                - r_rate (type): The annual discount rate used for calculating the present value of future costs.
                - r_year (type): The reference year for discounting, usually the current or the first year in the time vector.
                - tech_list (list(type)): List of technologies for which the costs and externalities will be discounted.
                - cost_dicts (dict(type)): Dictionary containing different types of costs (capex, fopex, vopex, gcc) with technology names as keys and cost values as lists for each year.
                - externality_fuel_lists (list(type)): List of fuels associated with externalities.
                - disc_externalities (dict(type)): Dictionary containing discounted externalities data, structured similarly to cost_dicts.

                Returns:
                - cost_dicts (dict(type)): Updated dictionary with discounted costs for each technology and cost type.
                - disc_externalities (dict(type)): Updated dictionary with discounted externalities for each technology and externality type.
                """
                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate / 100) ** (float(this_year) - r_year))
                    for tech in tech_list:
                        for cost_type in cost_dicts:
                            cost_dicts[cost_type][tech][y] *= disc_constant
                        for use_fuel in externality_fuel_lists:
                            for externality_type in ['globalwarming', 'localpollution']:
                                key = f'disc_externalities_{externality_type}_electricity'
                                try:
                                    disc_externalities[key][tech][use_fuel][y] *= disc_constant
                                except KeyError:
                                    pass  # In case the technology does not have an externality
                return cost_dicts, disc_externalities
            
            dict_local_country[this_country].update({'Electricity fuel use': deepcopy(fuel_use_electricity)})
            dict_local_country[this_country].update({'Global warming externalities in electricity': deepcopy(externalities_globalwarming_electricity)})
            dict_local_country[this_country].update({'Local pollution externalities in electricity': deepcopy(externalities_localpollution_electricity)})
            dict_local_country[this_country].update({'Emissions in electricity': deepcopy(emissions_electricity)})
            dict_local_country[this_country].update({'Electricity total capacity': deepcopy(total_capacity)})
            dict_local_country[this_country].update({'Electricity residual capacity': deepcopy(residual_capacity)})
            dict_local_country[this_country].update({'Electricity new capacity': deepcopy(new_capacity)})
            dict_local_country[this_country].update({'Electricity total production': deepcopy(total_production)})
            dict_local_country[this_country].update({'Electricity CAPEX': deepcopy(capex)})
            dict_local_country[this_country].update({'Electricity Fixed OPEX': deepcopy(fopex)})
            dict_local_country[this_country].update({'Electricity Variable OPEX': deepcopy(vopex)})
            dict_local_country[this_country].update({'Electricity Grid Connection Cost': deepcopy(gcc)})

            # ...disaggregate the new capacity:
            dict_local_country[this_country].update({'Electricity new capacity unplanned': deepcopy(cap_new_unplanned)})
            dict_local_country[this_country].update({'Electricity new capacity planned': deepcopy(cap_new_planned)})
            dict_local_country[this_country].update({'Electricity phase out capacity': deepcopy(cap_phase_out)})

            # ...let's store additional debugging variables per power plant (1):
            dict_local_country[this_country].update({'Electricity production share (unplanned)': deepcopy(ele_prod_share)})
            dict_local_country[this_country].update({'New energy assign': deepcopy(ele_endogenous)})
            dict_local_country[this_country].update({'Accumulated new capacity': deepcopy(cap_accum)})

            # ...let's store the "required energy" components:
            dict_local_country[this_country].update({'Electricity demand to supply': deepcopy(electrical_demand_to_supply)})
            dict_local_country[this_country].update({'Electricity planned supply': deepcopy(forced_newcap_energy_all)})

            # ...let's store additional debugging variables per power plant (2):
            dict_local_country[this_country].update({'Accumulated forced new capacity': deepcopy(accum_forced_newcap_by_tech)})
            dict_local_country[this_country].update({'Electricity planned supply per technology': deepcopy(forced_newcap_energy_by_tech)})
            dict_local_country[this_country].update({'Electricity residual supply': deepcopy(store_res_energy_all)})
            dict_local_country[this_country].update({'Electricity residual supply per tech': deepcopy(store_res_energy)})

            # *...here we need a supporting variable*:
            dict_local_country[this_country].update({'Electricity new production per tech': deepcopy(new_production)})

            # Update electricity data:
            # disaggregate the new capacity, 
            # let's store additional debugging variables per power plant (1)
            # let's store the "required energy" components
            # let's store additional debugging variables per power plant (2)
            # *...here we need a supporting variable*
            # *...here we add a non-energy cost*            
            # ...here we can execute the discount rate to 5 variables:
            '''
            'Global warming externalities in electricity'
            'Local pollution externalities in electricity'
            'Electricity CAPEX'
            'Electricity Fixed OPEX'
            'Electricity Variable OPEX'
            'Electricity Grid Connection Cost'
            '''
            disc_capex = deepcopy(capex)
            disc_fopex = deepcopy(fopex)
            disc_vopex = deepcopy(vopex)
            disc_gcc = deepcopy(gcc)
            disc_externalities_globalwarming_electricity = deepcopy(externalities_globalwarming_electricity)
            disc_externalities_localpollution_electricity = deepcopy(externalities_localpollution_electricity)

            cost_dicts={
                'disc_capex': deepcopy(capex),
                'disc_fopex': deepcopy(fopex),
                'disc_vopex': deepcopy(vopex),
                'disc_gcc': deepcopy(gcc)
            }
            disc_externalities={
                'disc_externalities_globalwarming_electricity': deepcopy(externalities_globalwarming_electricity),
                'disc_externalities_localpollution_electricity': deepcopy(externalities_localpollution_electricity)
            }
            
            cost_dicts, disc_externalities = discount_all_costs(time_vector, \
                r_rate, r_year, list_electric_sets_3, cost_dicts, \
                externality_fuels_list, disc_externalities)

            dict_local_country[this_country].update({'Electricity CAPEX (disc)': deepcopy(cost_dicts['disc_capex'])})
            dict_local_country[this_country].update({'Electricity Fixed OPEX (disc)': deepcopy(cost_dicts['disc_fopex'])})
            dict_local_country[this_country].update({'Electricity Variable OPEX (disc)': deepcopy(cost_dicts['disc_vopex'])})
            dict_local_country[this_country].update({'Electricity Grid Connection Cost (disc)': deepcopy(cost_dicts['disc_gcc'])})
            dict_local_country[this_country].update({'Global warming externalities in electricity (disc)': deepcopy(disc_externalities['disc_externalities_globalwarming_electricity'])})
            dict_local_country[this_country].update({'Local pollution externalities in electricity (disc)': deepcopy(disc_externalities['disc_externalities_localpollution_electricity'])})


            # At this point, we want to deal with the sum of local consumption of fossil
            # fuels for an assumption of exports change to the IROTE

            # for a_fuel in list_fuel:
            #     dict_energy_demand_by_fuel_sum[a_fuel] = [
            #         a + b for a, b in zip(
            #             dict_energy_demand_by_fuel_sum[a_fuel],
            #             dict_energy_demand_by_fuel[a_fuel])]

            # print('Record correct storage of overarching fuels')
            # sys.exit()

        dict_local_reg.update({this_reg: deepcopy(dict_local_country)})

    ###########################################################################
    # *Here we will implement the new IROTE
    if model_irote:
        for r in range(len(regions_list)):
            this_reg = regions_list[r]
    
            country_list = dict_regs_and_countries[this_reg]
            country_list.sort()
    
            # Add a filter to include countries with transport data only:
            country_list = [c for c in country_list if c in tr_list_app_countries_u]
    
            for c in range(len(country_list)):
                this_country = country_list[c]
                this_country_2 = dict_equiv_country_2[this_country]
                
                
                # At this point, we can add the IROTE estimation based on the previously stored dictionaries and lists
                '''
                IROTE_num = Renewable energy use + biomass
                IROTE_den = IROTE_num + Electricity fuel use + Non-electrical and non-biomass demand
                '''
                # Function #126
                def update_useful_pj_inputs(a_transf_key, add_trasf_val, dict_equiv_pp_fuel_rev, useful_pj_inputs_dict):
                    """
                    Update the useful_pj_inputs_dict with the appropriate transformation key.
                    This function is responsible for updating the dictionary containing useful
                    PJ inputs for the power plant fuel equivalence. It checks the transformation
                    key and adds the corresponding value to the dictionary, ensuring that each
                    fuel type is accounted for correctly.
    
                    Args:
                    - a_transf_key (type): The key representing the transformation in the input dictionary.
                    - add_trasf_val (type): The value to be added corresponding to the transformation key.
                    - dict_equiv_pp_fuel_rev (dict(type)): Dictionary for the equivalence of power plant fuels, mapping each fuel type to its equivalent category.
                    - useful_pj_inputs_dict (dict(type)): Dictionary to be updated with transformation values, where keys are fuel types and values are the corresponding PJ inputs.
                    """
                    if a_transf_key == 'Sugar cane and derivatives':
                        use_transf_key = dict_equiv_pp_fuel_rev['Renewable Thermal']
                        useful_pj_inputs_dict.update({use_transf_key: add_trasf_val})
                    else:
                        try:
                            use_transf_key = dict_equiv_pp_fuel_rev[a_transf_key]
                            useful_pj_inputs_dict.update({use_transf_key: add_trasf_val})
                        except KeyError:
                            print('No equivalence found for', a_transf_key)
    
                # Function #127
                def update_primary_inv_effic(useful_pj_inputs_dict, get_src_gen, primary_inv_effic_dict, get_src_prim_ene):
                    """
                    Update the primary inventory efficiency dictionary based on generation source and primary energy.
                    This function updates the primary inventory efficiency for each technology based on the provided
                    energy inputs and generation sources. It ensures that the efficiency is accurately calculated or
                    marked as an anomaly if the inputs are not consistent.
    
                    Args:
                    - useful_pj_inputs_dict (dict(type)): Dictionary with primary energy inputs.
                    - get_src_gen (type): The generation source value.
                    - primary_inv_effic_dict (dict(type)): Dictionary to be updated with inventory efficiencies, where keys are technology names and values are efficiency values.
                    - get_src_prim_ene (type): Primary energy source value for the generation.
    
                    Returns:
                    - primary_inv_effic_dict (dict(type)): Updated dictionary with inventory efficiencies for each technology.
                    """
                    if get_src_gen == 0 and get_src_prim_ene == 0:
                        primary_inv_effic_dict.update({itpk: 0})
                    elif get_src_gen == 0 and get_src_prim_ene != 0:
                        print('Anomaly', get_src_gen, round(get_src_prim_ene, 10))
                        # Consider handling the anomaly or exiting the script
                    else:
                        primary_inv_effic_dict.update({itpk: get_src_prim_ene / get_src_gen})
                    
                    return primary_inv_effic_dict
    
    
                # First, we need to calculate the renewable energy use by reverse-engineering the renewable production
                use_ref_eb = dict_database['EB'][this_reg][this_country_2]
                use_ref_eb_transf = use_ref_eb['Total transformation']['Power plants']
                all_eb_transf_keys = list(use_ref_eb_transf.keys())
                
                
    
                '''
                freeze for developing production expansion normalzied share
                dictionary
                '''
    
                # Unpack useful dictionaries previously stored
                total_production = dict_local_reg[this_reg][this_country]['Electricity total production']
                total_ele_prod = [0] * len(time_vector)
                for atech in list(total_production.keys()):
                    total_ele_prod = [a + b for a, b in zip(total_ele_prod, total_production[atech])]
                dict_energy_demand_by_fuel = dict_local_reg[this_reg][this_country]['Energy demand by fuel']
                fuel_use_electricity = dict_local_reg[this_reg][this_country]['Electricity fuel use']
                supply_dict = dict_database['EB'][this_reg][this_country_2]['Total supply']
                transf_dict = dict_database['EB'][this_reg][this_country_2]['Total transformation']
                dem_dict = dict_database['EB'][this_reg][this_country_2]['Final consumption']
                selfcon_dict = dict_database['EB'][this_reg][this_country_2]['Self-consumption']
                loss_dict = dict_database['EB'][this_reg][this_country_2]['Losses']
    
                # Calculate new exports
                export_dict_local = {}
                
                # Note: Deal with losses and self consumption that add to production + imports               
                
                # Find the relative change of consumption for specific fuel
                list_all_fuels = list(dict_energy_demand_by_fuel.keys())
                list_all_fuels = list_all_fuels[:list_all_fuels.index('Other secondary')+1]
                split_index = list_all_fuels.index('Electricity')
                list_primary_fuels = list_all_fuels[:split_index]
                list_secondary_fuels = list_all_fuels[split_index:]
                list_all_fuels_rev = list_secondary_fuels + list_primary_fuels
    
                # We need to find the fuel use electricy, but by fuel
                fuel_use_electricity_by_fuel = {}
                for atech in list(fuel_use_electricity.keys()):
                    if type(fuel_use_electricity[atech]) is not str:
                        for afuel in list(fuel_use_electricity[atech].keys()):
                            if afuel not in list(fuel_use_electricity_by_fuel.keys()):
                                fuel_use_electricity_by_fuel.update({afuel: fuel_use_electricity[atech][afuel]})
                            else:
                                fuel_use_electricity_by_fuel[afuel] = [a + b for a, b in zip(
                                    fuel_use_electricity_by_fuel[afuel], fuel_use_electricity[atech][afuel])]
                           
                # We need to have a preliminary assessment about transformations
                refinery_data = transf_dict['Refineries']
                refinery_ratio_and_sets = {'ratio':{}, 'sets':{}}
                for primfuel in list_primary_fuels:
                    prim_val_by = refinery_data[primfuel][str(time_vector[0])]
                    sec_val_by_sum = 0
                    set_list = []
                    for secfuel in list_secondary_fuels:
                        sec_val_by = refinery_data[secfuel][str(time_vector[0])]
                        if float(sec_val_by) > 0.0:
                            sec_val_by_sum += sec_val_by
                            set_list.append(secfuel)
                    if float(prim_val_by) != 0 and sec_val_by_sum > 0:
                        refinery_ratio_and_sets['ratio'].update({primfuel: -1*float(prim_val_by)/sec_val_by_sum})
                        refinery_ratio_and_sets['sets'].update({primfuel: set_list})
    
                # We need to store the total supply with the general production/import approach:
                total_supply_dict, total_prod_dict = {}, {}
    
                # Apply the exports to secondary fuels first, then primary fuels:
                for lif in list_all_fuels_rev:               
                    # Dealing with export projection by extracting the overall demand
                    list_lif = dict_energy_demand_by_fuel_sum[lif]
                    if list_lif[0] != 0:
                        # For the specific fuel, let's calculate exports and production
                        dict_energy_demand_by_fuel_sum_norm = [
                            v/list_lif[0] for v in list_lif]
    
                        export_fuel_by = supply_dict['Exports'][lif][str(time_vector[0])]
                        export_fuel_list = [-1 * v * export_fuel_by
                            for v in dict_energy_demand_by_fuel_sum_norm]
                    
                        if sum(dict_energy_demand_by_fuel_sum_norm)/len(dict_energy_demand_by_fuel_sum_norm) == 1:
                            print('please review this case if it happens')
                            sys.exit()
                    
                    else:
                        export_fuel_list = [0] * len(time_vector)
    
                    # Store value of exports
                    export_dict_local.update({lif: deepcopy(export_fuel_list)})
    
                    if lif in list(refinery_ratio_and_sets['ratio'].keys()):
                        iter_sec_sets = refinery_ratio_and_sets['sets'][lif]
                        sum_sec_set_list_exp = [0]*len(time_vector)
                        sum_sec_set_list_loc = [0]*len(time_vector)
                        for lif2 in iter_sec_sets:
                            sum_sec_set_list_exp = [a + b for a, b in zip(
                                export_dict_local[lif2], sum_sec_set_list_exp)]
                            sum_sec_set_list_loc = [a + b for a, b in zip(
                                dict_energy_demand_by_fuel[lif2], sum_sec_set_list_loc)] 
    
                        sum_sec_set_list = [a + b for a, b in zip(
                            sum_sec_set_list_exp, sum_sec_set_list_loc)]
                        
                        refinery_demand = [refinery_ratio_and_sets['ratio'][lif]*v
                            for v in sum_sec_set_list]
                    else:
                        refinery_demand = [0]*len(time_vector)
    
                    # Extracting the local demand:
                    list_demand = dict_energy_demand_by_fuel[lif]
    
                    # Extracting fuel demand for power plants:
                    try:
                        fuel_demand = fuel_use_electricity_by_fuel[lif]
                    except Exception:
                        fuel_demand = [0] * len(time_vector)
    
                    # Dealing with the production structure:
                    selfcon_by = selfcon_dict['none'][lif][str(time_vector[0])]
                    loss_by = loss_dict['none'][lif][str(time_vector[0])]
    
                    prod_fuel_by = supply_dict['Production'][lif][str(time_vector[0])]
                    import_fuel_by = supply_dict['Imports'][lif][str(time_vector[0])]
                    if prod_fuel_by + import_fuel_by == 0:
                        prod_fuel_ratio = 0
                        import_fuel_ratio = 0
                        loss_ratio = 0
                    else:
                        prod_fuel_ratio = prod_fuel_by / (prod_fuel_by + import_fuel_by)
                        import_fuel_ratio = import_fuel_by / (prod_fuel_by + import_fuel_by)
    
                        # We have the losses as transformation + demand as a share,
                        # but prefer the losses to production + imports ratio
                        loss_ratio = (selfcon_by + loss_by) / (prod_fuel_by + import_fuel_by)
    
                    '''
                    We have adopted the following approach:
                    total_supply = production + imports
                    demand + refinery + fuel_powerplant + loss = total_supply
                    loss / total_supply = loss_ratio
                    demand + refinery + fuel_powerplant = subtotal_supply
                    
                    => subtotal_supply/total_supply + loss_ratio = 1
                    => subtotal_supply/(1-loss_ratio) = total_supply
                    '''
    
                    subtotal_supply = [a + b + c for a, b, c in zip(
                        list_demand, refinery_demand, fuel_demand)]
                    
                    if loss_ratio < 1:
                        total_supply = [v/(1-loss_ratio) for v in subtotal_supply]
                    else:
                        total_supply = deepcopy(subtotal_supply)
                    total_prod = [v*prod_fuel_ratio for v in total_supply]
                    total_imports = [v*import_fuel_ratio for v in total_supply]
    
                    total_supply_dict.update({lif: deepcopy(total_supply)})
                    total_prod_dict.update({lif: deepcopy(total_prod)})
            
                
                
                
                useful_pj_inputs_dict = {}
                for a_transf_key in all_eb_transf_keys:
                    add_trasf_val = -1*float(use_ref_eb_transf[a_transf_key]['2021'])
                    if add_trasf_val > 0:
                        
                        update_useful_pj_inputs(
                            a_transf_key, 
                            add_trasf_val, 
                            dict_equiv_pp_fuel_rev, 
                            useful_pj_inputs_dict,
                        )
    
    
                primary_inv_effic_dict = {}
                renewable_key_list = []
                iter_tot_prod_keys = list(total_production.keys())
                store_res_energy_orig = dict_store_res_energy_orig[this_country_2]
                for itpk in iter_tot_prod_keys:
                    get_src_gen = store_res_energy_orig[itpk][0]
                    if itpk in list(useful_pj_inputs_dict.keys()):
                        get_src_prim_ene = useful_pj_inputs_dict[itpk]
                        primary_inv_effic_dict = update_primary_inv_effic(
                            useful_pj_inputs_dict, 
                            get_src_gen, 
                            primary_inv_effic_dict,
                            get_src_prim_ene
                        )                    
    
                    if 'Solar' in itpk:
                        # primary_inv_effic_dict.update({itpk: 1/0.15})
                        primary_inv_effic_dict.update({itpk: 1/0.8})
                        renewable_key_list.append(itpk)
                    if 'Wind' in itpk:
                        # primary_inv_effic_dict.update({itpk: 1/0.3})
                        primary_inv_effic_dict.update({itpk: 1/0.8})
                        renewable_key_list.append(itpk)
                    if 'Geothermal' in itpk and 'PP_Geothermal' not in list(useful_pj_inputs_dict.keys()):
                        primary_inv_effic_dict.update({itpk: 1/((1601.7292 * 0.0036)/50.9971403743161)})
                        renewable_key_list.append(itpk)
                    if 'Hydro' in itpk and 'PP_Hydro' not in list(useful_pj_inputs_dict.keys()):
                        primary_inv_effic_dict.update({itpk: 1})
                        renewable_key_list.append(itpk)
                    elif 'Hydro' in itpk and primary_inv_effic_dict[itpk] < 1:
                        primary_inv_effic_dict.update({itpk: 1})
                        renewable_key_list.append(itpk)
                    elif 'Hydro' in itpk:
                        renewable_key_list.append(itpk)                    
    
                    if itpk in renewable_key_list:
                        if primary_inv_effic_dict[itpk] < 1:
                            primary_inv_effic_dict[itpk] = 1
    
                
                # Now we need to iteratively calculate an indicator that has the renewable information for all years:
                # Function #128
                def calculate_renewable_generation(renewable_key_list, total_production, primary_inv_effic_dict):
                    """
                    Calculate the total renewable generation based on the individual contributions from different renewable sources.
                    This function aggregates the generation from various renewable sources to calculate the total renewable
                    generation. It takes into account the primary inventory efficiency of each renewable source to ensure that the
                    generation is accurately represented.
    
                    Args:
                    - renewable_key_list (list(type)): List of keys representing different renewable energy sources.
                    - total_production (dict(type)): Dictionary containing total production values for each renewable source.
                    - primary_inv_effic_dict (dict(type)): Dictionary containing primary inventory efficiency values for each renewable source.
    
                    Returns:
                    - reno_gen_list (list(type)): List containing the total renewable generation values for each year.
                    """
                    reno_gen_list = [0] * len(time_vector)
                    for rlk in range(len(renewable_key_list)):
                        this_total_production = total_production[renewable_key_list[rlk]]
                        this_primary_use = [primary_inv_effic_dict[renewable_key_list[rlk]] * val for val in this_total_production]
                        reno_gen_list = [a + b for a, b in zip(reno_gen_list, this_primary_use)]
                    return reno_gen_list
                
                # Function #129
                def calculate_other_biomass(dict_energy_demand_by_fuel):
                    """
                    Calculate the total energy from other biomass sources by aggregating individual contributions.
                    This function sums up the energy contributions from sources like 'Sugar cane and derivatives',
                    'Firewood', and 'Other primary sources' to calculate the total energy from other biomass sources.
    
                    Args:
                    - dict_energy_demand_by_fuel (dict(type)): Dictionary with energy demand values for each fuel type.
    
                    Returns:
                    - other_biomass (list(type)): List containing the total energy from other biomass sources for each year.
                    """
                    sug_cane_ene = dict_energy_demand_by_fuel['Sugar cane and derivatives']
                    firewood_ene = dict_energy_demand_by_fuel['Firewood']
                    otherprim_ene = dict_energy_demand_by_fuel['Other primary sources']
                    other_biomass = [a + b + c for a, b, c in zip(sug_cane_ene, firewood_ene, otherprim_ene)]
                    return other_biomass
                
                # Function #130
                def calculate_non_electricity_bio_demand(dict_energy_demand_by_fuel):
                    """
                    Calculate the total biomass demand for non-electricity purposes.
                    This function aggregates the demand from various biomass sources
                    that are not used for electricity generation, providing a total
                    biomass demand for non-electricity purposes.
    
                    Args:
                    - dict_energy_demand_by_fuel (dict(type)): Dictionary with energy demand values for each fuel type.
    
                    Returns:
                    - non_ele_bio_list (list(type)): List containing the total biomass demand for non-electricity purposes for each year.
                    """
                    non_ele_bio_list = [0] * len(time_vector)
                    for a_dedbf in dict_energy_demand_by_fuel:
                        if a_dedbf not in ['Electricity', 'Sugar cane and derivatives', 'Firewood', 'Other primary sources']:
                            non_ele_bio_list = [a + b for a, b in zip(non_ele_bio_list, dict_energy_demand_by_fuel[a_dedbf])]
                    return non_ele_bio_list
                
                # Function #131
                def calculate_fuel_use_pp(fuel_use_electricity):
                    """
                    Calculate the total fuel use for power plants.
                    This function aggregates the fuel use from various power
                    plants, providing a total fuel use for power plants.
    
                    Args:
                    - fuel_use_electricity (dict(type)): Dictionary with fuel use values for each power plant type.
    
                    Returns:
                    - fuel_use_pp_list (list(type)): List containing the total fuel use for power plants for each year.
                    """
                    fuel_use_pp_list = [0] * len(time_vector)
                    for a_fue in fuel_use_electricity:
                        if isinstance(fuel_use_electricity[a_fue], dict):
                            for a_fue_2 in fuel_use_electricity[a_fue]:
                                this_fue_list = fuel_use_electricity[a_fue][a_fue_2]
                                fuel_use_pp_list = [a + b for a, b in zip(fuel_use_pp_list, this_fue_list)]
                    return fuel_use_pp_list
                
                # Function #132
                def calculate_irote(reno_gen_list, other_biomass, non_ele_bio_list, fuel_use_pp_list):
                    """
                    Calculate the Indicator of Renewable and Other Types of Energy (IROTE).
                    This function calculates the IROTE by aggregating renewable generation,
                    other biomass energy, non-electricity biomass demand, and fuel use in
                    power plants. It provides a percentage representation of renewable and
                    other types of energy compared to the total energy use.
    
                    Args:
                    - reno_gen_list (list(type)): List of renewable generation values for each year.
                    - other_biomass (list(type)): List of other biomass energy values for each year.
                    - non_ele_bio_list (list(type)): List of non-electricity biomass demand values for each year.
                    - fuel_use_pp_list (list(type)): List of total fuel use for power plants for each year.
    
                    Returns:
                    - irote (list(type)): List containing the IROTE values for each year.
                    - num_irote (list(type)): Numerator values used in the calculation of IROTE.
                    - den_irote (list(type)): Denominator values used in the calculation of IROTE.
                    """
                    num_irote = [a + b for a, b in zip(reno_gen_list, other_biomass)]
                    den_irote = [a + b + c for a, b, c in zip(num_irote, non_ele_bio_list, fuel_use_pp_list)]
                    irote = [100 * a / b for a, b in zip(num_irote, den_irote)]
                    return irote, num_irote, den_irote
    
                    
                #reno_gen_list = [0] * len(time_vector)
                #
                reno_gen_list = calculate_renewable_generation(renewable_key_list, total_production, primary_inv_effic_dict)
                # We need to add biomass to the numerator, the one consumed on the demand side:
                other_biomass = calculate_other_biomass(dict_energy_demand_by_fuel)
                # Non-electricity and non-biomass demand:
                non_ele_bio_list = calculate_non_electricity_bio_demand(dict_energy_demand_by_fuel)
                # Add fuel use by power plants:
                fuel_use_pp_list = calculate_fuel_use_pp(fuel_use_electricity)
                
                
                # Extract the sum of primary supply not considered by renewables (pick fuel names):
                nonreno_primary_fuels_pick = ['Oil', 'Natural Gas', 'Coal']
                nonreno_second_fuels_pick = ['Electricity', 'Liquified Gas', 'Gasoline and alcohol',
                     'Kerose and jet fuel', 'Diesel', 'Fuel Oil', 'Coke', 'Gases']
    
                total_supply_primary = [0] * len(time_vector)
                total_supply_secondary = [0] * len(time_vector)
                total_production_secondary = [0] * len(time_vector)
                for pfp in nonreno_primary_fuels_pick:
                    total_supply_primary = [a + b for a, b in zip(
                        total_supply_primary, total_supply_dict[pfp])]
    
                for sfp in nonreno_second_fuels_pick:
                    total_supply_secondary = [a + b for a, b in zip(
                        total_supply_secondary, total_supply_dict[sfp])]
                    total_production_secondary = [a + b for a, b in zip(
                        total_production_secondary, total_prod_dict[sfp])]
    
                
                
                # Non-electricity and non-biomass demand:
                irote, num_irote, den_irote = calculate_irote(reno_gen_list, other_biomass, non_ele_bio_list, fuel_use_pp_list)
    
                irote_keys = ['IROTE', 'IROTE_NUM', 'IROTE_DEN']            
                irote_related_data = {
                    'IROTE': irote,
                    'IROTE_NUM': num_irote,
                    'IROTE_DEN': den_irote
                }            
                
                #Update data related electricity disc
                dict_local_reg[this_reg][this_country].update({'IROTE': deepcopy(irote_related_data['IROTE'])})
                dict_local_reg[this_reg][this_country].update({'IROTE_NUM': deepcopy(irote_related_data['IROTE_NUM'])})
                dict_local_reg[this_reg][this_country].update({'IROTE_DEN': deepcopy(irote_related_data['IROTE_DEN'])})

  
    ###########################################################################
    # *Here it is crucial to implement the exports as share of total LAC demand:

    # ...calculate the total natural gas demand
    lac_ng_dem = [0 for y in range(len(time_vector))]
    keys_2list_regions = list(dict_local_reg.keys())
    country_2_region = {}
    for ng_reg in keys_2list_regions:
        keys_2list_countries = list(dict_local_reg[ng_reg].keys())
        for ng_cntry in keys_2list_countries:
            query_dict = dict_local_reg[ng_reg][ng_cntry]
            local_ng_dem = []
            for y in range(len(time_vector)):
                add_val = \
                    query_dict['Energy demand by fuel']['Natural Gas'][y] + \
                    query_dict['Electricity fuel use']['PP_Thermal_Natural Gas']['Natural Gas'][y]  # this is a list
                local_ng_dem.append(add_val)

            for y in range(len(time_vector)):
                lac_ng_dem[y] += deepcopy(local_ng_dem[y])

            # ...store the dictionary below to quickly store the export values
            country_2_region.update({ng_cntry:ng_reg})

    # ...extract the exporting countries to LAC // assume *df_scen* is correct from previous loops
    df_scen_exports = filter_dataframe(df_scen, 'scenario_simple', scenario='% Exports for production', column='Parameter')
    df_scen_exports_countries = \
        df_scen_exports['Application_Countries'].tolist()
    # Add a filter to include countries with transport data only:
    df_scen_exports_countries = \
        [c for c in df_scen_exports_countries if c in tr_list_app_countries_u]

    df_scen_exports_pipeline = filter_dataframe(df_scen, 'scenario_simple', scenario='% Exports for production through pipeline', column='Parameter')
    df_scen_exports_countries_pipeline = \
        df_scen_exports_pipeline['Application_Countries'].tolist()
    # Add a filter to include countries with transport data only:
    df_scen_exports_countries_pipeline = \
        [c for c in df_scen_exports_countries_pipeline if c in tr_list_app_countries_u]
    
    # ...now we must extract all natural gas prices:
    df_ngas_prices = filter_dataframe(df_scen, 'two_columns_scenarios_special', scenario_2='Natural Gas',\
                        column='Parameter', column_2='Fuel')
    # ...now we must extract the quantitiy of natural gas exports to LAC!
    # In a loop, iterate across countries:
    for this_con in df_scen_exports_countries:
        df_scen_exports_select = filter_dataframe(df_scen_exports, 'scenario_simple', scenario=this_con, column='Application_Countries')
        df_scen_exports_select_pipe = filter_dataframe(df_scen_exports, 'scenario_simple', scenario=this_con, column='Parameter')

        exports_country = [0 for y in range(len(time_vector))]  # in PJ
        exports_country_pipe = [0 for y in range(len(time_vector))]
        exports_country_liq = [0 for y in range(len(time_vector))]
        exports_income = [0 for y in range(len(time_vector))]

        for y in range(len(time_vector)):
            this_year = int(time_vector[y])

            disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))

            export_price_pipe = df_ngas_prices.loc[0, int(time_vector[0])]  # FOR NOW ASSUME THE PRICE IS CONSTANT
            export_price_liq = df_ngas_prices.loc[1, int(time_vector[0])]

            # ...here we must calculate the natural gas exports for the country:
            exports_country[y] = \
                lac_ng_dem[y]*df_scen_exports_select.loc[0, this_year]/100

            if len(df_scen_exports_select_pipe.index.tolist()) != 0:
                # here we need to discriminate pipeline and non-pipeline elements
                q_ngas_pipe = \
                    exports_country[y]*df_scen_exports_select_pipe.loc[0, this_year]/100
            else:
                q_ngas_pipe = 0

            exports_country_pipe[y] = q_ngas_pipe
            exports_country_liq[y] = exports_country[y] - q_ngas_pipe

            exports_income[y] = \
                exports_country_pipe[y]*export_price_pipe + \
                exports_country_liq[y]*export_price_liq
            exports_income[y] *= disc_constant

        # ...now we must store the result and intermediary info to the dictionary
        this_reg = country_2_region[this_con]
        dict_local_reg[this_reg][this_con].update({'Natural Gas Exports (PJ)':deepcopy(exports_country)})
        dict_local_reg[this_reg][this_con].update({'Natural Gas Exports via Pipeline (PJ)':deepcopy(exports_country_pipe)})
        dict_local_reg[this_reg][this_con].update({'Natural Gas Exports Liquified (PJ)':deepcopy(exports_country_liq)})
        dict_local_reg[this_reg][this_con].update({'Natural Gas Exports Income (M USD)':deepcopy(exports_income)})  # only print the disocunted value

    ###########################################################################
    # *For fugitive emissions, we must use the "imports" sheet, with a similar approach as above

    # ...extract fugitive emissions of natural gas (exclusively):
    this_df_fugef_ngas = filter_dataframe(df4_ef, 'two_columns_scenarios', scenario='Production', scenario_2='Natural Gas',\
                        column='Apply', column_2='Fuel')
    fugef_ngas = this_df_fugef_ngas.iloc[0][per_first_yr]  # assume this is a constant

    this_df_fugef_ngas_2 = filter_dataframe(df4_ef, 'two_columns_scenarios', scenario='Imports', scenario_2='Natural Gas',\
                        column='Apply', column_2='Fuel')
    fugef_ngas_2 = this_df_fugef_ngas_2.iloc[0][per_first_yr]  # assume this is a constant

    # ...extract the dataframe with imports information:
    df_ngas_imports = filter_dataframe(df_scen, 'scenario_simple', scenario='% Imports for consumption', column='Parameter')
    df_ngas_imports_countries = \
        df_ngas_imports['Application_Countries'].tolist()
    # Add a filter to include countries with transport data only:
    df_ngas_imports_countries = \
        [c for c in df_ngas_imports_countries if c in tr_list_app_countries_u]

    # Function #133
    def get_local_export_data(query_dict, time_vector, case, reg=None, con=None, key=None, tech=None):
        """
        Retrieve local export data for a specific resource from a given dictionary based on the provided parameters.
        If the resource key is not found, the function returns a list of zeros for each year in the time vector.
        The function can handle both simple and large query structures by adjusting the retrieval process based on the 'case' parameter.

        Args:
        - query_dict (dict(type)): Dictionary containing export data for various resources. The structure of the dictionary may vary based on the 'case' parameter.
        - time_vector (list(type)): List of years for which the data is requested.
        - case (str): A string indicating the structure of 'query_dict'. Accepts 'simple' for flat dictionaries or 'large' for nested dictionaries.
        - reg (type, optional): Region identifier used in the case of a 'large' query structure. Defaults to None.
        - con (type, optional): Country identifier used in the case of a 'large' query structure. Defaults to None.
        - key (type, optional): Resource key used in the case of a 'large' query structure. Defaults to None.
        - tech (type, optional): Technology identifier used in the case of a 'large' query structure. Defaults to None.

        Returns:
        - local_export_data (list(type)): List containing the export values for each year in the 'time_vector'. If the key is not found, returns a list of zeros.
        """
        try:
            if case=='simple':
                local_export_data = query_dict
            elif case=='large':
                local_export_data = query_dict[reg][con][key][tech]
        except Exception:
            local_export_data = [0 for y in range(len(time_vector))]
        return local_export_data

    # ...iterate across country-wide consumption and find the imports, local production, and add the exports from above
    for acon in range(len(df_ngas_imports_countries)):
        this_con = df_ngas_imports_countries[acon]
        this_reg = country_2_region[this_con]

        query_dict = dict_local_reg[this_reg][this_con]
        local_ng_dem = []
        for y in range(len(time_vector)):
            add_val = \
                query_dict['Energy demand by fuel']['Natural Gas'][y] + \
                query_dict['Electricity fuel use']['PP_Thermal_Natural Gas']['Natural Gas'][y]  # this is a list
            local_ng_dem.append(add_val)

        local_ng_exp = get_local_export_data(query_dict['Natural Gas Exports (PJ)'], time_vector, 'simple')

        local_ng_production = []
        local_ng_fugitive_emissions = []

        imps_ng = []
        imps_ng_fugitive_emissions = []

        for y in range(len(time_vector)):
            this_year = int(time_vector[y])

            imports_share = df_ngas_imports.loc[acon, this_year]
            imports_PJ = local_ng_dem[y]*imports_share/100
            local_prod_PJ = local_ng_dem[y] - imports_PJ
            local_prod_PJ += local_ng_exp[y]

            local_ng_production.append(local_prod_PJ)
            local_ng_fugitive_emissions.append(local_prod_PJ*fugef_ngas)

            imps_ng.append(imports_PJ)
            imps_ng_fugitive_emissions.append(imports_PJ*fugef_ngas_2)

        dict_local_reg[this_reg][this_con].update({'Natural Gas Production (PJ)':deepcopy(local_ng_production)})  # aggregate
        dict_local_reg[this_reg][this_con].update({'Natural Gas Production Fugitive Emissions (MTon)':deepcopy(local_ng_fugitive_emissions)})
        dict_local_reg[this_reg][this_con].update({'Natural Gas Imports (PJ)':deepcopy(imps_ng)})
        dict_local_reg[this_reg][this_con].update({'Natural Gas Imports Fugitive Emissions (MTon)':deepcopy(imps_ng_fugitive_emissions)})

    ###########################################################################
    # *For job estimates, we must multiply times the installed capacity:
    # *For T&D estimates, we must check the electricity supply:

    # ...iterate across all countries and estimate the jobs
    for ng_reg in keys_2list_regions:
        keys_2list_countries = list(dict_local_reg[ng_reg].keys())
        this_reg = ng_reg
        for ng_cntry in keys_2list_countries:
            this_con = ng_cntry

            # ...now we must iterate across technologies with the technological capacity
            list_electric_sets_3.sort()  # this must work
            tech_counter = 0
            for tech in list_electric_sets_3:
                # -------------------------------------------------------------
                # >>> this section is for JOBS:
                list_cap = get_local_export_data(dict_local_reg, time_vector, 'large', reg=this_reg, con=this_con, key='Electricity total capacity', tech=tech)
                list_new_cap = get_local_export_data(dict_local_reg, time_vector, 'large', reg=this_reg, con=this_con, key='Electricity new capacity', tech=tech)
                list_new_prod = get_local_export_data(dict_local_reg, time_vector, 'large', reg=this_reg, con=this_con, key='Electricity new production per tech', tech=tech)
                list_demand_2_supply = get_local_export_data(dict_local_reg, time_vector, 'large', reg=this_reg, con=this_con, key='Electricity demand to supply', tech=tech)

                # ...we must also extract the jobs per unit of installed capacity
                mask_df4_job_fac = \
                    (df4_job_fac['Tech'] == tech)
                this_df4_job_fac = \
                    df4_job_fac.loc[mask_df4_job_fac]

                if len(this_df4_job_fac.index.tolist()) != 0:
                    jobs_factor_constru = this_df4_job_fac['Construction/installation (Job years/ MW)'].iloc[0]
                    jobs_factor_manufac = this_df4_job_fac['Manufacturing (Job years/ MW)'].iloc[0]
                    jobs_factor_opeyman = this_df4_job_fac['Operations & maintenance (Jobs/MW)'].iloc[0]
                    jobs_factor_decom = this_df4_job_fac['Decommissioning (Jobs/MW)'].iloc[0]
                else:
                    jobs_factor_constru = 0
                    jobs_factor_manufac = 0
                    jobs_factor_opeyman = 0
                    jobs_factor_decom = 0

                # ...we must create a LAC multiplier (based on the paper
                # https://link.springer.com/content/pdf/10.1007%2F978-3-030-05843-2_10.pdf)
                jobs_LAC_mult_vector_raw = ['' for x in range(len(time_vector))]
                for x in range(len(time_vector)):
                    if int(time_vector[x]) <= 2030:
                        jobs_LAC_mult_vector_raw[x] = 3.4
                    elif int(time_vector[x]) == 2040:
                        jobs_LAC_mult_vector_raw[x] = 3.1
                    elif int(time_vector[x]) == 2050:
                        jobs_LAC_mult_vector_raw[x] = 2.9
                    else:
                        pass
                jobs_LAC_mult_vector = \
                    interpolation_to_end(time_vector, ini_simu_yr,
                                         jobs_LAC_mult_vector_raw, 'ini',
                                         '')

                # ...we must estimate the jobs
                jobs_con_list_per_tech = \
                    [jobs_factor_constru*(1000*list_new_cap[y])*jobs_LAC_mult_vector[y] for y in range(len(time_vector))]
                jobs_man_list_per_tech = \
                    [jobs_factor_manufac*(1000*list_new_cap[y])*jobs_LAC_mult_vector[y] for y in range(len(time_vector))]
                jobs_oym_list_per_tech = \
                    [jobs_factor_opeyman*(1000*list_cap[y])*jobs_LAC_mult_vector[y] for y in range(len(time_vector))]
                jobs_dec_list_per_tech = \
                    [jobs_factor_decom*(1000*list_cap[y])*jobs_LAC_mult_vector[y] for y in range(len(time_vector))]

                # ...and store the results
                if tech_counter == 0:
                    dict_local_reg[this_reg][this_con].update({'Related construction jobs':{}})
                    dict_local_reg[this_reg][this_con].update({'Related manufacturing jobs':{}})
                    dict_local_reg[this_reg][this_con].update({'Related O&M jobs':{}})
                    dict_local_reg[this_reg][this_con].update({'Related decommissioning jobs':{}})
                dict_local_reg[this_reg][this_con]['Related construction jobs'].update({tech: jobs_con_list_per_tech})  # per tech
                dict_local_reg[this_reg][this_con]['Related manufacturing jobs'].update({tech: jobs_man_list_per_tech})
                dict_local_reg[this_reg][this_con]['Related O&M jobs'].update({tech: jobs_oym_list_per_tech})
                dict_local_reg[this_reg][this_con]['Related decommissioning jobs'].update({tech: jobs_dec_list_per_tech})

                # -------------------------------------------------------------
                # >>> this section is for T&D:
                try:
                    list_generation = \
                        dict_local_reg[this_reg][this_con]['Electricity total production'][tech]
                except Exception:
                    list_generation = [0 for y in range(len(time_vector))]

                # ...we must also extract the costs per unit of generated electricity
                mask_df4_tran_dist_fac = \
                    (df4_tran_dist_fac['Tech'] == tech)
                this_df4_tran_dist_fac = \
                    df4_tran_dist_fac.loc[mask_df4_tran_dist_fac]

                if len(this_df4_tran_dist_fac.index.tolist()) != 0:
                    transmi_capex = this_df4_tran_dist_fac['Transmission Capital Cost (M US$/PJ produced)'].iloc[0]
                    transmi_fopex = this_df4_tran_dist_fac['Transmission 2% Fixed Cost (M US$/PJ produced)'].iloc[0]
                    distri_capex = this_df4_tran_dist_fac['Distribution Capital Cost (M US$/PJ produced)'].iloc[0]
                    distri_fopex = this_df4_tran_dist_fac['Distribution 2% Fixed Cost (M US$/PJ produced)'].iloc[0]
                else:
                    transmi_capex = 0
                    transmi_fopex = 0
                    distri_capex = 0
                    distri_fopex = 0

                # ...we must estimate the t&d costs
                transmi_capex_list_per_tech = \
                    [transmi_capex*(list_new_prod[y]) for y in range(len(time_vector))]
                transmi_fopex_list_per_tech = \
                    [transmi_fopex*(list_generation[y]) for y in range(len(time_vector))]
                distri_capex_list_per_tech = \
                    [distri_capex*(list_new_prod[y]) for y in range(len(time_vector))]
                distri_fopex_list_per_tech = \
                    [distri_fopex*(list_generation[y]) for y in range(len(time_vector))]

                transmi_capex_list_per_tech_disc = deepcopy(transmi_capex_list_per_tech)
                transmi_fopex_list_per_tech_disc = deepcopy(transmi_fopex_list_per_tech)
                distri_capex_list_per_tech_disc = deepcopy(distri_capex_list_per_tech)
                distri_fopex_list_per_tech_disc = deepcopy(distri_fopex_list_per_tech)

                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    transmi_capex_list_per_tech_disc[y] *= disc_constant
                    transmi_fopex_list_per_tech_disc[y] *= disc_constant
                    distri_capex_list_per_tech_disc[y] *= disc_constant
                    distri_fopex_list_per_tech_disc[y] *= disc_constant

                # ...and store the results
                if tech_counter == 0:
                    dict_local_reg[this_reg][this_con].update({'Transmission CAPEX':{}})
                    dict_local_reg[this_reg][this_con].update({'Transmission Fixed OPEX':{}})
                    dict_local_reg[this_reg][this_con].update({'Distribution CAPEX':{}})
                    dict_local_reg[this_reg][this_con].update({'Distribution Fixed OPEX':{}})

                    dict_local_reg[this_reg][this_con].update({'Transmission CAPEX (disc)':{}})
                    dict_local_reg[this_reg][this_con].update({'Transmission Fixed OPEX (disc)':{}})
                    dict_local_reg[this_reg][this_con].update({'Distribution CAPEX (disc)':{}})
                    dict_local_reg[this_reg][this_con].update({'Distribution Fixed OPEX (disc)':{}})

                dict_local_reg[this_reg][this_con]['Transmission CAPEX'].update({tech: transmi_capex_list_per_tech})  # per tech
                dict_local_reg[this_reg][this_con]['Transmission Fixed OPEX'].update({tech: transmi_fopex_list_per_tech})
                dict_local_reg[this_reg][this_con]['Distribution CAPEX'].update({tech: distri_capex_list_per_tech})
                dict_local_reg[this_reg][this_con]['Distribution Fixed OPEX'].update({tech: distri_fopex_list_per_tech})
                dict_local_reg[this_reg][this_con]['Transmission CAPEX (disc)'].update({tech: transmi_capex_list_per_tech_disc})  # per tech
                dict_local_reg[this_reg][this_con]['Transmission Fixed OPEX (disc)'].update({tech: transmi_fopex_list_per_tech_disc})
                dict_local_reg[this_reg][this_con]['Distribution CAPEX (disc)'].update({tech: distri_capex_list_per_tech_disc})
                dict_local_reg[this_reg][this_con]['Distribution Fixed OPEX (disc)'].update({tech: distri_fopex_list_per_tech_disc})

                ###############################################################
                # ...increase the tech count:
                tech_counter += 1

    ###########################################################################
    # Store the elements
    dict_scen.update({this_scen: deepcopy(dict_local_reg)})
dict_scen_3 = deepcopy(dict_scen)
# sys.exit()
############################################################################################################################################
# Tier 2
dict_tier2_package = {
    'tax_params': tax_params,
    'country_list': country_list,
    'regions_list': regions_list,
    'scenario_list': scenario_list,
    'fuels': fuels,
    'types_all': types_all,
    'time_vector': time_vector,
    'dict_scen': dict_scen,
    'dict_activities_out': dict_activities_out,
    'params_tier2': params_tier2,
    'scenarios_cases_list': scenarios_cases_list,
    'dict_mult_depr': dict_mult_depr
}

def execute_bulac_tier2(dict_tier2_package):
    dict_scen, dict_scen_percentage, count_wrong_activity_fuel = bulac_tier2.run_model(dict_tier2_package)
    return dict_scen, dict_scen_percentage, count_wrong_activity_fuel

dict_scen, dict_scen_percentage, count_wrong_activity_fuel = execute_bulac_tier2(dict_tier2_package)
# sys.exit()
############################################################################################################################################

# ...our scenarios have run here
# 4) Now we can print the results file

# ...but first, review everything we have:
# Enlist the names of your output and the contained keys:
'''
OUTPUT:
'Global warming externalities by demand': tech (demand) / fuel / list with values as long as year
'Local pollution externalities by demand': tech (demand) / fuel / list with values as long as year
'Emissions by demand': tech (demand) / fuel / list with values as long as year
'Electricity fuel use': tech / fuel / list with values as long as year
'Global warming externalities in electricity': tech / fuel / list with values as long as year
'Local pollution externalities in electricity': tech / fuel / list with values as long as year
'Emissions in electricity': tech / fuel / list with values as long as year
'Electricity total capacity': tech / list with values as long as year
'Electricity residual capacity': tech / list with values as long as year
'Electricity new capacity': tech / list with values as long as year
'Electricity total production': tech / list with values as long as year || implicitly, electricity
'Electricity CAPEX': tech / list with values as long as year
'Electricity Fixed OPEX': tech / list with values as long as year
'Electricity Variable OPEX': tech / list with values as long as year
'Electricity Grid Connection Cost': tech / list with values as long as year
'''


# Enlist the names of your input and the contained keys:
'''
PARAMETERS:
this_externality_dict[fuel]['Global warming']  # constant
this_externality_dict[fuel]['Local pollution']  # constant
emissions_fuels_dict[fuel]  # list with values as long as year
idict_u_capex[tech]  # list with values as long as year
idict_u_fopex[tech]  # list with values as long as year
idict_u_vopex[tech]  # list with values as long as year
idict_u_gcc[tech]  # list with values as long as year
idict_cau[tech]  # list with values as long as year
idict_net_cap_factor[tech]  # list with values as long as year
idict_hr[tech]  # list with values as long as year
idict_oplife[tech]  # list with values as long as year
'''

### THIS IS ALL THE DATA THAT WE HAVE AVAILABLE:
# ...now, iterate and create a list of things you want to store:
list_dimensions = ['Adjust_ID','Strategy', 'Region', 'Country', 'Technology', 'Technology type', 'Fuel', 'Year']
list_inputs = ['Emission factor', #1
               'Unit CAPEX', #2
               'Unit fixed OPEX', #3
               'Unit variable OPEX', #4
               'Operational life'] #5

list_outputs = [
    'Fleet',  # 1
    'New Fleet',  # 2
    'Transport CAPEX [$]',  # 3
    'Transport Fixed OPEX [$]',  # 4
    'Transport Variable OPEX [$]',  # 5
    'Transport Tax Imports [$]',  # 6
    'Transport Tax IMESI_Venta [$]',  # 7
    'Transport Tax IVA_Venta [$]',  # 8
    'Transport Tax Patente [$]',  # 9
    'Transport Tax IMESI_Combust [$]',  # 10
    'Transport Tax IVA_Gasoil [$]',  # 11
    'Transport Tax IVA_Elec [$]',  # 12
    'Transport Tax IC [$]',  # 13
    'Transport Tax Otros_Gasoil [$]',  # 14
    'Transport Tax Tasa_Consular [$]',  # 15
    'Transport Tax Rodaje [$]',  # 16
    'Unit Tax Imports',  # 17
    'Unit Tax IMESI_Venta',  # 18
    'Unit Tax IVA_Venta',  # 19
    'Unit Tax Patente',  # 20
    'Unit Tax IMESI_Combust',  # 21
    'Unit Tax IVA_Gasoil',  # 22
    'Unit Tax IVA_Elec',  # 23
    'Unit Tax IC',  # 24
    'Unit Tax Otros_Gasoil',  # 25
    'Unit Tax Tasa_Consular',  # 26
    'Unit Tax Rodaje',  # 27
    'Fuel Consumption', # 28
    'Conversion Fuel Constant' # 29
]

list_inputs_add = [i + ' (input)' for i in list_inputs]
list_outputs_add = [i + ' (output)' for i in list_outputs]

h_scenario, h_strategy, h_region, h_country, h_tech, h_techtype, h_fuel, h_yr = \
    [], [], [], [], [], [], [], []

# Function #134
def determine_tech_type(tech, list_demand_sector_techs, list_electric_sets_3, types_all, 
                        techs_sw, types_livestock, types_liv_opex, types_rice_capex, 
                        types_rice_opex, types_rice_opex_2, types_shares_ac, types_shares_ref, types_shares_ext):
    """
    Determine the category of a given technology based on predefined lists of technology types. The function checks the technology against various lists to classify it into categories such as 'demand', 'power_plant', 'transport', etc.

    Args:
    - tech (str): The technology to be classified.
    - list_demand_sector_techs (list(type)): List of demand sector technologies.
    - list_electric_sets_3 (list(type)): List of electric set technologies.
    - types_all (list(type)): List of all types.
    - techs_sw (list(type)): List of waste technologies.
    - types_livestock (list(type)): List of livestock types.
    - types_liv_opex (list(type)): List of operating expense types for livestock.
    - types_rice_capex (list(type)): List of capital expense types for rice growth.
    - types_rice_opex (list(type)): List of operating expense types for rice growth.
    - types_rice_opex_2 (list(type)): List of alternative operating expense types for rice growth.
    - types_shares_ac (list(type)): List of air conditioning subsector types.
    - types_shares_ref (list(type)): List of refrigeration subsector types.
    - types_shares_ext (list(type)): List of extinguishers subsector types.

    Returns:
    - (str): The category of the technology.
    """
    if tech in list_demand_sector_techs:
        return 'demand'
    elif tech in list_electric_sets_3:
        return 'power_plant'
    elif tech in types_all:
        return 'transport'
    elif tech in techs_sw:
        return 'waste'
    elif tech in types_livestock:
        return 'cattle farming'
    elif tech in types_liv_opex:
        return 'costs for cattle farming'
    elif tech in types_rice_capex + types_rice_opex + types_rice_opex_2:
        return 'costs for rice growth'
    elif tech in types_shares_ac:
        return 'AC subsector'
    elif tech in types_shares_ref:
        return 'refrigeration subsector'
    elif tech in types_shares_ext:
        return 'extinguishers subsector'
    elif tech == '':
        return ''
    else:
        return 'none' 

# Function #135
def handle_processes(count_empties, h_list, data_dict, fuel, tech, case_condition, case='two', fst=None, scd=None, thd=None, fth=None):
    """
    Process and append data to the specified output list based on the given conditions and output ID.

    Args:
    - count_empties (int): count of list empties for each loop.
    - h_list (list): The output list to append data to.
    - data_dict (dict): The dictionary containing the data.
    - fuel (str): The fuel type.
    - tech (str): The technology type.
    - ase_condition (str): Indicating the combination of the condition ('01', '10', '00', '11').
        So: 
            if fuel == '' and tech != '': -> 10
            if fuel != '' and tech == '': -> 01
            if fuel != '' and tech != '': -> 00
            if fuel == '' and tech == '': -> 11
    
    - case (str): The case type indicating the depth of data access in the dictionary ('two', 'three', 'four').
        Example:
            h_o33.append(this_data_dict[list_outputs[32]][tech][y])
            case = 'three'
            fst = list_outputs[32]
            scd = tech
            thd = y
            fth_key = None
    
    - fst (str or list, optional): The key for the first position in the dictionary. Defaults to None.
    - scd (str or int, optional): The key for the second position in the dictionary. Defaults to None.
    - thd (str or int, optional): The key for the third position in the dictionary. Defaults to None.
    - fth (str or int, optional): The key for the fourth position in the dictionary. Defaults to None.

    Returns:
    - count_empties (int): Updated count of empty entries.
    - h_list (list): The updated list with appended data.
    """
    condition_flag = False

    if case_condition=='10':
        if fuel == '' and tech != '':
            condition_flag = True
    elif case_condition=='01':
        if fuel != '' and tech == '':
            condition_flag = True
    elif case_condition=='00':
        if fuel != '' and tech != '':
            condition_flag = True
    elif case_condition=='11':
        if fuel == '' and tech == '':
            condition_flag = True
            
            
    if condition_flag: 
        try:
            if case == 'two':
                value = data_dict[fst][scd]
            elif case == 'three':
                value = data_dict[fst][scd][thd]
            elif case == 'four':
                value = data_dict[fst][scd][thd][fth]
                
            if pd.isna(value):
                h_list.append(np.nan)
            else:
                h_list.append(value)
            
        except Exception:
            h_list.append(0)
            count_empties += 1                
    else:
        h_list.append(0)
        count_empties += 1  

    return count_empties, h_list

# Function #136
def pop_last_from_outputs(output_lists, output_range):
    """
    Remove the last element from each output list in the specified range.

    Args:
    - output_lists (dict(type)): Dictionary containing lists named as 'h_oX' where X is the output ID.
    - output_range (range): Range object specifying the output IDs to process.

    Returns:
    - output_lists(dict): The dictionary with updated lists after removing the last element from each.
    """
    for output_id in output_range:
        output_list_name = f'h_o{output_id}'
        if output_list_name in output_lists:
            output_lists[output_list_name].pop()            
    return output_lists

# Function #137
def pop_last_from_inputs(input_lists, input_range):
    """
    Remove the last element from each output list in the specified range.

    Args:
    - input_lists (dict): Dictionary containing lists named as 'h_iX' where X is the input ID.
    - input_range (range): Range object specifying the input IDs to process.

    Returns:
    - input_lists(dict): The dictionary with updated lists after removing the last element from each.
    """
    for input_id in input_range:
        input_list_name = f'h_i{input_id}'
        if input_list_name in input_lists:
            input_lists[input_list_name].pop()
    return input_lists
            

# ...here, clean up the fuels:
list_fuel_clean = [i for i in list_fuel if 'Total' not in i]
if overwrite_transport_model:
    list_fuel_clean += list(dict_eq_transport_fuels.keys())
list_fuel_clean += ['']

output_lists = {f'h_o{i}': [] for i in range(1, 30)}
input_lists = {f'h_i{i}': [] for i in range(1, 6)}

print('\n')
print('PROCESS 2 - PRINTING THE INPUTS AND OUTPUTS')
cases_list = list(dict_scen_percentage.keys())
for sc in range(len(cases_list)):
    this_scen_case = cases_list[sc]
    for s in range(len(scenario_list)):
        this_scen = scenario_list[s]
    
        # regions_list = list(dict_regs_and_countries.keys())
        # regions_list = ['4_The Amazon', '5_Southern Cone']
        # regions_list = ['1_Mexico', '2_Central America', '4_The Amazon', '5_Southern Cone']
        regions_list = ['1_Mexico', '2_Central America', '3_Caribbean', '4_The Amazon', '5_Southern Cone']
    
        for r in range(len(regions_list)):
            this_reg = regions_list[r]
    
            country_list = dict_regs_and_countries[this_reg]
            country_list.sort()
    
            # Add a filter to include countries with transport data only:
            country_list = [c for c in country_list if c in tr_list_app_countries_u]
    
            for c in range(len(country_list)):
                this_country = country_list[c]
                
                if not model_rac:
                    types_shares_ac = []
                    types_shares_ref = []
                    types_shares_ext = []
                    
                if not model_agro_and_waste:
                    techs_sw = []
                    types_livestock = []
                    types_liv_opex = []
                    types_rice_capex = []
                    types_rice_opex = []
                    types_rice_opex_2 = []
                    
                
                types_all_rac = types_shares_ac + types_shares_ref + types_shares_ext
                types_all_rac = list(set(types_all_rac))
                types_all_rac.sort()
    
                # First iterable: list_demand_sector_techs
                # Second iterable: list_electric_sets_3
                # inner iterable 1: list_fuel_clean
                # inner iterable 2: time_vector
                tech_iterable = list_demand_sector_techs +  list_electric_sets_3 +\
                    types_all + techs_sw + types_livestock + types_liv_opex + \
                        types_rice_capex + types_rice_opex + types_rice_opex_2 + \
                        types_all_rac + ['']
                        
                tech_iterable = types_all # only for this test that we need only taxes
                        
                
                for tech in tech_iterable:
                    for fuel in list_fuel_clean:
                        for y in range(len(time_vector)):
                            count_empties = 0
    
                            tech_type = determine_tech_type(tech, list_demand_sector_techs, list_electric_sets_3, types_all, 
                                                            techs_sw, types_livestock, types_liv_opex, types_rice_capex, 
                                                            types_rice_opex, types_rice_opex_2, types_shares_ac, types_shares_ref, types_shares_ext)
                            this_data_dict = \
                                dict_scen_percentage[this_scen_case][this_scen][this_reg][this_country]
    
                            # Store inputs:                          
                            count_empties, input_lists['h_i1'] = handle_processes(count_empties, input_lists['h_i1'], emissions_fuels_dict, fuel, tech, '01','two', fuel, y)
                            count_empties, input_lists['h_i2'] = handle_processes(count_empties, input_lists['h_i2'], idict_u_capex, fuel, tech, '10', 'two',tech, y)
                            count_empties, input_lists['h_i3'] = handle_processes(count_empties, input_lists['h_i3'], idict_u_fopex, fuel, tech, '10', 'two', tech, y)
                            count_empties, input_lists['h_i4'] = handle_processes(count_empties, input_lists['h_i4'], idict_u_vopex, fuel, tech, '10', 'two', tech, y)
                            count_empties, input_lists['h_i5'] = handle_processes(count_empties, input_lists['h_i5'], idict_oplife, fuel, tech, '10', 'two', tech, y)
    
                            # Store outputs:
                            # Lists of IDs for case conditions
                            list_output_01 = []
                            list_output_10 = []
                            list_output_00 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,29]
                            list_output_11 = []
                            
                            for output_id in range(1, 30):  # Loop for all outputs, the second number y one more than the last output
    
                                # Conditions to select the correct case condition for combination of "fuel" and "tech"                        
                                if output_id in list_output_01:
                                    case_condition = '01'
                                elif output_id in list_output_10:
                                    case_condition = '10'
                                elif output_id in list_output_00:
                                    case_condition = '00'
                                elif output_id in list_output_11:
                                    case_condition = '11'   
                                    
                                # Conditions to select the correct parameters for which one case

                                # if output_id in []:
                                #     # Case 'two'
                                #     case_type = 'two'
                                #     fst_key = list_outputs[output_id - 1]
                                #     scd_key = y
                                #     thd_key = None
                                #     fth_key = None
                                # elif output_id in []:
                                #     # Case 'three' 1
                                #     case_type = 'three'
                                #     fst_key = list_outputs[output_id - 1]
                                #     scd_key = tech
                                #     thd_key = y
                                #     fth_key = None
                                # elif output_id == 13:
                                #     # Case 'three' 2
                                #     case_type = 'three'
                                #     fst_key = list_outputs[output_id - 1]
                                #     scd_key = fuel
                                #     thd_key = y
                                #     fth_key = None
                                if output_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]:
                                    # Case 'four' 1
                                    case_type = 'four'
                                    fst_key = list_outputs[output_id - 1]
                                    scd_key = tech
                                    thd_key = fuel
                                    fth_key = y
                                elif output_id in [29]:
                                    # Case 'two'
                                    case_type = 'three'
                                    fst_key = list_outputs[output_id - 1]
                                    scd_key = tech
                                    thd_key = fuel
                                    fth_key = None
                                # elif output_id == 14:
                                #     # Case 'four' 2
                                #     case_type = 'four'
                                #     fst_key = list_outputs[output_id - 1]
                                #     scd_key = tech
                                #     thd_key = 'Total'
                                #     fth_key = y
    
                                # Process outputs                            
                                count_empties, output_lists[f'h_o{output_id}'] = handle_processes(count_empties, output_lists[f'h_o{output_id}'], this_data_dict, fuel, tech, case_condition, case_type, fst_key, scd_key, thd_key, fth_key)
                                                        
                            if count_empties == 34:  # gotta pop, because it is an empty row:
                                # Inputs
                                input_lists = pop_last_from_inputs(input_lists, range(1, 6))
                                # Outputs
                                output_lists = pop_last_from_outputs(output_lists, range(1, 30))
    
                            else:
                                h_scenario.append(this_scen_case)
                                h_strategy.append(this_scen)
                                h_region.append(this_reg)
                                h_country.append(this_country)
                                h_tech.append(tech)
                                h_techtype.append(tech_type)
                                h_fuel.append(fuel)
                                h_yr.append(time_vector[y])

# Review if *zero* elements exist:    
# List of names variables

variable_names = \
    [f'h_i{i}' for i in range(1, 6)] + \
    [f'h_o{i}' for i in range(1, 30)]

# Construct list_variables by accessing dictionaries
list_inputs_variables = [input_lists[var_name] for var_name in variable_names if var_name in input_lists]
list_outputs_variables = [output_lists[var_name] for var_name in variable_names if var_name in output_lists]
list_variables = list_inputs_variables + list_outputs_variables

h_count = 0
h_zeros = []
for h in list_variables:
    h = [float(i) for i in h] 
    if sum(h) == 0.0:
        h_zeros.append(h_count)
    h_count += 1


# Review the lengths:
print(1, list_dimensions[0], len(h_scenario)) #1
print(2, list_dimensions[1], len(h_strategy)) #2
print(3, list_dimensions[2], len(h_region)) #3
print(4, list_dimensions[3], len(h_country)) #4
print(5, list_dimensions[4], len(h_tech)) #5
print(6, list_dimensions[5], len(h_techtype)) #6
print(7, list_dimensions[6], len(h_fuel)) #7
print(8, list_dimensions[7], len(h_yr)) #8
# Inputs
for i in range(len(list_inputs_add)):
    print(9 + i, list_inputs_add[i], len(input_lists[f'h_i{i + 1}']))
# Outputs
for i in range(len(list_outputs_add)):
    print(14 + i, list_outputs_add[i], len(output_lists[f'h_o{i + 1}']))

# Print message about wrong distribution of factors for unit tax calculation
if params_tier2['by_factores_fuel'] and count_wrong_activity_fuel != 0:
    print('Check the file activity_warnings.txt and make correction into\nTaxes_Factors.xlsx\n\n')

# Convert to output:
print('\n')
print('Convert lists to dataframe for printing:')
dict_output = {list_dimensions[0]: h_scenario, #1
               list_dimensions[1]: h_strategy, #2
               list_dimensions[2]: h_region, #3
               list_dimensions[3]: h_country, #4
               list_dimensions[4]: h_tech, #5
               list_dimensions[5]: h_techtype, #6
               list_dimensions[6]: h_fuel, #7
               list_dimensions[7]: h_yr, #8                            
               }
# Inputs
for i in range(len(list_inputs_add)):
    dict_output[list_inputs_add[i]] = input_lists[f'h_i{i + 1}']
# Outputs
for i in range(len(list_outputs_add)):
    dict_output[list_outputs_add[i]] = output_lists[f'h_o{i + 1}']

# Let's print variable costs
df_output_name = 'model_BULAC_simulation_0.csv'
df_output = pd.DataFrame.from_dict(dict_output)
df_output['Future'] = 0
df_output.to_csv(path + '/' + df_output_name, index=None, header=True)

# df_output_name_f0 = 'model_BULAC_simulation_0.csv'
# df_output_f0 = deepcopy(df_output)
# df_output_f0['Future'] = 0
# list_inner = list_dimensions + [
#                     'Emissions by demand (output)',
#                     'Emissions in electricity (output)',
#                     'Energy demand by sector (output)',
#                     'Energy demand by fuel (output)',
#                     'Energy intensity by sector (output)',
#                     'Global warming externalities by demand (output)',
#                     'Global warming externalities in electricity (output)',
#                     'Local pollution externalities by demand (output)',
#                     'Local pollution externalities in electricity (output)',
#                     'Global warming externalities in electricity (disc) (output)',
#                     'Local pollution externalities in electricity (disc) (output)',
#                     'Fleet (output)',
#                     'New Fleet (output)',
#                     'Transport CAPEX [$] (output)',
#                     'Transport Fixed OPEX [$] (output)',
#                     'Transport Variable OPEX [$] (output)',
#                     'Transport Tax Imports [$] (output)',
#                     'Transport Tax IMESI_Venta [$] (output)',
#                     'Transport Tax IVA_Venta [$] (output)',
#                     'Transport Tax Patente [$] (output)',
#                     'Transport Tax IMESI_Combust [$] (output)',
#                     'Transport Tax IVA_Gasoil [$] (output)',
#                     'Transport Tax IVA_Elec [$] (output)',
#                     'Transport Tax IC [$] (output)',
#                     'Transport Tax Otros_Gasoil [$] (output)',
#                     'Transport Tax Tasa_Consular [$] (output)',
#                     'Transport Tax Rodaje [$] (output)',
#                     'Future']
# df_output_f0_out = df_output_f0[list_inner]
# df_output_f0_out.to_csv(path + '/' + df_output_name_f0, index=None, header=True)


# Recording final time of execution:
end_f = time.time()
te_f = -start_1 + end_f  # te: time_elapsed
print(str(te_f) + ' seconds /', str(te_f/60) + ' minutes')
print('*: This automatic analysis is finished.')
