# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:26:13 2024
Last updated: Jan. 19, 2025

@author: Climate Lead Group; Luis Victor-Gallardo, Jairo Quirós-Tortós,
        Andrey Salazar-Vargas
Suggested citation: UNEP (2022). Is Natural Gas a Good Investment for Latin 
                                America and the Caribbean? From Economic to 
                                Employment and Climate Impacts of the Power
                                Sector. https://wedocs.unep.org/handle/20.500.11822/40923
"""

import pandas as pd
from copy import deepcopy
import sys
import re

# Import functions that support Tier 2 of this model:
from model_bulac_tier2_funcs import calculate_tax_differences, \
    calculate_fiscal_gap_and_adjusted_tax_contributions, sum_and_assign_weights, \
    calculate_tax_impact, apply_tax_impact_to_structure, \
    calculate_unit_tax_for_all_scenarios, calculate_unit_tax_one_tax_by_time, \
    calculate_unit_tax_by_activity, read_factors, verify_factors, \
    redistribute_factors_if_no_activity, get_activity_for_fuel, \
    redistribute_across_technologies, calculate_unit_tax_with_factors_for_fuels, \
    integrate_unit_taxes_into_dict_scen, changes_tax_keys_unit_taxes_dict, \
    take_factors_distribution, calculate_unit_tax_by_sum_activity


def run_model(dict_tier2_package):
    
    # Unpacking the dictionary into local variables
    tax_params = dict_tier2_package['tax_params']
    country_list = dict_tier2_package['country_list']
    regions_list = dict_tier2_package['regions_list']
    scenario_list = dict_tier2_package['scenario_list']
    fuels = dict_tier2_package['fuels']
    types_all = dict_tier2_package['types_all']
    time_vector = dict_tier2_package['time_vector']
    dict_scen = dict_tier2_package['dict_scen']
    dict_activities_out = dict_tier2_package['dict_activities_out']
    params_tier2 = dict_tier2_package['params_tier2']
    scenarios_cases_list = dict_tier2_package['scenarios_cases_list']
    dict_mult_depr = dict_tier2_package['dict_mult_depr']
    
    dict_scen_2 = deepcopy(dict_scen)
    
    
    # Change the name of a tax
    tax_params[tax_params.index('Impuesto_Carbono')] = 'IC'
    tax_params.append('Total Taxes Incomes')
    
    regions_list = ['5_Southern Cone']  # Assume this is constant or defined elsewhere
    
    # Initialize the scenario dictionary
    dict_income_scen = {
        scen: {
            region: {
                country: {atax: [] for atax in tax_params}
                for country in country_list
            } for region in regions_list
        } for scen in scenario_list
    }
    
    # Initialize the scenario dictionary
    dict_tax_scen = {
        scen: {
            region: {
                country: {
                    atax: {
                        tech: {
                            fuel: [] for fuel in fuels
                        } for tech in types_all
                    } for atax in tax_params
                } for country in country_list
            } for region in regions_list
        } for scen in scenario_list #if scen != 'BAU'
    }
    
    # Prepare the dictionary before removing an important element
    for scen in scenario_list:
        for region in regions_list:
            for country in country_list:
                # Initialize total_taxes_incomes outside the atax loop
                total_taxes_incomes = [0] * len(time_vector)
                for atax in tax_params:
                    if atax != 'Total Taxes Incomes':  # Ignore this entry when summing taxes
                        # Move dict initialization here if needed
                        dict_income_scen[scen][region][country][atax] = []
                        for y in range(len(time_vector)):
                            sum_year_temp = 0  # Move inside year loop to reset for each year
                            for tech in types_all:
                                for fuel in fuels:
                                    tax_year = dict_scen[scen][region][country][f'Transport Tax {atax} [$]'][tech][fuel][y]
                                    sum_year_temp += tax_year
                                    # Check if BAU and apply specific logic or append to a scenario-specific list
                                    # if scen != 'BAU':
                                    dict_tax_scen[scen][region][country][atax][tech][fuel].append(tax_year)
    
                            total_taxes_incomes[y] += sum_year_temp
                            dict_income_scen[scen][region][country][atax].append(sum_year_temp)
                            
                # Save the total taxes incomes after processing all taxes
                dict_income_scen[scen][region][country]['Total Taxes Incomes'] = total_taxes_incomes
    
    
    # Remove 'Total Taxes Incomes' from the list so it does not affect future operations
    tax_params.remove('Total Taxes Incomes')
    
    # Call the calculate_tax_differences function to compute the tax differences
    # between BAU and other scenarios. This call utilizes the dict_income_scen
    # dictionary, which must be defined with the required structure
    # before calling the function.
    # tax_differences = calculate_tax_differences(dict_income_scen)
    
    # Call the calculate_fiscal_gap_and_adjusted_tax_contributions function to determine
    # fiscal gaps and adjust tax contributions for the specified year across different scenarios.
    # This function analyzes differences in total tax incomes between the 'BAU' scenario and other
    # scenarios to compute the fiscal gap.
    # It then adjusts the contribution of each tax type by adding the fiscal gap to the current
    # tax contribution for each country and region. 'year_to_adjust' is the target year for
    # which the calculations are performed and should be defined along with 'time_vector'
    # before calling this function.
    year_to_adjust = 2050
    fiscal_gaps, adjustments, adjustments_percent, tax_contribution = calculate_fiscal_gap_and_adjusted_tax_contributions(year_to_adjust, dict_income_scen, time_vector)
    # sys.exit()
    # Calculate unit taxes for base case without adjusments
    unit_taxes_base_case, unit_taxes_percentage = calculate_unit_tax_for_all_scenarios(year_to_adjust, dict_scen, dict_tax_scen, dict_activities_out, time_vector, params_tier2, dict_mult_depr)
    

    # sys.exit()
    
    # Take the base BAU scenario of unit taxes
    unit_taxes_bau = {}
    unit_taxes_bau['BAU'] = deepcopy(unit_taxes_base_case['BAU'])
    unit_taxes_bau_percentage = {}
    unit_taxes_bau_percentage['BAU'] = deepcopy(unit_taxes_percentage['BAU'])

    
    # Define the Excel file path
    taxes_excel_name = 'Taxes_Factors.xlsx'
    taxes_excel = pd.ExcelFile(taxes_excel_name)
    sheet_name = 'FACTORS'
    
    # Read the 'FACTORS' sheet directly into a DataFrame and immediately close the file
    sheet_factors = pd.read_excel(taxes_excel_name, sheet_name='FACTORS')
    
    # Create a list of columns that contain 'FACTORS' in their name
    columns_factors = [col for col in sheet_factors.columns if 'FACTORS' in col]

    # Initialize the dictionary to accumulate the filtered data per scenario
    filtered_dict_scen = {}
    filtered_dict_scen_percentage = {}
    count_wrong_activity_fuel = None   
    
    # Update dict_scen with unit taxes data
    for adjust_id in scenarios_cases_list:
    # for adjust_id, scenarios in unit_taxes_one_tax_time.items():
        for scenario, regions in unit_taxes_base_case.items():
            if scenario == 'BAU':
                continue
            for region, countries in regions.items():
                for country, taxes in countries.items():
                    for tax, techs_data in taxes.items():
                        # Check if the adjust_id is in the tax name and in tax_params
                        if adjust_id == scenarios_cases_list[0]:
                            # Initialize only if necessary
                            if adjust_id not in filtered_dict_scen:
                                filtered_dict_scen[adjust_id] = {}
                                filtered_dict_scen_percentage[adjust_id] = {}
                            # Store the unit taxes and the first level scenarios
                            filtered_dict_scen = {adjust_id: dict_scen}
                            filtered_dict_scen = integrate_unit_taxes_into_dict_scen(unit_taxes_base_case, filtered_dict_scen)
                            filtered_dict_scen_percentage = {adjust_id: dict_scen}
                            filtered_dict_scen_percentage = integrate_unit_taxes_into_dict_scen(unit_taxes_percentage, filtered_dict_scen_percentage)
                            
                            base_dict_scen = deepcopy(unit_taxes_base_case)
                            
                        elif adjust_id == 'one_tax_by_time' and params_tier2['by_one_tax_by_time']:
                            # Calculate unit taxes for the year year_to_adjust using predefined scenario,
                            # adjustments, and activities data.
                            unit_taxes_one_tax_time = {}
                            unit_taxes_one_tax_time_percentages = {}
                            for tax_id in tax_params:
                                unit_taxes_one_tax_time_temp = {}
                                unit_taxes_one_tax_time_percentages_temp = {}
                                
                                unit_taxes_one_tax_time_temp, unit_taxes_one_tax_time_percentages_temp, dict_activity_distribution_gap = calculate_unit_tax_one_tax_by_time(
                                    year_to_adjust,
                                    dict_scen,
                                    deepcopy(adjustments_percent),
                                    dict_tax_scen,
                                    dict_activities_out,
                                    time_vector,
                                    fiscal_gaps,
                                    tax_contribution,
                                    params_tier2,
                                    tax_id,
                                    dict_mult_depr,
                                    base_dict_scen)
                                unit_temp = deepcopy(unit_taxes_one_tax_time_temp)
                                
                                unit_taxes_one_tax_time_temp.update(unit_taxes_bau)
                                unit_taxes_one_tax_time_temp = changes_tax_keys_unit_taxes_dict(deepcopy(unit_taxes_one_tax_time_temp))
                                unit_taxes_one_tax_time[tax_id] = unit_taxes_one_tax_time_temp

                                unit_taxes_one_tax_time_percentages_temp.update(unit_taxes_bau_percentage)
                                unit_taxes_one_tax_time_percentages_temp = changes_tax_keys_unit_taxes_dict(deepcopy(unit_taxes_one_tax_time_percentages_temp))
                                unit_taxes_one_tax_time_percentages[tax_id] = unit_taxes_one_tax_time_percentages_temp
                            
                            filtered_dict_scen.update(unit_taxes_one_tax_time)
                            filtered_dict_scen_percentage.update(unit_taxes_one_tax_time_percentages)
                            
    
                        else:
                            for col_i in range(len(columns_factors)):
                                factors_by_tax, factors_by_fuel = \
                                    take_factors_distribution(sheet_name,
                                                          columns_factors[col_i],
                                                          taxes_excel, scenario_list,
                                                          regions_list, country_list,
                                                          params_tier2['by_factores_fuel'])
                                # print(len(columns_factors))
                                # print(columns_factors)
                                # sys.exit()

                                if adjust_id != scenarios_cases_list[0] and adjust_id not in tax_params and adjust_id == 'factors_by_activity' and params_tier2['by_activity']:
                                    # Call the function to calculate unit taxes using predefined factors, 
                                    # fiscal gaps and make distribution according activity
                                    # unit_taxes_activity_by_factor, unit_taxes_activity_by_factor_percentage = calculate_unit_tax_by_sum_activity(
                                    unit_taxes_activity_by_factor, unit_taxes_activity_by_factor_percentage = calculate_unit_tax_by_activity(
                                        year=year_to_adjust,
                                        dict_scen=dict_scen_2,
                                        adjustments_percent=deepcopy(adjustments_percent),
                                        dict_tax_scen=dict_tax_scen,
                                        dict_activities_out=dict_activities_out,
                                        fiscal_gaps=fiscal_gaps,
                                        factors=factors_by_tax,
                                        time_vector=time_vector,
                                        params_tier2=params_tier2,
                                        dict_mult_depr=dict_mult_depr,
                                        base_dict_scen=base_dict_scen
                                    )
                                    # Extract the complete number after underscore
                                    factor_number = re.search(r'\d+$', columns_factors[col_i]).group()
                                    
                                    # Include unit taxes of the BAU scenario
                                    unit_taxes_activity_by_factor.update(unit_taxes_bau)
                                    unit_taxes_activity_by_factor = changes_tax_keys_unit_taxes_dict(deepcopy(unit_taxes_activity_by_factor))
                                    filtered_dict_scen[f'{adjust_id}_{factor_number}']= deepcopy(unit_taxes_activity_by_factor)
                                    unit_taxes_activity_by_factor_percentage.update(unit_taxes_bau_percentage)
                                    unit_taxes_activity_by_factor_percentage = changes_tax_keys_unit_taxes_dict(deepcopy(unit_taxes_activity_by_factor_percentage))
                                    filtered_dict_scen_percentage[f'{adjust_id}_{factor_number}']= deepcopy(unit_taxes_activity_by_factor_percentage)
                            
                                elif adjust_id != scenarios_cases_list[0] and adjust_id not in tax_params and adjust_id == 'factors' and params_tier2['by_factores_fuel']:
                                    # Call the function to calculate unit taxes using predefined factors for 
                                    # each tax/tech/fuel and fiscal gaps
                                    count_wrong_activity_fuel = 0   
                                    unit_taxes_by_factor, unit_taxes_by_factor_percentage, warning_count = calculate_unit_tax_with_factors_for_fuels(
                                        year=year_to_adjust,
                                        dict_scen=dict_scen_2,
                                        dict_tax_scen=dict_tax_scen,
                                        dict_activities_out=dict_activities_out,
                                        fiscal_gaps=fiscal_gaps,
                                        factors_by_fuel=factors_by_fuel,
                                        time_vector=time_vector,
                                        count=count_wrong_activity_fuel,
                                        params_tier2=params_tier2,
                                        dict_mult_depr=dict_mult_depr,
                                        base_dict_scen=base_dict_scen
                                    )
                                    # Extract the complete number after underscore
                                    factor_number = re.search(r'\d+$', columns_factors[col_i]).group()

                                    # Include unit taxes of the BAU scenario
                                    unit_taxes_by_factor.update(unit_taxes_bau)
                                    unit_taxes_by_factor = changes_tax_keys_unit_taxes_dict(deepcopy(unit_taxes_by_factor))
                                    filtered_dict_scen[f'{adjust_id}_{factor_number}']= deepcopy(unit_taxes_by_factor)
                                    unit_taxes_by_factor_percentage.update(unit_taxes_bau_percentage)
                                    unit_taxes_by_factor_percentage = changes_tax_keys_unit_taxes_dict(deepcopy(unit_taxes_by_factor_percentage))
                                    filtered_dict_scen_percentage[f'{adjust_id}_{factor_number}']= deepcopy(unit_taxes_by_factor_percentage)
    
    
    # Update dict_scen with filtered data by scenario/case
    dict_scen = {}
    dict_scen.update(filtered_dict_scen)
    dict_scen_percentage = {}
    dict_scen_percentage.update(filtered_dict_scen_percentage)

    
    return dict_scen, dict_scen_percentage, count_wrong_activity_fuel#, dict_activities_out