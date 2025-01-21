# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:55:54 2024
Last updated: Jan. 19, 2025

@author: Climate Lead Group; Luis Victor-Gallardo, Jairo Quirós-Tortós,
        Andrey Salazar-Vargas
Suggested citation: UNEP (2022). Is Natural Gas a Good Investment for Latin 
                                America and the Caribbean? From Economic to 
                                Employment and Climate Impacts of the Power
                                Sector. https://wedocs.unep.org/handle/20.500.11822/40923
"""
import pandas as pd
import sys
import numpy as np
from copy import deepcopy

# Import functions that support this model:
from model_bulac_funcs import interpolation_to_end

def calculate_tax_differences(dict_income_scen):
    """
    Calculates the differences in total tax incomes between each scenario and the BAU (Business As Usual) scenario.
    
    This function iterates over each scenario in the input dictionary, comparing tax incomes to those of the BAU scenario
    for each region and country, and computes the difference (gap) for each year listed under 'Total Taxes Incomes'.
    
    Args:
        dict_income_scen (dict): A dictionary structured with scenario names as keys and values as nested dictionaries,
                                 where each nested dictionary contains regional data. Each region's data further maps to 
                                 countries containing lists of tax incomes under 'Total Taxes Incomes'.
    
    Returns:
        dict: A dictionary containing the tax differences for each scenario (except BAU), with each scenario key mapping
              to a nested dictionary of regions, which in turn map to countries. Each country's entry is a list of
              differences in tax incomes compared to the BAU scenario for the corresponding years.
    
    Example:
        Given a dictionary with the structure:
        {
            "BAU": {"Region1": {"Country1": {"Total Taxes Incomes": [100, 110, 120]}}},
            "Scenario1": {"Region1": {"Country1": {"Total Taxes Incomes": [90, 105, 115]}}}
        }
        The output will be:
        {
            "Scenario1": {"Region1": {"Country1": {"Taxes Gap": [-10, -5, -5]}}}
        }
    """
    # Dictionary to store the results
    result_diff = {}

    # Extract the list of 'Total Taxes Incomes' from the BAU scenario
    bau_data = dict_income_scen['BAU']

    # Iterate over all scenarios except 'BAU'
    for scenario, region_data in dict_income_scen.items():
        if scenario != 'BAU':
            result_diff[scenario] = {}
            for region, country_data in region_data.items():
                result_diff[scenario][region] = {}
                for country, tax_data in country_data.items():
                    # Perform the subtraction only if 'Total Taxes Incomes' is present in both scenarios
                    if 'Total Taxes Incomes' in tax_data and 'Total Taxes Incomes' in bau_data[region][country]:
                        # Subtract the BAU list from the current scenario list
                        result_diff[scenario][region][country] = {
                            'Taxes Gap': [
                                tax - bau
                                for bau, tax in zip(bau_data[region][country]['Total Taxes Incomes'], tax_data['Total Taxes Incomes'])
                            ]
                        }

    return result_diff

    
def calculate_fiscal_gap_and_adjusted_tax_contributions(year, dict_income_scen, time_vector):
    """
    Calculates the fiscal gap and adjusts tax contributions for each tax type by adding the fiscal gap to the current contributions.

    This function processes tax data across multiple scenarios to identify fiscal gaps between the 'BAU' scenario and other scenarios for a given year.
    It then adjusts the contribution of each tax by summing the fiscal gap with the current tax contributions for each country and region.

    Args:
        year (int): The year for which the calculations are to be performed. This should be a value present in the 'time_vector'.
        dict_income_scen (dict): A nested dictionary containing tax data across different scenarios, regions, and countries.
        time_vector (list): A list of years for which data is available, used to find the index of the specified 'year'.

    Returns:
        tuple of dicts:
            fiscal_gaps (dict): A dictionary detailing the fiscal gaps calculated between the 'BAU' scenario and each of the other scenarios.
            adjustments (dict): A dictionary showing the adjusted tax contributions for each tax type after adding the fiscal gap.
            tax_contribution (dict): A dictionary containing the original tax contributions before adjustments.
    
    Raises:
        ValueError: If the specified 'year' is not found in 'time_vector', an error is raised indicating the year is not in the time vector.
    """
    
    try:
        year_index = time_vector.index(year)
    except ValueError:
        print(f"El año {year} no se encuentra en time_vector.")
        return {}, {}

    fiscal_gaps = {}
    adjustments = {}
    adjustments_percent = {}
    tax_contribution = {}

    # Extract BAU scenario incomes for all regions and countries
    bau_incomes = {region: {country: data['Total Taxes Incomes'][year_index]
                            for country, data in countries.items()}
                   for region, countries in dict_income_scen['BAU'].items()}

    # Iterate over all scenarios except 'BAU'
    for scenario in dict_income_scen:
        if scenario != 'BAU':
            fiscal_gaps[scenario] = {}
            adjustments[scenario] = {}
            adjustments_percent[scenario] = {}
            tax_contribution[scenario] = {}
            for region, countries in dict_income_scen[scenario].items():
                fiscal_gaps[scenario][region] = {}
                adjustments[scenario][region] = {}
                adjustments_percent[scenario][region] = {}
                tax_contribution[scenario][region] = {}
                for country, data in countries.items():
                    scenario_income = data['Total Taxes Incomes'][year_index]
                    bau_income = bau_incomes[region][country]
                    fiscal_gap = bau_income - scenario_income
                    fiscal_gaps[scenario][region][country] = fiscal_gap

                    # Adjust each tax by adding the fiscal gap to the current tax income
                    adjustments[scenario][region][country] = {}
                    adjustments_percent[scenario][region][country] = {}
                    tax_contribution[scenario][region][country] = {}
                    for tax in data:
                        if tax != 'Total Taxes Incomes':
                            current_tax_income = data[tax][year_index]
                            tax_adjustment = current_tax_income + fiscal_gap
                            adjustments[scenario][region][country][tax] = tax_adjustment
                            if current_tax_income != 0:
                                # adjustments_percent[scenario][region][country][tax] = tax_adjustment/current_tax_income
                                adjustments_percent[scenario][region][country][tax] = fiscal_gap/current_tax_income
                            else:
                                adjustments_percent[scenario][region][country][tax] = tax_adjustment/fiscal_gap # is the same as fiscal_gap/fiscal_gap
                            tax_contribution[scenario][region][country][tax] = current_tax_income

    return fiscal_gaps, adjustments, adjustments_percent, tax_contribution



def sum_and_assign_weights(dic):
    """
    Calculates and assigns weights as a percentage of the total values in each category to the items in the dictionary.

    This function iterates over a dictionary where each key is a category and its value is a nested dictionary of technologies,
    which themselves contain nested dictionaries with 'value' entries. It computes the total value per category, assigns weights
    based on these values, and ensures that the sum of weights in each category is approximately 1 (with a tolerance for floating-point errors).

    Args:
        dic (dict): A dictionary where keys are categories and values are nested dictionaries of technologies. Each technology
                    is a dictionary of fuels, where each fuel is a dictionary containing at least a 'value' key.

    Returns:
        None: Modifies the dictionary in place by adding a 'factor_fiscal_gap' key to the fuel dictionaries that denotes the weight.

    Raises:
        SystemExit: If the total weights in any category do not sum to approximately 1, indicating an inconsistency in weight calculations.
    """
    tax_exceptions = ['IC', 'IMESI_Combust', 'IVA_Gasoil', 'Otros_Gasoil']
    for category, techs in dic.items():
        if category in tax_exceptions:
            continue
        # Calculate the total sum of values for each category
        total_per_category = sum(
            details['value'] for tech in techs.values() for fuel, details in tech.items() if isinstance(details, dict) and 'value' in details
        )
        
        # Now, assign weights within each category
        for tech, fuels in list(techs.items()):
            if not fuels:  # Check if the sub-dictionary is empty
                del techs[tech]
                continue
            for fuel, details in list(fuels.items()):
                if isinstance(details, dict) and 'value' in details:
                    weight = details['value'] / total_per_category  # Calculate the weight as a percentage
                    details['factor_fiscal_gap'] = weight  # Add the weight to the existing dictionary
                elif not details:  # If the details are an empty dictionary
                    del fuels[fuel]

        # Check the sum of weights for each category
        all_weights = [details['factor_fiscal_gap'] for tech in techs.values() for fuel, details in tech.items() if 'factor_fiscal_gap' in details]
        total_weights = sum(all_weights)
        if abs(total_weights - 1) > 0.01:  # Allow a small tolerance for floating point errors
            print(f"Error in category {category}: The sum of all weights is {total_weights}, not 1.")
            sys.exit(1)
        else:
            pass
    


def calculate_tax_impact(dict_activity_distribution_gap, new_tax_contribution):
    """
    Calculates the tax impact for various fuels based on their fiscal gap factors, tax percentages, and a new tax contribution value.

    This function iterates through a dictionary structure that includes categories, technologies, and fuels. It uses the factor fiscal gap,
    tax percentage, and value of each fuel to compute the tax impact, which is then stored back in the dictionary under a new key 'tax_impact'.

    Args:
        dict_activity_distribution_gap (dict): A nested dictionary containing categories, technologies, and fuels. Each fuel entry should
                                               include 'value', 'factor_fiscal_gap', and 'tax_percentage' keys.
        new_tax_contribution (float): The new tax contribution amount which is used to calculate the tax impacts.

    Returns:
        None: Modifies the dictionary in place by adding a 'tax_impact' key to each fuel dictionary.
    """
    tax_exceptions = ['IC', 'IMESI_Combust', 'IVA_Gasoil', 'Otros_Gasoil']
    for category, techs in dict_activity_distribution_gap.items():
        if category in tax_exceptions:
            continue
        for tech, fuels in techs.items():
            for fuel, details in fuels.items():
                if isinstance(details, dict) and 'value' in details and 'factor_fiscal_gap' in details and 'tax_percentage' in details:
                    impact = (details['factor_fiscal_gap'] * details['tax_percentage'] * new_tax_contribution) / details['value']
                    details['tax_impact'] = impact


def apply_tax_impact_to_structure(unit_taxes, unit_taxes_percentage, dict_activity_distribution_gap, scenario, region, country, time_vector, year_index, contribution_tax_actual_0, params_tier2, tax_id, dict_mult_depr):
    """
    Applies calculated tax impacts to a structured data dictionary for specific scenario, region, and country over a timeline.

    This function updates a structured data dictionary ('unit_taxes') with tax impact values for each tax type, technology, and fuel.
    It supports interpolation of tax impact values based on a defined interpolation strategy from initial year contributions.

    Args:
        unit_taxes (dict): A complex nested dictionary where tax impacts are stored for each scenario, region, country, tax, technology, and fuel.
        dict_activity_distribution_gap (dict): A dictionary similar to the one used in calculate_tax_impact with the added 'tax_impact' key for each fuel.
        scenario (str): The scenario under which the tax impacts are applied.
        region (str): The region under which the tax impacts are applied.
        country (str): The country under which the tax impacts are applied.
        time_vector (list): A list of years for which data is available and where tax impacts are to be applied.
        year_index (int): The index in 'time_vector' that corresponds to the current year for which tax impacts are being applied.
        contribution_tax_actual_0 (dict): A dictionary containing initial year tax contributions for interpolation.
        params_tier2 (dict): A dictionary containing parameters like interpolation strategy and start year for interpolation.

    Returns:
        None: Directly modifies the 'unit_taxes' dictionary to include tax impact values across the defined timeline.
    """
    tax_exceptions = ['IC', 'IMESI_Combust', 'IVA_Gasoil', 'Otros_Gasoil']
    fuel_exceptions = ['ELECTRICIDAD', 'HIDROGENO']
    
    # Define the inital year to interpolation
    index_initial_year_interpolation = time_vector.index(params_tier2['year_ini_inter'])
    
    for tax, techs in dict_activity_distribution_gap.items():
        for tech, fuels in techs.items():
            for fuel, details in fuels.items():
                if 'tax_impact' in details:
                    unit_tax = details['tax_impact']
                    unit_tax_percentage = details['tax_impact']/dict_mult_depr[tax][tech][fuel]
                    activity = details['value']
                    values_list = ['' for year in time_vector]
                    values_list[year_index] = unit_tax
                    values_list_percentage = ['' for year in time_vector]
                    values_list_percentage[year_index] = unit_tax_percentage
                    incomes_list = ['' for year in time_vector]
                    incomes_list[year_index] = unit_tax * activity
                    
                    if params_tier2['interpolate']:
                        values_list[index_initial_year_interpolation] = contribution_tax_actual_0[(tax, tech, fuel)]
                        values_list_percentage[index_initial_year_interpolation] = contribution_tax_actual_0[(tax, tech, fuel)]
                        incomes_list[index_initial_year_interpolation] = contribution_tax_actual_0[(f'Transport Tax {tax} [$]', tech, fuel)]
                        
                        # for y in range(0,index_initial_year_interpolation):
                        #     values_list[y] = values_list[index_initial_year_interpolation]
                        #     values_list_percentage[y] = values_list_percentage[index_initial_year_interpolation]
                        #     incomes_list[y] = incomes_list[index_initial_year_interpolation]
                        
                        values_list = interpolation_to_end(time_vector, \
                            params_tier2['year_ini_inter'], values_list, 'interme', 'Population')
                        values_list_percentage = interpolation_to_end(time_vector, \
                            params_tier2['year_ini_inter'], values_list_percentage, 'interme', 'Population')
                        incomes_list = interpolation_to_end(time_vector, \
                            params_tier2['year_ini_inter'], incomes_list, 'interme', 'Population')
                    
                    if tax in tax_exceptions and fuel in fuel_exceptions:
                        continue
                    elif tax in tax_exceptions and fuel not in fuel_exceptions:
                        unit_taxes[scenario][region][country][tax][tech][fuel] = values_list
                        unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = incomes_list
                        unit_taxes_percentage[scenario][region][country][tax][tech][fuel] = values_list_percentage
                        unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = incomes_list
                    elif tax == tax_id and tax not in tax_exceptions and scenario == 'NETZERO':
                        unit_taxes[scenario][region][country][tax][tech][fuel] = values_list
                        unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = incomes_list
                        unit_taxes_percentage[scenario][region][country][tax][tech][fuel] = values_list_percentage
                        unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = incomes_list
                    elif tax == tax_id and tax in tax_exceptions and scenario != 'NETZERO':
                        unit_taxes[scenario][region][country][tax][tech][fuel] = values_list
                        unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = incomes_list
                        unit_taxes_percentage[scenario][region][country][tax][tech][fuel] = values_list_percentage
                        unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = incomes_list
                    elif tax == tax_id and tax not in tax_exceptions and scenario != 'NETZERO':
                        unit_taxes[scenario][region][country][tax][tech][fuel] = values_list
                        unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = incomes_list
                        unit_taxes_percentage[scenario][region][country][tax][tech][fuel] = values_list_percentage
                        unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = incomes_list
                        


def calculate_unit_tax_for_all_scenarios(year, dict_scen, dict_tax_scen, dict_activities_out, time_vector, params_tier2, dict_mult_depr):
    """
    Calculates the unit tax for all scenarios based on actual tax amounts and activity data, incorporating optional interpolation.
    This function processes multiple scenarios comprehensively, applying the same calculation logic to each without performing any
    gap calculations or adjustments specific to differences between scenarios.

    Args:
        year (int): The target year for which the unit tax calculations are performed.
        dict_scen (dict): A dictionary containing the activity data for all scenarios, structured by scenario, region, and country.
        dict_tax_scen (dict): A dictionary containing the actual tax amounts for all scenarios, also structured by scenario, region, and country.
        dict_activities_out (dict): A dictionary detailing the relationships between tax types and the associated activity data under various scenarios.
        time_vector (list): A comprehensive list of all years for which data is available, used to determine indices for calculations.
        params_tier2 (dict): Parameters dictating secondary processing behaviors such as interpolation methods and settings.

    Returns:
        dict: A structured dictionary containing calculated unit taxes for each scenario, organized by scenario, region, country, tax type, technology, and fuel.
              Each entry includes a list of unit taxes for each year, potentially interpolated based on the settings provided.

    Raises:
        ValueError: Raises a ValueError if the specified year is not found in the time_vector, indicating a potential data availability issue.
    """
    # Define the inital year to interpolation
    index_initial_year_interpolation = time_vector.index(params_tier2['year_ini_inter'])
    
    try:
        year_index = time_vector.index(year)
    except ValueError:
        print(f"The year {year} is not found in the time_vector.")
        return {}

    unit_taxes = {}
    unit_taxes_percentage = {}
    for scenario, region_country_data in dict_scen.items():
        unit_taxes[scenario] = {}
        unit_taxes_percentage[scenario] = {}
        for region, countries in region_country_data.items():
            unit_taxes[scenario][region] = {}
            unit_taxes_percentage[scenario][region] = {}
            for country, taxes in countries.items():
                unit_taxes[scenario][region][country] = {}
                unit_taxes_percentage[scenario][region][country] = {}
                for tax, techs_types in dict_activities_out['Relation Tax-Activity2'].items():
                    unit_taxes[scenario][region][country][tax] = {}
                    unit_taxes_percentage[scenario][region][country][tax] = {}
                    for tech, fuel_types in techs_types.items():
                        unit_taxes[scenario][region][country][tax][tech] = {}
                        unit_taxes_percentage[scenario][region][country][tax][tech] = {}
                        for fuel, cost_type in fuel_types.items():
                            scenario_data = dict_scen[scenario][region][country]
                            tax_data = dict_tax_scen[scenario][region][country][tax][tech][fuel]
                            unit_convert_percentage = dict_mult_depr[tax][tech][fuel]
                            if float(unit_convert_percentage) == float(0):
                                unit_convert_percentage = 1
                                
                            values_list = ['' for _ in time_vector]
                            values_list_percentage = ['' for _ in time_vector]
                            
                            for year in range(len(time_vector)):

                                if cost_type == 'CapitalCost':
                                    activity_amount = scenario_data['New Fleet'][tech][fuel][year]
                                    # if params_tier2['interpolate']:
                                    #     activity_amount_0 = scenario_data['New Fleet'][tech][fuel][index_initial_year_interpolation]
                                elif cost_type == 'CapitalCost*':
                                    activity_amount = scenario_data['Fleet'][tech][fuel][year]
                                    # if params_tier2['interpolate']:
                                    #     activity_amount_0 = scenario_data['Fleet'][tech][fuel][index_initial_year_interpolation]
                                elif cost_type == 'KmCost':
                                    activity_amount = scenario_data['Fleet'][tech][fuel][year]
                                    # if params_tier2['interpolate']:
                                    #     activity_amount_0 = scenario_data['Fleet'][tech][fuel][index_initial_year_interpolation]
                                else:
                                    activity_amount = scenario_data['Fuel Consumption'][tech][fuel][year]
                                    conversion_factor = scenario_data['Conversion Fuel Constant'][tech][fuel]
                                    activity_amount /= conversion_factor
                                    # if params_tier2['interpolate']:
                                    #     activity_amount_0 = scenario_data['Fuel Consumption'][tech][fuel][index_initial_year_interpolation]
                                    #     activity_amount_0 /= conversion_factor
    
                                tax_actual = tax_data[year]
                                unit_tax = tax_actual / activity_amount if activity_amount != 0 else 0
                                unit_tax_percentage = tax_actual / (activity_amount*unit_convert_percentage) if activity_amount != 0 else 0
    
                                # if scenario == 'NETZERO' and tech == 'CamionesA' and fuel == 'GASOLINA/ALCOHOL' and tax == 'IVA_Elec':
                                #     print(scenario)
                                #     print(tech)
                                #     print('unit_tax:',unit_tax)
                                #     print('unit_tax_percentage:',unit_tax_percentage)
                                #     print('factores:',activity_amount,unit_convert_percentage)
                                    # sys.exit(9)
    
                                # values_list = ['' for _ in time_vector]
                                values_list[year] = unit_tax
                                # values_list_percentage = ['' for _ in time_vector]
                                values_list_percentage[year] = unit_tax_percentage

                            # tax_actual_0 = tax_data[index_initial_year_interpolation]
                            # unit_tax_0 = tax_actual_0 / activity_amount_0 if activity_amount_0 != 0 else 0
                            # values_list[index_initial_year_interpolation] = unit_tax_0
                            # unit_tax_0_percentage = tax_actual_0 / (activity_amount_0*unit_convert_percentage) if activity_amount_0 != 0 else 0
                            # values_list_percentage[index_initial_year_interpolation] = unit_tax_0_percentage
                            
                            # for y in range(0,index_initial_year_interpolation):
                            #     values_list[y] = values_list[index_initial_year_interpolation]
                            #     values_list_percentage[y] = values_list_percentage[index_initial_year_interpolation]
                            
                            # Call to a placeholder function for actual interpolation
                            # values_list = interpolation_to_end(time_vector, params_tier2['year_ini_inter'], values_list, 'last', 'Population')
                            # values_list_percentage = interpolation_to_end(time_vector, params_tier2['year_ini_inter'], values_list_percentage, 'last', 'Population')

                            unit_taxes[scenario][region][country][tax][tech][fuel] = values_list
                            unit_taxes_percentage[scenario][region][country][tax][tech][fuel] = values_list_percentage

                            # if scenario == 'ACELERADO' and tax == 'Patente' and tech == 'Buses MVD' and fuel == 'DIESEL OIL':
                            #     print(1)
                            #     print(values_list)
                            #     print(values_list_percentage)



    return unit_taxes, unit_taxes_percentage



# one tax by time
def calculate_unit_tax_one_tax_by_time(year, dict_scen, adjustments_percent, dict_tax_scen, dict_activities_out, time_vector, fiscal_gaps, tax_contribution, params_tier2, tax_id, dict_mult_depr, base_dict_scen):
    """
    Calculates the unit tax for all scenarios for a given year, accounting for adjustments in tax percentages, 
    fiscal gaps, and initial tax contributions, with an option to interpolate tax values across different years.

    This function iterates through each scenario, excluding the Business As Usual (BAU), and calculates the unit tax 
    by considering changes in tax amounts due to fiscal adjustments and varying levels of economic activities.

    Args:
        year (int): The year for which the unit tax is calculated.
        dict_scen (dict): A dictionary containing scenario data with activity amounts for various technologies and fuels.
        adjustments_percent (dict): A dictionary detailing percentage adjustments for tax contributions for each scenario.
        dict_tax_scen (dict): A dictionary containing actual tax amounts for each scenario.
        dict_activities_out (dict): A dictionary describing the relation between tax types and activity data.
        time_vector (list): List of all years available for analysis.
        fiscal_gaps (dict): A dictionary containing fiscal gap amounts for each scenario.
        tax_contribution (dict): A dictionary detailing initial tax contributions for each tax type and scenario.
        params_tier2 (dict): Dictionary containing parameters for tier 2 processes including interpolation settings.
        tax_id (string): String with the Adjust ID value.

    Returns:
        dict: A structured dictionary containing calculated unit taxes for each scenario, organized by region, country, 
              tax type, technology, and fuel. Each entry includes a list of unit taxes for each year, potentially interpolated 
              based on the settings provided.

    Raises:
        ValueError: If the specified year is not found in the time_vector, a ValueError is raised indicating the data issue.
    """
    tax_exceptions = ['IC', 'IMESI_Combust', 'IVA_Gasoil', 'Otros_Gasoil']
    fuel_exceptions = ['ELECTRICIDAD', 'HIDROGENO']
    
    # Define the inital year to interpolation
    index_initial_year_interpolation = time_vector.index(params_tier2['year_ini_inter'])
    
    try:
        year_index = time_vector.index(year)
    except ValueError:
        print(f"The year {year} is not found in the time_vector.")
        return {}
    
    unit_taxes = {}
    unit_taxes_percentage = {}
    
    for scenario in adjustments_percent:
        if scenario != 'BAU':
            unit_taxes[scenario] = {}
            unit_taxes_percentage[scenario] = {}            
            for region in adjustments_percent[scenario]:
                unit_taxes[scenario][region] = {}
                unit_taxes_percentage[scenario][region] = {}
                
                for country in adjustments_percent[scenario][region]:
                    unit_taxes[scenario][region][country] = {}
                    unit_taxes_percentage[scenario][region][country] = {}
                    scenario_data = dict_scen[scenario][region][country]
                    dict_activity_distribution_gap = {}
                    contribution_tax_actual_0 = {}
                    
                    tax_adjustment_percent_exceptions = 0
                    # Identify the last tax
                    last_tax = list(dict_activities_out['Relation Tax-Activity2'].keys())[-1]
                    
                    for tax, techs_types in dict_activities_out['Relation Tax-Activity2'].items():
                        unit_taxes[scenario][region][country][tax] = {}
                        unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'] = {}
                        unit_taxes_percentage[scenario][region][country][tax] = {}
                        unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'] = {}
                        if tax_contribution[scenario][region][country][tax] == 0:
                            dict_activity_distribution_gap[tax] = {}
                        
                        for tech, fuel_types in techs_types.items():
                            unit_taxes[scenario][region][country][tax][tech] = {}
                            unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech] = {}
                            unit_taxes_percentage[scenario][region][country][tax][tech] = {}
                            unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech] = {}
                            if tax_contribution[scenario][region][country][tax] == 0:
                                dict_activity_distribution_gap[tax][tech] = {}
                            
                            for fuel, cost_type in fuel_types.items():
                                unit_taxes[scenario][region][country][tax][tech][fuel] = {}
                                unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = {}
                                unit_taxes_percentage[scenario][region][country][tax][tech][fuel] = {}
                                unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = {}
                                unit_convert_percentage = dict_mult_depr[tax][tech][fuel]
                                if float(unit_convert_percentage) == float(0):
                                    unit_convert_percentage = 1
                                if tax_contribution[scenario][region][country][tax] == 0:
                                    dict_activity_distribution_gap[tax][tech][fuel] = {}
                                    
                                values_list = ['' for year in time_vector]
                                tax_adjustment_percent = adjustments_percent[scenario][region][country].get(tax, 0)
                                # Calculate tax_adjustment_percent_exceptions
                                # if scenario == 'NETZERO' and tax in tax_exceptions:
                                #     # tax_adjustment_percent_exceptions += tax_adjustment_percent - 1
                                #     tax_adjustment_percent_exceptions += tax_adjustment_percent
                                    
                                values_list_base = deepcopy(base_dict_scen[scenario][region][country][tax][tech][fuel])

                                
                                if cost_type == 'CapitalCost':
                                    activity_amount = scenario_data['New Fleet'][tech][fuel][year_index]
                                    activity_base = scenario_data['New Fleet'][tech][fuel]
                                elif cost_type == 'CapitalCost*':
                                    activity_amount = scenario_data['Fleet'][tech][fuel][year_index]
                                    activity_base = scenario_data['Fleet'][tech][fuel]
                                elif cost_type == 'KmCost':
                                    activity_amount = scenario_data['Fleet'][tech][fuel][year_index]
                                    activity_base = scenario_data['Fleet'][tech][fuel]
                                else:
                                    conversion_factors = scenario_data['Conversion Fuel Constant'][tech][fuel]
                                    activity_amount = scenario_data['Fuel Consumption'][tech][fuel][year_index]
                                    activity_amount /= conversion_factors
                                    activity_base = scenario_data['Fuel Consumption'][tech][fuel]
                                    activity_base = [x / conversion_factors for x in activity_base]
        
                                
                                tax_actual = dict_tax_scen[scenario][region][country][tax][tech][fuel][year_index]
                                
                                if float(activity_amount) > 0.0 or float(activity_amount) < 0.0:
                                    if tax == tax_id:
                                        # unit_tax = ((tax_adjustment_percent-1) * tax_actual) / activity_amount
                                        unit_tax = ((tax_adjustment_percent) * tax_actual) / activity_amount
                                    else:
                                        unit_tax = 0.0
                                else:
                                    if tax == tax_id:
                                        # unit_tax = ((tax_adjustment_percent-1) * tax_actual)  # Assign full adjustment if there is no activity
                                        unit_tax = ((tax_adjustment_percent) * tax_actual)  # Assign full adjustment if there is no activity
                                    else:
                                        unit_tax = 0.0


                                
                                values_list[year_index] = unit_tax
                                # if tax in tax_exceptions and scenario == 'NETZERO':
                                #     values_list[year_index] = 0.0
                                #     values_list_base[year_index] = 0.0
                                     
                                # if tax in tax_exceptions and fuel in fuel_exceptions and unit_tax != np.float64(0.0):
                                #     values_list[year_index] = 0.0
                                # elif tax in tax_exceptions and fuel in fuel_exceptions and unit_tax == np.float64(0.0):
                                #     values_list[year_index] = unit_tax
                                # else:
                                #     values_list[year_index] = unit_tax

                                

                                values_list[index_initial_year_interpolation] = 0.0
                                for y in range(0,index_initial_year_interpolation):
                                    values_list[y] = 0
                                
                                values_list = interpolation_to_end(time_vector, \
                                    params_tier2['year_ini_inter'], values_list, 'interme', 'Population')
                                values_list_final = [a + b for a, b in zip(values_list_base, values_list)]
                
                                
                                unit_taxes[scenario][region][country][tax][tech][fuel] = values_list_final
                                unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = [a * b for a, b in zip(values_list_final, activity_base)]
                                unit_taxes_percentage[scenario][region][country][tax][tech][fuel] = [x / unit_convert_percentage for x in values_list_final]
                                unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = [a * b for a, b in zip(values_list_final, activity_base)]


                                # if scenario == "NETZERO" and tax == last_tax and not tax in tax_exceptions:
                                #     if float(unit_convert_percentage) == float(0):
                                #         unit_convert_percentage = 1
                                #     if tax_contribution[scenario][region][country][tax_id] == 0:
                                #         dict_activity_distribution_gap[tax_id][tech][fuel] = {}
                                
                                #     values_list = ['' for year in time_vector]
                                #     tax_adjustment_percent = adjustments_percent[scenario][region][country].get(tax_id, 0)
                                    
                                #     # Añadir tax_adjustment_percent_exceptions al ajuste de porcentaje
                                #     tax_adjustment_percent += tax_adjustment_percent_exceptions
                                    
                                    
                                    
                                    
                                    
                                    
                                
                                #     # values_list_base = deepcopy(base_dict_scen[scenario][region][country][tax_id][tech][fuel])
                                    
                                #     if cost_type == 'CapitalCost':
                                #         activity_amount = scenario_data['New Fleet'][tech][fuel][year_index]
                                #         activity_base = scenario_data['New Fleet'][tech][fuel]
                                #     elif cost_type == 'CapitalCost*':
                                #         activity_amount = scenario_data['Fleet'][tech][fuel][year_index]
                                #         activity_base = scenario_data['Fleet'][tech][fuel]
                                #     elif cost_type == 'KmCost':
                                #         activity_amount = scenario_data['Fleet'][tech][fuel][year_index]
                                #         activity_base = scenario_data['Fleet'][tech][fuel]
                                #     else:
                                #         conversion_factors = scenario_data['Conversion Fuel Constant'][tech][fuel]
                                #         activity_amount = scenario_data['Fuel Consumption'][tech][fuel][year_index]
                                #         activity_amount /= conversion_factors
                                #         activity_base = scenario_data['Fuel Consumption'][tech][fuel]
                                #         activity_base = [x / conversion_factors for x in activity_base]
                                
                                #     tax_actual = dict_tax_scen[scenario][region][country][tax_id][tech][fuel][year_index]
                                
                                #     if float(activity_amount) > 0.0 or float(activity_amount) < 0.0:
                                #         # unit_tax = ((tax_adjustment_percent - 1) * tax_actual) / activity_amount
                                #         unit_tax = ((tax_adjustment_percent) * tax_actual) / activity_amount
                                #     else:
                                #         # unit_tax = ((tax_adjustment_percent - 1) * tax_actual)
                                #         unit_tax = ((tax_adjustment_percent) * tax_actual)
                                
                                #     values_list[year_index] = unit_tax
                                #     values_list[index_initial_year_interpolation] = 0.0
                                #     for y in range(0, index_initial_year_interpolation):
                                #         values_list[y] = 0
                                
                                #     values_list = interpolation_to_end(
                                #         time_vector, params_tier2['year_ini_inter'], values_list, 'interme', 'Population'
                                #     )
                                #     values_list_final = [a + b for a, b in zip(values_list_base, values_list)]
                                
                                #     unit_taxes[scenario][region][country][tax_id][tech][fuel] = values_list_final
                                #     unit_taxes[scenario][region][country][f'Transport Tax {tax_id} [$]'][tech][fuel] = [
                                #         a * b for a, b in zip(values_list_final, activity_base)
                                #     ]
                                #     unit_taxes_percentage[scenario][region][country][tax_id][tech][fuel] = [
                                #         x / unit_convert_percentage for x in values_list_final
                                #     ]
                                #     unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax_id} [$]'][tech][fuel] = [
                                #         a * b for a, b in zip(values_list_final, activity_base)
                                #     ]


    return unit_taxes, unit_taxes_percentage, dict_activity_distribution_gap






# # one tax by time
# def calculate_unit_tax_one_tax_by_time(year, dict_scen, adjustments_percent, dict_tax_scen, dict_activities_out, time_vector, fiscal_gaps, tax_contribution, params_tier2, tax_id, dict_mult_depr, base_dict_scen):
#     """
#     Calculates the unit tax for all scenarios for a given year, accounting for adjustments in tax percentages, 
#     fiscal gaps, and initial tax contributions, with an option to interpolate tax values across different years.

#     This function iterates through each scenario, excluding the Business As Usual (BAU), and calculates the unit tax 
#     by considering changes in tax amounts due to fiscal adjustments and varying levels of economic activities.

#     Args:
#         year (int): The year for which the unit tax is calculated.
#         dict_scen (dict): A dictionary containing scenario data with activity amounts for various technologies and fuels.
#         adjustments_percent (dict): A dictionary detailing percentage adjustments for tax contributions for each scenario.
#         dict_tax_scen (dict): A dictionary containing actual tax amounts for each scenario.
#         dict_activities_out (dict): A dictionary describing the relation between tax types and activity data.
#         time_vector (list): List of all years available for analysis.
#         fiscal_gaps (dict): A dictionary containing fiscal gap amounts for each scenario.
#         tax_contribution (dict): A dictionary detailing initial tax contributions for each tax type and scenario.
#         params_tier2 (dict): Dictionary containing parameters for tier 2 processes including interpolation settings.
#         tax_id (string): String with the Adjust ID value.

#     Returns:
#         dict: A structured dictionary containing calculated unit taxes for each scenario, organized by region, country, 
#               tax type, technology, and fuel. Each entry includes a list of unit taxes for each year, potentially interpolated 
#               based on the settings provided.

#     Raises:
#         ValueError: If the specified year is not found in the time_vector, a ValueError is raised indicating the data issue.
#     """
#     tax_exceptions = ['IC', 'IMESI_Combust', 'IVA_Gasoil', 'Otros_Gasoil']
#     fuel_exceptions = ['ELECTRICIDAD', 'HIDROGENO']
    
#     # Define the inital year to interpolation
#     index_initial_year_interpolation = time_vector.index(params_tier2['year_ini_inter'])
    
#     try:
#         year_index = time_vector.index(year)
#     except ValueError:
#         print(f"The year {year} is not found in the time_vector.")
#         return {}
    
#     unit_taxes = {}
#     unit_taxes_percentage = {}
    
#     for scenario in adjustments_percent:
#         if scenario != 'BAU':
#             unit_taxes[scenario] = {}
#             unit_taxes_percentage[scenario] = {}            
#             for region in adjustments_percent[scenario]:
#                 unit_taxes[scenario][region] = {}
#                 unit_taxes_percentage[scenario][region] = {}
                
#                 for country in adjustments_percent[scenario][region]:
#                     unit_taxes[scenario][region][country] = {}
#                     unit_taxes_percentage[scenario][region][country] = {}
#                     scenario_data = dict_scen[scenario][region][country]
#                     dict_activity_distribution_gap = {}
#                     contribution_tax_actual_0 = {}
#                     for tax, techs_types in dict_activities_out['Relation Tax-Activity2'].items():
#                         unit_taxes[scenario][region][country][tax] = {}
#                         unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'] = {}
#                         unit_taxes_percentage[scenario][region][country][tax] = {}
#                         unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'] = {}
#                         if tax_contribution[scenario][region][country][tax] == 0:
#                             dict_activity_distribution_gap[tax] = {}
                        
#                         for tech, fuel_types in techs_types.items():
#                             unit_taxes[scenario][region][country][tax][tech] = {}
#                             unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech] = {}
#                             unit_taxes_percentage[scenario][region][country][tax][tech] = {}
#                             unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech] = {}
#                             if tax_contribution[scenario][region][country][tax] == 0:
#                                 dict_activity_distribution_gap[tax][tech] = {}
                            
#                             for fuel, cost_type in fuel_types.items():
#                                 unit_taxes[scenario][region][country][tax][tech][fuel] = {}
#                                 unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = {}
#                                 unit_taxes_percentage[scenario][region][country][tax][tech][fuel] = {}
#                                 unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = {}
#                                 unit_convert_percentage = dict_mult_depr[tax][tech][fuel]
#                                 if float(unit_convert_percentage) == float(0):
#                                     unit_convert_percentage = 1
#                                 if tax_contribution[scenario][region][country][tax] == 0:
#                                     dict_activity_distribution_gap[tax][tech][fuel] = {}
                                    
#                                 values_list = ['' for year in time_vector]
#                                 tax_adjustment_percent = adjustments_percent[scenario][region][country].get(tax, 0)
                                
#                                 values_list_base = deepcopy(base_dict_scen[scenario][region][country][tax][tech][fuel])
                                
#                                 # for year in range(len(time_vector)):
                                
#                                 if cost_type == 'CapitalCost':
#                                     activity_amount = scenario_data['New Fleet'][tech][fuel][year_index]
#                                     activity_base = scenario_data['New Fleet'][tech][fuel]
#                                 elif cost_type == 'CapitalCost*':
#                                     activity_amount = scenario_data['Fleet'][tech][fuel][year_index]
#                                     activity_base = scenario_data['Fleet'][tech][fuel]
#                                 elif cost_type == 'KmCost':
#                                     activity_amount = scenario_data['Fleet'][tech][fuel][year_index]
#                                     activity_base = scenario_data['Fleet'][tech][fuel]
#                                 else:
#                                     conversion_factors = scenario_data['Conversion Fuel Constant'][tech][fuel]
#                                     activity_amount = scenario_data['Fuel Consumption'][tech][fuel][year_index]
#                                     activity_amount /= conversion_factors
#                                     activity_base = scenario_data['Fuel Consumption'][tech][fuel]
#                                     activity_base = [x / conversion_factors for x in activity_base]
        
                                
#                                 tax_actual = dict_tax_scen[scenario][region][country][tax][tech][fuel][year_index]
#                                 # tax_actual_0 = dict_tax_scen[scenario][region][country][tax][tech][fuel][index_initial_year_interpolation]
                                
#                                 # if float(tax_contribution[scenario][region][country][tax]) == 0.0 and float(activity_amount) != 0.0:                                    
#                                 #     new_tax_contribution = fiscal_gaps[scenario][region][country]
#                                 #     dict_activity_distribution_gap[tax][tech][fuel]['value'] = activity_amount
#                                 #     dict_activity_distribution_gap[tax][tech][fuel]['tax_percentage'] = tax_adjustment_percent
                                
#                                 if float(activity_amount) > 0.0 or float(activity_amount) < 0.0:
#                                     if tax == tax_id:
#                                         # unit_tax = (tax_adjustment_percent * tax_actual) / activity_amount
#                                         unit_tax = ((tax_adjustment_percent-1) * tax_actual) / activity_amount
#                                     else:
#                                         # unit_tax = tax_actual / activity_amount
#                                         unit_tax = 0.0
#                                 else:
#                                     if tax == tax_id:
#                                         unit_tax = ((tax_adjustment_percent-1) * tax_actual)  # Assign full adjustment if there is no activity
#                                     else:
#                                         # unit_tax = tax_actual
#                                         unit_tax = 0.0
                                    
#                                 # if (float(activity_amount_0) > 0.0 or float(activity_amount_0) < 0.0):
#                                 #     contribution_tax_actual_0[(tax, tech, fuel)] = 0.0
#                                 #     contribution_tax_actual_0[(f'Transport Tax {tax} [$]', tech, fuel)] = tax_actual_0
#                                 # else:
#                                 #     if tax_actual_0 != 0 or float(tax_actual_0) != 0.0:
#                                 #         contribution_tax_actual_0[(tax, tech, fuel)] = 0.0
#                                 #         contribution_tax_actual_0[(f'Transport Tax {tax} [$]', tech, fuel)] = tax_actual_0
#                                 #     else:
#                                 #         contribution_tax_actual_0[(tax, tech, fuel)] = 0.0
#                                 #         contribution_tax_actual_0[(f'Transport Tax {tax} [$]', tech, fuel)] = tax_actual_0
#                                 #         # Because tax_actual_0=0 and activity_amount_0=0 so,
#                                 #         # 0/0 isn't posible, but if is assing 1, before when
#                                 #         # apply the formula tax_actual_0=unit_tax_0*activity_amount
#                                 #         # tax_actual_0=1*0=0

                                
#                                 values_list[year_index] = unit_tax
#                                 if tax in tax_exceptions and scenario == 'NETZERO':
#                                     values_list[year_index] = 0.0
#                                     values_list_base[year_index] = 0.0
                                     
#                                 elif tax in tax_exceptions and fuel in fuel_exceptions and unit_tax != np.float64(0.0):
#                                     values_list[year_index] = 0.0
#                                 elif tax in tax_exceptions and fuel in fuel_exceptions and unit_tax == np.float64(0.0):
#                                     values_list[year_index] = unit_tax
#                                 else:
#                                     values_list[year_index] = unit_tax

                                

#                                 values_list[index_initial_year_interpolation] = 0.0
#                                 for y in range(0,index_initial_year_interpolation):
#                                     values_list[y] = 0
                                
#                                 values_list = interpolation_to_end(time_vector, \
#                                     params_tier2['year_ini_inter'], values_list, 'interme', 'Population')
#                                 values_list_final = [a + b for a, b in zip(values_list_base, values_list)]
                
                                
#                                 unit_taxes[scenario][region][country][tax][tech][fuel] = values_list_final
#                                 unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = [a * b for a, b in zip(values_list_final, activity_base)]
#                                 unit_taxes_percentage[scenario][region][country][tax][tech][fuel] = [x / unit_convert_percentage for x in values_list_final]
#                                 unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = [a * b for a, b in zip(values_list_final, activity_base)]
                                
#                                 if tax in tax_exceptions and scenario == 'NETZERO':
#                                     values_list_check = ['' for year in time_vector]
                                    
#                                     # for y in range(index_initial_year_interpolation,len(time_vector)):
                                        
#                                     #     if cost_type == 'CapitalCost':
#                                     #         activity_amount = scenario_data['New Fleet'][tech][fuel][y]
#                                     #     elif cost_type == 'CapitalCost*':
#                                     #         activity_amount = scenario_data['Fleet'][tech][fuel][y]
#                                     #     elif cost_type == 'KmCost':
#                                     #         activity_amount = scenario_data['Fleet'][tech][fuel][y]
#                                     #     else:
#                                     #         conversion_factors = scenario_data['Conversion Fuel Constant'][tech][fuel]
#                                     #         activity_amount = scenario_data['Fuel Consumption'][tech][fuel][y]
#                                     #         activity_amount /= conversion_factors
                                        
#                                     #     activity_base[y] += activity_amount
                                    
#                                     # unit_tax = ((tax_adjustment_percent-1) * tax_actual) / activity_amount
                                    
#                                     # unit_tax_check = unit_taxes[scenario][region][country][tax_id][tech][fuel][year_index] + unit_tax
#                                     # values_list_check[year_index] = unit_tax
#                                     # values_list_check = interpolation_to_end(time_vector, \
#                                     #     params_tier2['year_ini_inter'], values_list_check, 'interme', 'Population')
                                        
#                                     if tax_id not in unit_taxes[scenario][region][country] and tax != tax_id:
#                                         adjust_check = (
#                                             adjustments_percent[scenario][region][country].get(tax_id, 0) + (tax_adjustment_percent - 1)
#                                         )
#                                         adjustments_percent[scenario][region][country][tax_id] = adjust_check
#                                     elif tax_id not in unit_taxes[scenario][region][country] and tax == tax_id:
#                                          adjust_check = (
#                                             adjustments_percent[scenario][region][country].get('IVA_Elec', 0) + (tax_adjustment_percent - 1)
#                                         )
#                                          adjustments_percent[scenario][region][country]['IVA_Elec'] = adjust_check

#                                     if tax_id in unit_taxes[scenario][region][country] and tax != tax_id:
#                                         # print(tax_id,adjustments_percent[scenario][region][country].get(tax_id, 0))
#                                         adjust_check = (
#                                             adjustments_percent[scenario][region][country].get(tax_id, 0) + (tax_adjustment_percent - 1)
#                                         )
                                        
                                        
                                        
#                                         if float(activity_amount) > 0.0 or float(activity_amount) < 0.0:
#                                             if tax == tax_id:
#                                                 # unit_tax = (tax_adjustment_percent * tax_actual) / activity_amount
#                                                 unit_tax = ((adjust_check-1) * tax_actual) / activity_amount
#                                             else:
#                                                 # unit_tax = tax_actual / activity_amount
#                                                 unit_tax = 0.0
#                                         else:
#                                             if tax == tax_id:
#                                                 unit_tax = ((adjust_check-1) * tax_actual)  # Assign full adjustment if there is no activity
#                                             else:
#                                                 # unit_tax = tax_actual
#                                                 unit_tax = 0.0
                                        
                                        
#                                         # import warnings

#                                         # # Convert RuntimeWarnings to exceptions
#                                         # warnings.simplefilter("error", RuntimeWarning)
                                        
#                                         # try:
#                                         #     unit_tax = ((adjust_check - 1) * tax_actual) / activity_amount
#                                         # except RuntimeWarning as e:
#                                         #     print(f"RuntimeWarning encountered: {e}")
#                                         #     print(f"Value of adjust_check: {adjust_check}")
#                                         #     print(f"Value of adjust percent original: {tax_adjustment_percent}")
#                                         #     print(f"Value of tax_actual: {tax_actual}")
#                                         #     print(f"Value of activity_amount: {activity_amount}")
#                                         #     sys.exit(1)
#                                         # except ZeroDivisionError:
#                                         #     print("Error: Division by zero detected.")
#                                         #     print(f"Value of activity_amount: {activity_amount}")
#                                         #     sys.exit(1)
                                        
#                                         # unit_tax_check = unit_taxes[scenario][region][country][tax_id][tech][fuel][year_index] + unit_tax
#                                         values_list_check[year_index] = unit_tax
#                                         values_list_check[index_initial_year_interpolation] = 0.0
#                                         for y in range(0,index_initial_year_interpolation):
#                                             values_list_check[y] = 0
#                                         values_list_check = interpolation_to_end(time_vector, \
#                                             params_tier2['year_ini_inter'], values_list_check, 'interme', 'Population')
#                                         values_list_final = [a + b for a, b in zip(values_list_base, values_list_check)]
                                        
#                                         unit_taxes[scenario][region][country][tax_id][tech][fuel] = values_list_check
#                                         unit_taxes[scenario][region][country][f'Transport Tax {tax_id} [$]'][tech][fuel] = [a * b for a, b in zip(values_list_check, activity_base)]
#                                         unit_taxes_percentage[scenario][region][country][tax_id][tech][fuel] = [x / unit_convert_percentage for x in values_list_check]
#                                         unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax_id} [$]'][tech][fuel] = [a * b for a, b in zip(values_list_check, activity_base)]    
                                    
#                     # # Perform operations to finalize tax calculations and adjustments
#                     # sum_and_assign_weights(dict_activity_distribution_gap)   
#                     # calculate_tax_impact(dict_activity_distribution_gap, new_tax_contribution)    
#                     # apply_tax_impact_to_structure(unit_taxes, unit_taxes_percentage, dict_activity_distribution_gap, scenario, region, country, time_vector, year_index, contribution_tax_actual_0, params_tier2, tax_id, dict_mult_depr)

#     return unit_taxes, unit_taxes_percentage, dict_activity_distribution_gap

# # one tax by time
# def calculate_unit_tax_one_tax_by_time(year, dict_scen, adjustments_percent, dict_tax_scen, dict_activities_out, time_vector, fiscal_gaps, tax_contribution, params_tier2, tax_id, dict_mult_depr, base_dict_scen):
#     """
#     Calculates the unit tax for all scenarios for a given year, accounting for adjustments in tax percentages, 
#     fiscal gaps, and initial tax contributions, with an option to interpolate tax values across different years.

#     This function iterates through each scenario, excluding the Business As Usual (BAU), and calculates the unit tax 
#     by considering changes in tax amounts due to fiscal adjustments and varying levels of economic activities.

#     Args:
#         year (int): The year for which the unit tax is calculated.
#         dict_scen (dict): A dictionary containing scenario data with activity amounts for various technologies and fuels.
#         adjustments_percent (dict): A dictionary detailing percentage adjustments for tax contributions for each scenario.
#         dict_tax_scen (dict): A dictionary containing actual tax amounts for each scenario.
#         dict_activities_out (dict): A dictionary describing the relation between tax types and activity data.
#         time_vector (list): List of all years available for analysis.
#         fiscal_gaps (dict): A dictionary containing fiscal gap amounts for each scenario.
#         tax_contribution (dict): A dictionary detailing initial tax contributions for each tax type and scenario.
#         params_tier2 (dict): Dictionary containing parameters for tier 2 processes including interpolation settings.
#         tax_id (string): String with the Adjust ID value.

#     Returns:
#         dict: A structured dictionary containing calculated unit taxes for each scenario, organized by region, country, 
#               tax type, technology, and fuel. Each entry includes a list of unit taxes for each year, potentially interpolated 
#               based on the settings provided.

#     Raises:
#         ValueError: If the specified year is not found in the time_vector, a ValueError is raised indicating the data issue.
#     """
#     tax_exceptions = ['IC', 'IMESI_Combust', 'IVA_Gasoil', 'Otros_Gasoil']
#     fuel_exceptions = ['ELECTRICIDAD', 'HIDROGENO']
    
#     # Define the inital year to interpolation
#     index_initial_year_interpolation = time_vector.index(params_tier2['year_ini_inter'])
    
#     try:
#         year_index = time_vector.index(year)
#     except ValueError:
#         print(f"The year {year} is not found in the time_vector.")
#         return {}
    
#     unit_taxes = {}
#     unit_taxes_percentage = {}
    
#     for scenario in adjustments_percent:
#         if scenario != 'BAU':
#             unit_taxes[scenario] = {}
#             unit_taxes_percentage[scenario] = {}            
#             for region in adjustments_percent[scenario]:
#                 unit_taxes[scenario][region] = {}
#                 unit_taxes_percentage[scenario][region] = {}
                
#                 for country in adjustments_percent[scenario][region]:
#                     unit_taxes[scenario][region][country] = {}
#                     unit_taxes_percentage[scenario][region][country] = {}
#                     scenario_data = dict_scen[scenario][region][country]
#                     dict_activity_distribution_gap = {}
#                     contribution_tax_actual_0 = {}
#                     for tax, techs_types in dict_activities_out['Relation Tax-Activity2'].items():
#                         unit_taxes[scenario][region][country][tax] = {}
#                         unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'] = {}
#                         unit_taxes_percentage[scenario][region][country][tax] = {}
#                         unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'] = {}
#                         if tax_contribution[scenario][region][country][tax] == 0:
#                             dict_activity_distribution_gap[tax] = {}
                        
#                         for tech, fuel_types in techs_types.items():
#                             unit_taxes[scenario][region][country][tax][tech] = {}
#                             unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech] = {}
#                             unit_taxes_percentage[scenario][region][country][tax][tech] = {}
#                             unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech] = {}
#                             if tax_contribution[scenario][region][country][tax] == 0:
#                                 dict_activity_distribution_gap[tax][tech] = {}
                            
#                             for fuel, cost_type in fuel_types.items():
#                                 unit_taxes[scenario][region][country][tax][tech][fuel] = {}
#                                 unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = {}
#                                 unit_taxes_percentage[scenario][region][country][tax][tech][fuel] = {}
#                                 unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = {}
#                                 unit_convert_percentage = dict_mult_depr[tax][tech][fuel]
#                                 if float(unit_convert_percentage) == float(0):
#                                     unit_convert_percentage = 1
#                                 if tax_contribution[scenario][region][country][tax] == 0:
#                                     dict_activity_distribution_gap[tax][tech][fuel] = {}
                                
#                                 if cost_type == 'CapitalCost':
#                                     activity_amount = scenario_data['New Fleet'][tech][fuel][year_index]
#                                     activity_base = scenario_data['New Fleet'][tech][fuel]
#                                     activity_amount_0 = scenario_data['New Fleet'][tech][fuel][index_initial_year_interpolation]
#                                 elif cost_type == 'CapitalCost*':
#                                     activity_amount = scenario_data['Fleet'][tech][fuel][year_index]
#                                     activity_base = scenario_data['Fleet'][tech][fuel]
#                                     activity_amount_0 = scenario_data['Fleet'][tech][fuel][index_initial_year_interpolation]
#                                 elif cost_type == 'KmCost':
#                                     activity_amount = scenario_data['Fleet'][tech][fuel][year_index]
#                                     activity_base = scenario_data['Fleet'][tech][fuel]
#                                     activity_amount_0 = scenario_data['Fleet'][tech][fuel][index_initial_year_interpolation]
#                                 else:
#                                     activity_amount = scenario_data['Fuel Consumption'][tech][fuel][year_index]
#                                     conversion_factors = scenario_data['Conversion Fuel Constant'][tech][fuel]
#                                     activity_amount /= conversion_factors

#                                     activity_amount_0 = scenario_data['Fuel Consumption'][tech][fuel][index_initial_year_interpolation]
#                                     activity_amount_0 /= conversion_factors                                
        
#                                 tax_adjustment_percent = adjustments_percent[scenario][region][country].get(tax, 0)
#                                 tax_actual = dict_tax_scen[scenario][region][country][tax][tech][fuel][year_index]
#                                 if params_tier2['interpolate']:
#                                     tax_actual_0 = dict_tax_scen[scenario][region][country][tax][tech][fuel][index_initial_year_interpolation]
                                
#                                 if float(tax_contribution[scenario][region][country][tax]) == 0.0 and float(activity_amount) != 0.0:                                    
#                                     new_tax_contribution = fiscal_gaps[scenario][region][country]
#                                     dict_activity_distribution_gap[tax][tech][fuel]['value'] = activity_amount
#                                     dict_activity_distribution_gap[tax][tech][fuel]['tax_percentage'] = tax_adjustment_percent
                                
#                                 if float(activity_amount) > 0.0 or float(activity_amount) < 0.0:
#                                     if tax == tax_id:
#                                         unit_tax = (tax_adjustment_percent * tax_actual) / activity_amount
#                                         unit_tax_percentage = (tax_adjustment_percent * tax_actual) / (activity_amount*unit_convert_percentage)
#                                     else:
#                                         unit_tax = tax_actual / activity_amount
#                                         unit_tax_percentage = tax_actual / (activity_amount*unit_convert_percentage)
#                                 else:
#                                     if tax == tax_id:
#                                         unit_tax = (tax_adjustment_percent * tax_actual)  # Assign full adjustment if there is no activity
#                                         unit_tax_percentage = (tax_adjustment_percent * tax_actual)/(unit_convert_percentage)
#                                     else:
#                                         unit_tax = tax_actual
#                                         unit_tax_percentage = tax_actual/(unit_convert_percentage)
                                    
#                                 if params_tier2['interpolate'] and (float(activity_amount_0) > 0.0 or float(activity_amount_0) < 0.0):
#                                     unit_tax_0 = tax_actual_0 / activity_amount_0
#                                     unit_tax_0_percentage = tax_actual_0 / (activity_amount_0*unit_convert_percentage)
#                                     contribution_tax_actual_0[(tax, tech, fuel)] = unit_tax_0
#                                     contribution_tax_actual_0[(f'Transport Tax {tax} [$]', tech, fuel)] = tax_actual_0
#                                 else:
#                                     if params_tier2['interpolate']:
#                                         if tax_actual_0 != 0 or float(tax_actual_0) != 0.0:
#                                             unit_tax_0 = 0
#                                             unit_tax_0_percentage = 0
#                                             contribution_tax_actual_0[(tax, tech, fuel)] = unit_tax_0
#                                             contribution_tax_actual_0[(f'Transport Tax {tax} [$]', tech, fuel)] = tax_actual_0
#                                         else:
#                                             unit_tax_0 = 0#1 # It's forcefully normalized
#                                             unit_tax_0_percentage = 0
#                                             contribution_tax_actual_0[(tax, tech, fuel)] = unit_tax_0
#                                             contribution_tax_actual_0[(f'Transport Tax {tax} [$]', tech, fuel)] = tax_actual_0
#                                             # Because tax_actual_0=0 and activity_amount_0=0 so,
#                                             # 0/0 isn't posible, but if is assing 1, before when
#                                             # apply the formula tax_actual_0=unit_tax_0*activity_amount
#                                             # tax_actual_0=1*0=0

#                                 values_list = ['' for year in time_vector]
#                                 values_list[year_index] = unit_tax
#                                 values_list_percentage = ['' for year in time_vector]
#                                 values_list_percentage[year_index] = unit_tax_percentage
#                                 incomes_list = ['' for _ in time_vector]
#                                 incomes_list[year_index] = tax_actual
#                                 if tax in tax_exceptions and fuel in fuel_exceptions and unit_tax != np.float64(0.0):
#                                     values_list[year_index] = 0
#                                     values_list_percentage[year_index] = 0
#                                     incomes_list[year_index] = 0
#                                 elif tax in tax_exceptions and fuel in fuel_exceptions and unit_tax == np.float64(0.0):
#                                     values_list[year_index] = unit_tax
#                                     values_list_percentage[year_index] = unit_tax_percentage
#                                     if tax == tax_id:
#                                         incomes_list[year_index] = tax_adjustment_percent * tax_actual
#                                     else:
#                                         incomes_list[year_index] = tax_actual
#                                 else:
#                                     values_list[year_index] = unit_tax
#                                     values_list_percentage[year_index] = unit_tax_percentage
#                                     if tax == tax_id:
#                                         incomes_list[year_index] = tax_adjustment_percent * tax_actual
#                                     else:
#                                         incomes_list[year_index] = tax_actual
                                
#                                 if params_tier2['interpolate']:
#                                     values_list[index_initial_year_interpolation] = unit_tax_0
#                                     values_list_percentage[index_initial_year_interpolation] = unit_tax_0_percentage
#                                     incomes_list[index_initial_year_interpolation] = tax_actual_0
#                                     if tax in tax_exceptions and fuel in fuel_exceptions and unit_tax_0 != np.float64(0.0):
#                                         values_list[index_initial_year_interpolation] = 0
#                                         values_list_percentage[index_initial_year_interpolation] = 0
#                                     elif tax in tax_exceptions and fuel in fuel_exceptions and unit_tax_0 == np.float64(0.0):
#                                         values_list[index_initial_year_interpolation] = unit_tax_0
#                                         values_list_percentage[index_initial_year_interpolation] = unit_tax_0_percentage
#                                     elif tax not in tax_exceptions and fuel not in fuel_exceptions and unit_tax_0 != np.float64(0.0):
#                                         values_list[index_initial_year_interpolation] = 0
#                                         values_list_percentage[index_initial_year_interpolation] = 0
#                                     elif tax not in tax_exceptions and fuel not in fuel_exceptions and unit_tax_0 == np.float64(0.0):
#                                         values_list[index_initial_year_interpolation] = unit_tax_0
#                                         values_list_percentage[index_initial_year_interpolation] = unit_tax_0_percentage
#                                     else:
#                                         values_list[index_initial_year_interpolation] = unit_tax_0
#                                         values_list_percentage[index_initial_year_interpolation] = unit_tax_0_percentage
                                    
#                                     for y in range(0,index_initial_year_interpolation):
#                                         values_list[y] = 0
#                                     #     values_list_percentage[y] = values_list_percentage[index_initial_year_interpolation]
#                                     #     incomes_list[y] = incomes_list[index_initial_year_interpolation]
                                    
#                                     values_list = interpolation_to_end(time_vector, \
#                                         params_tier2['year_ini_inter'], values_list, 'interme', 'Population')
#                                     values_list_percentage = interpolation_to_end(time_vector, \
#                                         params_tier2['year_ini_inter'], values_list_percentage, 'interme', 'Population')
#                                     incomes_list = interpolation_to_end(time_vector, \
#                                         params_tier2['year_ini_inter'], incomes_list, 'interme', 'Population')                    
                                
#                                 unit_taxes[scenario][region][country][tax][tech][fuel] = values_list
#                                 unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = incomes_list
#                                 unit_taxes_percentage[scenario][region][country][tax][tech][fuel] = values_list_percentage
#                                 unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = incomes_list
                                
#                     # Perform operations to finalize tax calculations and adjustments
#                     sum_and_assign_weights(dict_activity_distribution_gap)   
#                     calculate_tax_impact(dict_activity_distribution_gap, new_tax_contribution)    
#                     apply_tax_impact_to_structure(unit_taxes, unit_taxes_percentage, dict_activity_distribution_gap, scenario, region, country, time_vector, year_index, contribution_tax_actual_0, params_tier2, tax_id, dict_mult_depr)

#     return unit_taxes, unit_taxes_percentage, dict_activity_distribution_gap

# factors by activity scenarios
def calculate_unit_tax_by_activity(year, dict_scen, adjustments_percent, dict_tax_scen, dict_activities_out, fiscal_gaps, factors, time_vector, params_tier2, dict_mult_depr, base_dict_scen, alternative_tax='IVA_Elec'):
    """
    Calculates unit taxes across various scenarios by incorporating fiscal gaps, adjustment factors, and specific economic activities,
    with an option for interpolating these values over a timeline. This function adjusts unit taxes based on a combination of economic activities
    and fiscal policies, considering different scenarios except for the 'BAU' (Business As Usual).

    Args:
        year (int): The target year for which the unit taxes are to be calculated.
        dict_scen (dict): A dictionary containing the scenario-specific data, which includes detailed activities for different technologies and fuels.
        adjustments_percent (dict): A dictionary holding adjustment percentages for each tax, categorized by scenario, region, and country.
        dict_tax_scen (dict): A dictionary with actual tax amounts for each tax type across different scenarios, regions, countries, technologies, and fuels.
        dict_activities_out (dict): A dictionary mapping tax types to corresponding activity data, which helps in the calculation of unit taxes.
        fiscal_gaps (dict): A dictionary detailing fiscal gaps for various scenarios, which are used to adjust the tax calculations.
        factors (dict): A dictionary containing factors that are used to distribute the fiscal gaps among various taxes.
        time_vector (list): A list of all years for which data is available and calculations need to be made.
        params_tier2 (dict): Parameters that include settings for interpolation, which can be used to smooth out changes in tax calculations over time.

    Returns:
        dict: A nested dictionary structured by scenario, region, country, tax type, technology, and fuel. Each entry contains a list of calculated unit
              taxes for each year, adjusted for fiscal gaps and specific activities.

    Raises:
        ValueError: If the specified year is not found within the 'time_vector', indicating that the data for that year is unavailable.
    """
    tax_exceptions = ['IC', 'IMESI_Combust', 'IVA_Gasoil', 'Otros_Gasoil']
    fuel_exceptions = ['ELECTRICIDAD', 'HIDROGENO']

    factors_NETZERO = deepcopy(factors)

    # Define the inital year to interpolation
    index_initial_year_interpolation = time_vector.index(params_tier2['year_ini_inter'])

    try:
        year_index = time_vector.index(year)
    except ValueError:
        print(f"The year {year} is not found in the time_vector.")
        return {}

    unit_taxes = {}
    unit_taxes_percentage = {}
    
    for scenario in adjustments_percent:
        if scenario != 'BAU':
            
            if scenario == 'NETZERO':
                for tax_ex in tax_exceptions:
                    factors_NETZERO[alternative_tax] += factors_NETZERO[tax_ex]
                    factors_NETZERO[tax_ex] = 0
            
            unit_taxes[scenario] = {}
            unit_taxes_percentage[scenario] = {}
            for region in adjustments_percent[scenario]:
                unit_taxes[scenario][region] = {}
                unit_taxes_percentage[scenario][region] = {}
                for country in adjustments_percent[scenario][region]:
                    unit_taxes[scenario][region][country] = {}
                    unit_taxes_percentage[scenario][region][country] = {}
                    scenario_data = dict_scen[scenario][region][country]
                    fiscal_gap = fiscal_gaps[scenario][region][country]
                    if scenario == 'NETZERO':
                        fiscal_distribution = {tax: fiscal_gap * factors_NETZERO[tax] for tax in factors_NETZERO if tax != "Total Taxes Incomes"}
                    else:
                        fiscal_distribution = {tax: fiscal_gap * factors[tax] for tax in factors if tax != "Total Taxes Incomes"}
        
                    for tax, techs_types in dict_activities_out['Relation Tax-Activity2'].items():
                        unit_taxes[scenario][region][country][tax] = {}
                        unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'] = {}
                        unit_taxes_percentage[scenario][region][country][tax] = {}
                        unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'] = {}
                        if tax == "Total Taxes Incomes":
                            continue
                        total_activity_for_tax = 0
                        total_tax_contribution_for_tax = 0
                        activity_contribution = {}
                        tax_per_fuel_contribution = {}
        
                        for tech, fuel_types in techs_types.items():
                            for fuel, cost_type in fuel_types.items():
                                # Correctly accessing activity based on cost_type
                                if cost_type == 'CapitalCost':
                                    activity = scenario_data['New Fleet'][tech][fuel][year_index]
                                    activity_amount_0 = scenario_data['New Fleet'][tech][fuel][index_initial_year_interpolation]
                                elif cost_type == 'CapitalCost*':
                                    activity = scenario_data['Fleet'][tech][fuel][year_index]
                                    activity_amount_0 = scenario_data['Fleet'][tech][fuel][index_initial_year_interpolation]
                                elif cost_type == 'KmCost':
                                    activity = scenario_data['Fleet'][tech][fuel][year_index]
                                    activity_amount_0 = scenario_data['Fleet'][tech][fuel][index_initial_year_interpolation]
                                else:  # 'VariableCost' and other costs
                                    activity = scenario_data['Fuel Consumption'][tech][fuel][year_index]
                                    conversion_factors = scenario_data['Conversion Fuel Constant'][tech][fuel]
                                    activity /= conversion_factors
                                    activity_amount_0 = scenario_data['Fuel Consumption'][tech][fuel][index_initial_year_interpolation]
                                    activity_amount_0 /= conversion_factors
                                if (tax in tax_exceptions and scenario == 'NETZERO'):
                                    tax_contribution_for_fuel = 0
                                else:
                                    tax_contribution_for_fuel = dict_tax_scen[scenario][region][country][tax][tech][fuel][year_index]
                                
                                if (tax in tax_exceptions and scenario == 'NETZERO'):
                                    activity = 0

                                total_activity_for_tax += activity
                                total_tax_contribution_for_tax += tax_contribution_for_fuel
                                activity_contribution[(tech, fuel)] = activity
                                tax_per_fuel_contribution[(tech, fuel)] = tax_contribution_for_fuel
        
                        for tech, fuel_types in techs_types.items():
                            unit_taxes[scenario][region][country][tax][tech] = {}
                            unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech] = {}
                            unit_taxes_percentage[scenario][region][country][tax][tech] = {}
                            unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech] = {}
                            
                            for fuel, cost_type in fuel_types.items():
                                activity_amount = activity_contribution[(tech, fuel)]
                                tax_per_fuel_amount = tax_per_fuel_contribution[(tech, fuel)]
                                proportion_activity = activity_amount / total_activity_for_tax if total_activity_for_tax > 0 else 0
                                specific_adjustment = fiscal_distribution[tax] * proportion_activity
                                unit_convert_percentage = dict_mult_depr[tax][tech][fuel]
                                if float(unit_convert_percentage) == float(0):
                                    unit_convert_percentage = 1
                                    
                                # Correctly accessing activity based on cost_type
                                if cost_type == 'CapitalCost':
                                    activity_amount_0 = scenario_data['New Fleet'][tech][fuel][index_initial_year_interpolation]
                                    activity_base = scenario_data['New Fleet'][tech][fuel]
                                elif cost_type == 'CapitalCost*':
                                    activity_amount_0 = scenario_data['Fleet'][tech][fuel][index_initial_year_interpolation]
                                    activity_base = scenario_data['Fleet'][tech][fuel]
                                elif cost_type == 'KmCost':
                                    activity_amount_0 = scenario_data['Fleet'][tech][fuel][index_initial_year_interpolation]
                                    activity_base = scenario_data['Fleet'][tech][fuel]
                                else:  # 'VariableCost' and other costs
                                    conversion_factors = scenario_data['Conversion Fuel Constant'][tech][fuel]
                                    activity_amount_0 = scenario_data['Fuel Consumption'][tech][fuel][index_initial_year_interpolation]
                                    activity_amount_0 /= conversion_factors
                                    activity_base = scenario_data['Fuel Consumption'][tech][fuel]
                                    activity_base = [x / conversion_factors for x in activity_base]

                                if activity_amount > 0:
                                    unit_tax = (specific_adjustment) / activity_amount
                                else:
                                    unit_tax = 0.0  # No activity means no unit tax calculation necessary

                                values_list_base = deepcopy(base_dict_scen[scenario][region][country][tax][tech][fuel])

                                values_list = ['' for year in time_vector]
                                values_list[year_index] = unit_tax
                                if tax in tax_exceptions and scenario == 'NETZERO':
                                    values_list[year_index] = 0.0
                                else:
                                    values_list[year_index] = unit_tax
                                
                                values_list[index_initial_year_interpolation] = 0.0
                                if tax not in tax_exceptions and fuel not in fuel_exceptions and scenario == 'NETZERO':
                                    values_list[index_initial_year_interpolation] = 0.0
                                else:
                                    values_list[index_initial_year_interpolation] = 0.0
                                
                                for y in range(0,index_initial_year_interpolation):
                                    values_list[y] = 0.0
                                
                                values_list = interpolation_to_end(time_vector, \
                                    params_tier2['year_ini_inter'], values_list, 'interme', 'Population')  
                                values_list_final = [a + b for a, b in zip(values_list_base, values_list)]

                                unit_taxes[scenario][region][country][tax][tech][fuel] = values_list_final
                                unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = [a * b for a, b in zip(values_list_final, activity_base)]
                                unit_taxes_percentage[scenario][region][country][tax][tech][fuel] = [x / unit_convert_percentage for x in values_list_final]
                                unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = [a * b for a, b in zip(values_list_final, activity_base)]
                            
                                # if scenario == 'ACELERADO' and tax == 'Patente' and tech == 'Buses MVD' and fuel == 'DIESEL OIL':
                                #     print(2)
                                #     print(values_list_base)
                                #     print(values_list)
                                #     print(values_list_final)
                                #     sys.exit()

    return unit_taxes, unit_taxes_percentage

# # factors by activity scenarios (andrey function)
# def calculate_unit_tax_by_activity(year, dict_scen, adjustments_percent, dict_tax_scen, dict_activities_out, fiscal_gaps, factors, time_vector, params_tier2, dict_mult_depr, base_dict_scen, alternative_tax='IVA_Elec'):
#     """
#     Calculates unit taxes across various scenarios by incorporating fiscal gaps, adjustment factors, and specific economic activities,
#     with an option for interpolating these values over a timeline. This function adjusts unit taxes based on a combination of economic activities
#     and fiscal policies, considering different scenarios except for the 'BAU' (Business As Usual).

#     Args:
#         year (int): The target year for which the unit taxes are to be calculated.
#         dict_scen (dict): A dictionary containing the scenario-specific data, which includes detailed activities for different technologies and fuels.
#         adjustments_percent (dict): A dictionary holding adjustment percentages for each tax, categorized by scenario, region, and country.
#         dict_tax_scen (dict): A dictionary with actual tax amounts for each tax type across different scenarios, regions, countries, technologies, and fuels.
#         dict_activities_out (dict): A dictionary mapping tax types to corresponding activity data, which helps in the calculation of unit taxes.
#         fiscal_gaps (dict): A dictionary detailing fiscal gaps for various scenarios, which are used to adjust the tax calculations.
#         factors (dict): A dictionary containing factors that are used to distribute the fiscal gaps among various taxes.
#         time_vector (list): A list of all years for which data is available and calculations need to be made.
#         params_tier2 (dict): Parameters that include settings for interpolation, which can be used to smooth out changes in tax calculations over time.

#     Returns:
#         dict: A nested dictionary structured by scenario, region, country, tax type, technology, and fuel. Each entry contains a list of calculated unit
#               taxes for each year, adjusted for fiscal gaps and specific activities.

#     Raises:
#         ValueError: If the specified year is not found within the 'time_vector', indicating that the data for that year is unavailable.
#     """
#     tax_exceptions = ['IC', 'IMESI_Combust', 'IVA_Gasoil', 'Otros_Gasoil']
#     fuel_exceptions = ['ELECTRICIDAD', 'HIDROGENO']

#     factors_NETZERO = deepcopy(factors)

#     # Define the inital year to interpolation
#     index_initial_year_interpolation = time_vector.index(params_tier2['year_ini_inter'])

#     try:
#         year_index = time_vector.index(year)
#     except ValueError:
#         print(f"The year {year} is not found in the time_vector.")
#         return {}

#     unit_taxes = {}
#     unit_taxes_percentage = {}
    
#     for scenario in adjustments_percent:
#         if scenario != 'BAU':
            
#             if scenario == 'NETZERO':
#                 for tax_ex in tax_exceptions:
#                     factors_NETZERO[alternative_tax] += factors_NETZERO[tax_ex]
#                     factors_NETZERO[tax_ex] = 0
            
#             unit_taxes[scenario] = {}
#             unit_taxes_percentage[scenario] = {}
#             for region in adjustments_percent[scenario]:
#                 unit_taxes[scenario][region] = {}
#                 unit_taxes_percentage[scenario][region] = {}
#                 for country in adjustments_percent[scenario][region]:
#                     unit_taxes[scenario][region][country] = {}
#                     unit_taxes_percentage[scenario][region][country] = {}
#                     scenario_data = dict_scen[scenario][region][country]
#                     fiscal_gap = fiscal_gaps[scenario][region][country]
#                     if scenario == 'NETZERO':
#                         fiscal_distribution = {tax: fiscal_gap * factors_NETZERO[tax] for tax in factors_NETZERO if tax != "Total Taxes Incomes"}
#                     else:
#                         fiscal_distribution = {tax: fiscal_gap * factors[tax] for tax in factors if tax != "Total Taxes Incomes"}
        
#                     for tax, techs_types in dict_activities_out['Relation Tax-Activity2'].items():
#                         unit_taxes[scenario][region][country][tax] = {}
#                         unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'] = {}
#                         unit_taxes_percentage[scenario][region][country][tax] = {}
#                         unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'] = {}
#                         if tax == "Total Taxes Incomes":
#                             continue
#                         total_activity_for_tax = 0
#                         total_tax_contribution_for_tax = 0
#                         activity_contribution = {}
#                         tax_per_fuel_contribution = {}
        
#                         for tech, fuel_types in techs_types.items():
#                             for fuel, cost_type in fuel_types.items():
#                                 # Correctly accessing activity based on cost_type
#                                 if cost_type == 'CapitalCost':
#                                     activity = scenario_data['New Fleet'][tech][fuel][year_index]
#                                     if params_tier2['interpolate']:
#                                         activity_amount_0 = scenario_data['New Fleet'][tech][fuel][index_initial_year_interpolation]
#                                 elif cost_type == 'CapitalCost*':
#                                     activity = scenario_data['Fleet'][tech][fuel][year_index]
#                                     if params_tier2['interpolate']:
#                                         activity_amount_0 = scenario_data['Fleet'][tech][fuel][index_initial_year_interpolation]
#                                 else:  # 'VariableCost' and other costs
#                                     activity = scenario_data['Fuel Consumption'][tech][fuel][year_index]
#                                     conversion_factors = scenario_data['Conversion Fuel Constant'][tech][fuel]
#                                     activity /= conversion_factors
#                                     if params_tier2['interpolate']:
#                                         activity_amount_0 = scenario_data['Fuel Consumption'][tech][fuel][index_initial_year_interpolation]
#                                         activity_amount_0 /= conversion_factors
#                                 if (tax in tax_exceptions and scenario == 'NETZERO'):
#                                     tax_contribution_for_fuel = 0
#                                 else:
#                                     tax_contribution_for_fuel = dict_tax_scen[scenario][region][country][tax][tech][fuel][year_index]
                                
#                                 if (tax in tax_exceptions and scenario == 'NETZERO'):
#                                     activity = 0

#                                 total_activity_for_tax += activity
#                                 total_tax_contribution_for_tax += tax_contribution_for_fuel
#                                 activity_contribution[(tech, fuel)] = activity
#                                 tax_per_fuel_contribution[(tech, fuel)] = tax_contribution_for_fuel
        
#                         for tech, fuel_types in techs_types.items():
#                             unit_taxes[scenario][region][country][tax][tech] = {}
#                             unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech] = {}
#                             unit_taxes_percentage[scenario][region][country][tax][tech] = {}
#                             unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech] = {}
                            
#                             for fuel, cost_type in fuel_types.items():
#                                 activity_amount = activity_contribution[(tech, fuel)]
#                                 tax_per_fuel_amount = tax_per_fuel_contribution[(tech, fuel)]
#                                 proportion_activity = activity_amount / total_activity_for_tax if total_activity_for_tax > 0 else 0
#                                 specific_tax_contribution = tax_per_fuel_amount
#                                 specific_adjustment = fiscal_distribution[tax] * proportion_activity
#                                 unit_convert_percentage = dict_mult_depr[tax][tech][fuel]
#                                 if float(unit_convert_percentage) == float(0):
#                                     unit_convert_percentage = 1
                                    
#                                 if params_tier2['interpolate']:
#                                     tax_actual_0 = dict_scen[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel][index_initial_year_interpolation]

#                                 # Correctly accessing activity based on cost_type
#                                 if cost_type == 'CapitalCost':
#                                     if params_tier2['interpolate']:
#                                         activity_amount_0 = scenario_data['New Fleet'][tech][fuel][index_initial_year_interpolation]
#                                 elif cost_type == 'CapitalCost*':
#                                     if params_tier2['interpolate']:
#                                         activity_amount_0 = scenario_data['Fleet'][tech][fuel][index_initial_year_interpolation]
#                                 else:  # 'VariableCost' and other costs
#                                     conversion_factors = scenario_data['Conversion Fuel Constant'][tech][fuel]
#                                     if params_tier2['interpolate']:
#                                         activity_amount_0 = scenario_data['Fuel Consumption'][tech][fuel][index_initial_year_interpolation]
#                                         activity_amount_0 /= conversion_factors

#                                 if activity_amount > 0:
#                                     unit_tax = (specific_tax_contribution + specific_adjustment) / activity_amount
#                                     unit_tax_percentage = (specific_tax_contribution + specific_adjustment) / (activity_amount*unit_convert_percentage)
#                                 else:
#                                     unit_tax = 0  # No activity means no unit tax calculation necessary
#                                     unit_tax_percentage = 0
#                                 if params_tier2['interpolate'] and (float(activity_amount_0) > 0.0 or float(activity_amount_0) < 0.0):
#                                     unit_tax_0 = tax_actual_0 / activity_amount_0
#                                     unit_tax_0_percentage = tax_actual_0 / (activity_amount_0*unit_convert_percentage)
#                                 else:
#                                     if params_tier2['interpolate']:
#                                         if tax_actual_0 != 0 or float(tax_actual_0) != 0.0:
#                                             unit_tax_0 = 0
#                                             unit_tax_0_percentage = 0
#                                         else:
#                                             unit_tax_0 = 0 # It's forcefully normalized 
#                                             # Because tax_actual_0=0 and activity_amount_0=0 so,
#                                             # 0/0 isn't posible, but if is assing 1, before when
#                                             # apply the formula tax_actual_0=unit_tax_0*activity_amount
#                                             # tax_actual_0=1*0=0
#                                             unit_tax_0_percentage = 0
                                
#                                 values_list = ['' for year in time_vector]
#                                 values_list[year_index] = unit_tax
#                                 values_list_percentage = ['' for year in time_vector]
#                                 values_list_percentage[year_index] = unit_tax_percentage
#                                 incomes_list = ['' for _ in time_vector]
#                                 incomes_list[year_index] = specific_tax_contribution
#                                 if tax in tax_exceptions and scenario == 'NETZERO':
#                                     values_list[year_index] = 0
#                                     values_list_percentage[year_index] = 0
#                                     incomes_list[year_index] = 0
#                                 else:
#                                     values_list[year_index] = unit_tax
#                                     values_list_percentage[year_index] = unit_tax_percentage
#                                     incomes_list[year_index] = specific_tax_contribution + specific_adjustment
                                
#                                 if params_tier2['interpolate']:
#                                     values_list[index_initial_year_interpolation] = unit_tax_0
#                                     values_list_percentage[index_initial_year_interpolation] = unit_tax_0_percentage
#                                     incomes_list[index_initial_year_interpolation] = tax_actual_0
#                                     if tax not in tax_exceptions and fuel not in fuel_exceptions and scenario == 'NETZERO':
#                                         values_list[index_initial_year_interpolation] = 0
#                                         values_list_percentage[index_initial_year_interpolation] = 0
#                                     else:
#                                         values_list[index_initial_year_interpolation] = unit_tax_0
#                                         values_list_percentage[index_initial_year_interpolation] = unit_tax_0_percentage
                                    
#                                     for y in range(0,index_initial_year_interpolation):
#                                         values_list[y] = 0
#                                         values_list_percentage[y] = 0
#                                     #     incomes_list[y] = incomes_list[index_initial_year_interpolation]
                                    
#                                     values_list = interpolation_to_end(time_vector, \
#                                         params_tier2['year_ini_inter'], values_list, 'interme', 'Population')
#                                     values_list_percentage = interpolation_to_end(time_vector, \
#                                         params_tier2['year_ini_inter'], values_list_percentage, 'interme', 'Population')
#                                     # incomes_list = interpolation_to_end(time_vector, \
#                                     #     params_tier2['year_ini_inter'], incomes_list, 'interme', 'Population')
                                        
#                                 unit_taxes[scenario][region][country][tax][tech][fuel] = values_list
#                                 unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = incomes_list
#                                 unit_taxes_percentage[scenario][region][country][tax][tech][fuel] = values_list_percentage
#                                 unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = incomes_list
                                
#                                 if scenario == 'NETZERO' and tax == 'Rodaje' and tech == 'Automoviles' and fuel == 'ELECTRICIDAD':
#                                     print(2)
#                                     print(activity_amount,unit_tax)
#     return unit_taxes, unit_taxes_percentage




def read_factors(xls_path, sheet_name, factor_column_name):
    """
    Reads adjustment factors from an Excel sheet and structures them into two dictionaries based on their applicability to specific taxes,
    technologies, and fuels. This function simplifies the process of accessing factors for specific calculations in tax and subsidy scenarios.

    Args:
        xls_path (str): The file path for the Excel workbook.
        sheet_name (str): The name of the sheet within the workbook from which to read the factors.
        factor_column_name (str): The name of the column containing the factors.

    Returns:
        tuple: Contains two dictionaries:
            - factors_by_tax: A dictionary where each key is a tax name, and the value is the factor applicable directly to that tax.
            - factors_by_fuel: A nested dictionary where each tax is a key, and its value is another dictionary containing technologies and
                               fuels with their corresponding factors.

    Raises:
        FileNotFoundError: If the Excel file at the specified path does not exist.
        ValueError: If the Excel file is incorrectly formatted or missing necessary columns ('TAX', 'TECHNOLOGY', 'FUEL', and the specified factor column).
    """
    df = pd.read_excel(xls_path, sheet_name=sheet_name)

    factors_by_fuel = {}
    factors_by_tax = {}

    for idx, row in df.iterrows():
        tax = row['TAX']
        technology = row['TECHNOLOGY']
        fuel = row['FUEL']
        factor = row[factor_column_name]

        if isinstance(factor, str) and '%' in factor:
            factor = float(factor.replace('%', '')) / 100  # Handle percentage strings if present

        if tax not in factors_by_fuel:
            factors_by_fuel[tax] = {'factor': None, 'techs': {}}

        if pd.isna(technology):
            factors_by_fuel[tax]['factor'] = factor
            factors_by_tax[tax] = factor  # Store factor for the tax directly in factors_by_tax
        else:
            if technology not in factors_by_fuel[tax]['techs']:
                factors_by_fuel[tax]['techs'][technology] = {'factor': None, 'fuels': {}}
            
            if pd.isna(fuel):
                factors_by_fuel[tax]['techs'][technology]['factor'] = factor
            else:
                if fuel not in factors_by_fuel[tax]['techs'][technology]['fuels']:
                    factors_by_fuel[tax]['techs'][technology]['fuels'][fuel] = factor

    return factors_by_tax, factors_by_fuel


def verify_factors(factors_by_fuel):
    """
    Verifies the consistency of tax, technology, and fuel factors within the provided structure to ensure that they sum to approximately 1,
    which is essential for maintaining balanced allocations in fiscal scenarios. This function checks each level of the data structure:
    taxes, technologies within taxes, and fuels within technologies.

    Args:
        factors_by_fuel (dict): A nested dictionary containing scenario-based tax factors, where each tax can have nested technologies,
                                and each technology can have nested fuel factors.

    Outputs:
        This function prints warning messages directly to the console if any of the tax, technology, or fuel factors do not sum to approximately 1,
        indicating potential issues in the factor distribution that may affect calculations based on these factors.
    """
    for scenario, regions in factors_by_fuel.items():
        if scenario != 'BAU':  # Skipping the 'Business As Usual' scenario which might be treated differently    
            for region, countries in regions.items():
                for country, taxes in countries.items():
                    # Calculate and verify the sum of tax factors for each country within a region and scenario
                    tax_total = sum(tax['factor'] for tax in taxes.values() if 'factor' in tax)
                    if not 0.99 <= tax_total <= 1.01:
                        print(f"Warning: Total tax factors for {scenario}-{region}-{country} do not sum to 1. Sum is {tax_total}")
                    for tax, techs in taxes.items():
                        # Calculate and verify the sum of technology factors within each tax
                        tech_total = sum(tech['factor'] for tech in techs['techs'].values())
                        if not (0.99 <= tech_total <= 1.01 or tech_total == 0.0):
                            print(f"Warning: Tech factors for {tax} in {scenario}-{region}-{country} do not sum to 1. Sum is {tech_total}")
                        for tech, fuels in techs['techs'].items():
                            # Calculate and verify the sum of fuel factors within each technology
                            fuel_total = sum(fuel for fuel in fuels['fuels'].values())
                            if not (0.99 <= fuel_total <= 1.01 or fuel_total == 0.0):
                                print(f"Warning: Fuel factors for {tech} under {tax} in {scenario}-{region}-{country} do not sum to 1. Sum is {fuel_total}")

def redistribute_factors_if_no_activity(factors_by_fuel, dict_scen, year_index, params_tier2, tax_exceptions, fuel_exceptions, time_vector, dict_activities_out, alternative_tax='IVA_Elec'):
    """
    Redistributes factors within the same technology or across technologies within the same tax category if there's no activity for certain fuels.
    This redistribution takes into account only those fuels that originally had a factor assigned and redistributes to active fuels based on their
    existing factor proportions. If no active fuels are available, it attempts to redistribute across other technologies in the tax category.
    """
    for scenario, regions in factors_by_fuel.items():
        if scenario != 'BAU':  # Skip 'BAU' as it's not part of the redistribution
            for region, countries in regions.items():
                for country, taxes in countries.items():
                    for tax, techs in taxes.items():
                        type_tax = dict_activities_out['Relation Tax-Activity'][tax]
                        for tech, fuel_data in techs['techs'].items():
                            active_fuels = {}
                            inactive_fuels = {}

                            # Continue the normal processing for each fuel and scenario
                            for fuel, factor in fuel_data['fuels'].items():
                                activity, activity_amount_0,activity_base = get_activity_for_fuel(dict_scen[scenario][region][country], tech, fuel, year_index, params_tier2, time_vector, type_tax)
                                if activity > 0:
                                    active_fuels[fuel] = {'activity': activity, 'factor': factor}
                                elif activity <= 0 and factor > 0:
                                    inactive_fuels[fuel] = factor

                            # Redistribute factors between active and inactive fuels
                            if inactive_fuels and active_fuels:
                                total_factor_to_redistribute = sum(active_fuels[fuel]['factor'] for fuel in active_fuels)
                                if total_factor_to_redistribute == 0:
                                    # print(f"Warning: No active fuels have factors for redistribution, for: {scenario}/{tax}/{tech}/{fuel}.")
                                    pass
                                else:
                                    for inactive_fuel, redistributed_factor in inactive_fuels.items():
                                        for fuel in active_fuels:
                                            proportion = active_fuels[fuel]['factor'] / total_factor_to_redistribute
                                            fuel_data['fuels'][fuel] += redistributed_factor * proportion
                                            fuel_data['fuels'][inactive_fuel] = 0

                            # Reset tech factors and tax factor after redistributing
                            if scenario == 'NETZERO' and tax in tax_exceptions:
                                for tech in techs['techs']:
                                    techs['techs'][tech]['factor'] = 0
                                taxes[alternative_tax]['factor'] += taxes[tax]['factor']
                                taxes[tax]['factor'] = 0
                            # If no active fuels, redistribute factors to other technologies
                            if not active_fuels and any(inactive_fuels.values()):
                                print(tax, tech, fuel)
                                total_redistributed_factor = sum(inactive_fuels.values())
                                redistribute_across_technologies(techs['techs'], total_redistributed_factor)




def get_activity_for_fuel(scenario_data, tech, fuel, year_index, params_tier2, time_vector, type_tax):
    """
    Retrieves activity data for a given fuel under a specific technology and scenario based on the year index. The function also considers
    different cost types ('CapitalCost', 'CapitalCost*', and others) to determine the correct data path and performs conditional data handling
    based on whether interpolation is enabled.

    Args:
        scenario_data (dict): A nested dictionary containing detailed data for a specific scenario, structured by technology and fuel.
        tech (str): The technology category under which the fuel's activity data is classified.
        fuel (str): The specific fuel type for which activity data needs to be retrieved.
        year_index (int): The index representing the year in the dataset for which the data is to be retrieved.

    Returns:
        tuple: A tuple containing the activity data for the specified fuel and technology. If interpolation is enabled, it returns the activity
               data for both the specified year and the initial year (index 0). If not, the second element of the tuple is None.
    """
    # Define the inital year to interpolation
    index_initial_year_interpolation = time_vector.index(params_tier2['year_ini_inter'])
    if 'CapitalCost' in type_tax:
        activity = scenario_data['New Fleet'][tech][fuel][year_index]
        activity_amount_0 = scenario_data['New Fleet'][tech][fuel][index_initial_year_interpolation]
        activity_base = scenario_data['New Fleet'][tech][fuel]
    elif 'CapitalCost*' in type_tax:
        activity = scenario_data['Fleet'][tech][fuel][year_index]
        activity_amount_0 = scenario_data['Fleet'][tech][fuel][index_initial_year_interpolation]
        activity_base = scenario_data['Fleet'][tech][fuel]
    elif 'KmCost' in type_tax:
        activity = scenario_data['Fleet'][tech][fuel][year_index]
        activity_amount_0 = scenario_data['Fleet'][tech][fuel][index_initial_year_interpolation]
        activity_base = scenario_data['Fleet'][tech][fuel]
    else:
        activity = scenario_data['Fuel Consumption'][tech][fuel][year_index]
        conversion_factors = scenario_data['Conversion Fuel Constant'][tech][fuel]
        activity /= conversion_factors
        activity_amount_0 = scenario_data['Fuel Consumption'][tech][fuel][index_initial_year_interpolation]
        activity_amount_0 /= conversion_factors
        activity_base = scenario_data['Fuel Consumption'][tech][fuel]
        activity_base = [x / conversion_factors for x in activity_base]

    return activity, activity_amount_0,activity_base

def redistribute_across_technologies(techs, factor_to_redistribute):
    """
    Redistributes a given factor to other technologies proportionally based on their existing factors. This function is typically called
    when all fuels in one technology are inactive and their factors need to be reallocated to ensure proper balance within the scenario.

    Args:
        techs (dict): A dictionary of technologies where each key is a technology name and each value is a dictionary containing details
                      about the technology, including its current factor.
        factor_to_redistribute (float): The total factor amount to be redistributed among the active technologies.

    Raises:
        ZeroDivisionError: If the total factor is zero, which would occur if no technologies have factors assigned, leading to a division by zero
                            error when attempting to calculate proportions for redistribution.
    """
    # print(techs.values())
    total_factor = sum(tech['factor'] for tech in techs.values() if 'factor' in tech) 
    for tech in techs.values():
        tech['factor'] += factor_to_redistribute * (tech['factor'] / total_factor)


# factors scenarios
def calculate_unit_tax_with_factors_for_fuels(year, dict_scen, dict_tax_scen, dict_activities_out, fiscal_gaps, factors_by_fuel, time_vector, count, params_tier2, dict_mult_depr, base_dict_scen):
    """
    Calculates the unit taxes for each fuel across various scenarios, adjusting for fiscal gaps and specific factors, with the option to interpolate results over time.
    This function also logs warnings when certain conditions such as zero activity with non-zero factors are detected.

    Args:
        year (int): The target year for which taxes are calculated.
        dict_scen (dict): Scenario data containing activity levels for various technologies and fuels.
        dict_tax_scen (dict): Tax data for each scenario, region, country, tax, technology, and fuel.
        dict_activities_out (dict): A mapping of tax types to activity data for further reference (not directly used in this function).
        fiscal_gaps (dict): Fiscal gap values for each scenario and region.
        factors_by_fuel (dict): Configured factors for each fuel under each technology and tax.
        time_vector (list): A list of years for which data is available and calculations need to be made.
        count (int): A counter to track the number of warnings or specific conditions encountered.
        params_tier2 (dict): Parameters that include settings for interpolation, which can be used to smooth out changes in tax calculations over time.

    Returns:
        tuple: Returns a tuple containing:
               - A nested dictionary structured by scenario, region, country, tax, technology, and fuel, each containing lists of calculated unit taxes for each year.
               - An updated count of warnings encountered during processing.

    Warnings:
        Warnings are logged to "activity_warnings.txt" and include cases where fuels have factors assigned but no corresponding activity data.
        Each warning increments the `count` variable, which is returned with the final data structure.
    """
    factors_by_fuel_temp = deepcopy(factors_by_fuel)
    # verify_factors(factors_by_fuel)
    tax_exceptions = ['IC', 'IMESI_Combust', 'IVA_Gasoil', 'Otros_Gasoil']
    fuel_exceptions = ['ELECTRICIDAD', 'HIDROGENO']
    
    # Define the inital year to interpolation
    index_initial_year_interpolation = time_vector.index(params_tier2['year_ini_inter'])
    
    try:
        year_index = time_vector.index(year)
    except ValueError:
        print(f"The year {year} is not found in the time_vector.")
        return {}
    
    unit_taxes = {}
    unit_taxes_percentage = {}
    
    # Open a file to write the warnings
    with open("activity_warnings.txt", "w") as file:
        for scenario in factors_by_fuel:
            if scenario != 'BAU':
                file.write(f"#***********************************************************************************************#\n{scenario}:\n")
                unit_taxes[scenario] = {}
                unit_taxes_percentage[scenario] = {}
                for region in factors_by_fuel[scenario]:
                    unit_taxes[scenario][region] = {}
                    unit_taxes_percentage[scenario][region] = {}
                    for country in factors_by_fuel[scenario][region]:
                        unit_taxes[scenario][region][country] = {}
                        unit_taxes_percentage[scenario][region][country] = {}
                        scenario_data = dict_scen[scenario][region][country]
                        fiscal_gap = fiscal_gaps[scenario][region][country]
                        factors_config = factors_by_fuel[scenario][region][country]
                        factors_by_fuel = factors_by_fuel_temp
                        # redistribute_factors_if_no_activity(factors_by_fuel, dict_scen, year_index, params_tier2, tax_exceptions, fuel_exceptions, time_vector, dict_activities_out)  # Redistribute factors if needed
                        verify_factors(factors_by_fuel)

                        for tax, techs in factors_config.items():
                            file.write(f"################################################################################\n{tax}:\n")
                            unit_taxes[scenario][region][country][tax] = {}
                            unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'] = {}
                            unit_taxes_percentage[scenario][region][country][tax] = {}
                            unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'] = {}
                            total_gap_for_tax = fiscal_gap * techs['factor']  # Tax specific factor
                            type_tax = dict_activities_out['Relation Tax-Activity'][tax]
                            
                            for tech, fuels in techs['techs'].items():
                                file.write(f"{tech}:\n")
                                unit_taxes[scenario][region][country][tax][tech] = {}
                                unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech] = {}
                                unit_taxes_percentage[scenario][region][country][tax][tech] = {}
                                unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech] = {}
                                for fuel_key, factor_data in fuels.items():
                                    if fuel_key == 'factor':  # Skip the 'factor' key
                                        total_gap_for_tax_tech = total_gap_for_tax * fuels['factor']
                                        continue
                                    for fuel, factor_value in factor_data.items():
                                        activity_data, activity_amount_0,activity_base1 = get_activity_for_fuel(scenario_data, tech, fuel, year_index, params_tier2, time_vector, type_tax)
                                        unit_convert_percentage = dict_mult_depr[tax][tech][fuel]
                                        if float(unit_convert_percentage) == float(0):
                                            unit_convert_percentage = 1
                                        
                                        tax_contribution_for_fuel = dict_tax_scen[scenario][region][country][tax][tech][fuel][year_index]


                                        # Correctly accessing activity based on cost_type
                                        if type_tax == 'CapitalCost':
                                            # activity_amount_0 = scenario_data['New Fleet'][tech][fuel][index_initial_year_interpolation]
                                            activity_base = scenario_data['New Fleet'][tech][fuel]
                                        elif type_tax == 'CapitalCost*':
                                            # activity_amount_0 = scenario_data['Fleet'][tech][fuel][index_initial_year_interpolation]
                                            activity_base = scenario_data['Fleet'][tech][fuel]
                                        elif type_tax == 'KmCost':
                                            # activity_amount_0 = scenario_data['Fleet'][tech][fuel][index_initial_year_interpolation]
                                            activity_base = scenario_data['Fleet'][tech][fuel]
                                        else:  # 'VariableCost' and other costs
                                            conversion_factors = scenario_data['Conversion Fuel Constant'][tech][fuel]
                                            # activity_amount_0 = scenario_data['Fuel Consumption'][tech][fuel][index_initial_year_interpolation]
                                            # activity_amount_0 /= conversion_factors
                                            activity_base = scenario_data['Fuel Consumption'][tech][fuel]
                                            activity_base = [x / conversion_factors for x in activity_base]


                                        if activity_data == 0.0 and factor_value != 0.0:
                                            file.write(f"\"Warning: No activity for {fuel}, but it has a factor assigned.\"\n")
                                            count += 1
                                        tax_adjustment = total_gap_for_tax_tech * factor_value
                                        if float(activity_data) > 0.0:
                                            unit_tax = (tax_adjustment) / activity_data
                                        else:
                                            unit_tax = 0  # No activity to distribute the tax

                                        values_list_base = deepcopy(base_dict_scen[scenario][region][country][tax][tech][fuel])

                                        values_list = ['' for year in time_vector]
                                        values_list[year_index] = unit_tax
                                        if tax in tax_exceptions and unit_tax != np.float64(0.0) and scenario == 'NETZERO':
                                        # if tax in tax_exceptions and fuel in fuel_exceptions and unit_tax != np.float64(0.0) and scenario == 'No':
                                            values_list[year_index] = 0
                                        elif tax in tax_exceptions and unit_tax == np.float64(0.0) and scenario == 'NETZERO':
                                        # elif tax in tax_exceptions and fuel in fuel_exceptions and unit_tax == np.float64(0.0) and scenario == 'No':
                                            values_list[year_index] = unit_tax
                                        else:
                                            values_list[year_index] = unit_tax
                                        
                                        # if scenario == 'NETZERO' and tech == 'CamionesA' and fuel == 'GASOLINA/ALCOHOL' and tax == 'IVA_Elec':
                                        #     print(scenario)
                                        #     print(tech)
                                        #     print('unit_tax:',unit_tax)
                                        #     print('unit_tax_percentage:',unit_tax_percentage)
                                        #     print('factores:',tax_adjustment, tax_contribution_for_fuel,activity_data,unit_convert_percentage)
                                        #     print('tax_adjustment:',total_gap_for_tax_tech,factor_value)
                                        #     sys.exit(9)

                                        
                                        values_list[index_initial_year_interpolation] = 0.0
                                        for y in range(0,index_initial_year_interpolation):
                                            values_list[y] = 0.0

                                        values_list = interpolation_to_end(time_vector, \
                                            params_tier2['year_ini_inter'], values_list, 'interme', 'Population')
                                        values_list_final = [a + b for a, b in zip(values_list_base, values_list)]
                                        # print(values_list_base[-1],values_list[-1],values_list_final[-1])
                                        
                                        # if scenario == 'ACELERADO' and tax == 'Patente' and tech == 'Buses MVD' and fuel == 'DIESEL OIL':
                                        #     print(2)
                                        #     print(activity_base[-1],values_list_final[-1])
                                        #     print(values_list)
                                        
                                        # activity_base[-1] = activity_data

                                        unit_taxes[scenario][region][country][tax][tech][fuel] = values_list_final
                                        unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = [a * b for a, b in zip(values_list_final, activity_base)]
                                        unit_taxes_percentage[scenario][region][country][tax][tech][fuel] = [x / unit_convert_percentage for x in values_list_final]
                                        unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = [a * b for a, b in zip(values_list_final, activity_base)]
                                    file.write("-----------------------------------------------------------------------------\n")

    return unit_taxes, unit_taxes_percentage, count

# # factors scenarios
# def calculate_unit_tax_with_factors_for_fuels(year, dict_scen, dict_tax_scen, dict_activities_out, fiscal_gaps, factors_by_fuel, time_vector, count, params_tier2, dict_mult_depr, base_dict_scen):
#     """
#     Calculates the unit taxes for each fuel across various scenarios, adjusting for fiscal gaps and specific factors, with the option to interpolate results over time.
#     This function also logs warnings when certain conditions such as zero activity with non-zero factors are detected.

#     Args:
#         year (int): The target year for which taxes are calculated.
#         dict_scen (dict): Scenario data containing activity levels for various technologies and fuels.
#         dict_tax_scen (dict): Tax data for each scenario, region, country, tax, technology, and fuel.
#         dict_activities_out (dict): A mapping of tax types to activity data for further reference (not directly used in this function).
#         fiscal_gaps (dict): Fiscal gap values for each scenario and region.
#         factors_by_fuel (dict): Configured factors for each fuel under each technology and tax.
#         time_vector (list): A list of years for which data is available and calculations need to be made.
#         count (int): A counter to track the number of warnings or specific conditions encountered.
#         params_tier2 (dict): Parameters that include settings for interpolation, which can be used to smooth out changes in tax calculations over time.

#     Returns:
#         tuple: Returns a tuple containing:
#                - A nested dictionary structured by scenario, region, country, tax, technology, and fuel, each containing lists of calculated unit taxes for each year.
#                - An updated count of warnings encountered during processing.

#     Warnings:
#         Warnings are logged to "activity_warnings.txt" and include cases where fuels have factors assigned but no corresponding activity data.
#         Each warning increments the `count` variable, which is returned with the final data structure.
#     """
#     factors_by_fuel_temp = deepcopy(factors_by_fuel)
#     # verify_factors(factors_by_fuel)
#     tax_exceptions = ['IC', 'IMESI_Combust', 'IVA_Gasoil', 'Otros_Gasoil']
#     fuel_exceptions = ['ELECTRICIDAD', 'HIDROGENO']
    
#     # Define the inital year to interpolation
#     index_initial_year_interpolation = time_vector.index(params_tier2['year_ini_inter'])
    
#     try:
#         year_index = time_vector.index(year)
#     except ValueError:
#         print(f"The year {year} is not found in the time_vector.")
#         return {}
    
#     unit_taxes = {}
#     unit_taxes_percentage = {}
    
#     # Open a file to write the warnings
#     with open("activity_warnings.txt", "w") as file:
#         for scenario in factors_by_fuel:
#             if scenario != 'BAU':
#                 file.write(f"#***********************************************************************************************#\n{scenario}:\n")
#                 unit_taxes[scenario] = {}
#                 unit_taxes_percentage[scenario] = {}
#                 for region in factors_by_fuel[scenario]:
#                     unit_taxes[scenario][region] = {}
#                     unit_taxes_percentage[scenario][region] = {}
#                     for country in factors_by_fuel[scenario][region]:
#                         unit_taxes[scenario][region][country] = {}
#                         unit_taxes_percentage[scenario][region][country] = {}
#                         scenario_data = dict_scen[scenario][region][country]
#                         fiscal_gap = fiscal_gaps[scenario][region][country]
#                         factors_config = factors_by_fuel[scenario][region][country]
#                         factors_by_fuel = factors_by_fuel_temp
#                         redistribute_factors_if_no_activity(factors_by_fuel, dict_scen, year_index, params_tier2, tax_exceptions, fuel_exceptions, time_vector, dict_activities_out)  # Redistribute factors if needed
#                         verify_factors(factors_by_fuel)

                        
                        
#                         # if scenario == 'NETZERO':
#                         #     # print(scenario, factors_config)
#                         #     sys.exit()
#                         for tax, techs in factors_config.items():
#                             file.write(f"################################################################################\n{tax}:\n")
#                             unit_taxes[scenario][region][country][tax] = {}
#                             unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'] = {}
#                             unit_taxes_percentage[scenario][region][country][tax] = {}
#                             unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'] = {}
#                             total_gap_for_tax = fiscal_gap * techs['factor']  # Tax specific factor
#                             type_tax = dict_activities_out['Relation Tax-Activity'][tax]
                            
#                             for tech, fuels in techs['techs'].items():
#                                 file.write(f"{tech}:\n")
#                                 unit_taxes[scenario][region][country][tax][tech] = {}
#                                 unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech] = {}
#                                 unit_taxes_percentage[scenario][region][country][tax][tech] = {}
#                                 unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech] = {}
#                                 for fuel_key, factor_data in fuels.items():
#                                     if fuel_key == 'factor':  # Skip the 'factor' key
#                                         total_gap_for_tax_tech = total_gap_for_tax * fuels['factor']
#                                         continue
#                                     for fuel, factor_value in factor_data.items():
#                                         activity_data, activity_amount_0 = get_activity_for_fuel(scenario_data, tech, fuel, year_index, params_tier2, time_vector, type_tax)
#                                         unit_convert_percentage = dict_mult_depr[tax][tech][fuel]
#                                         if float(unit_convert_percentage) == float(0):
#                                             unit_convert_percentage = 1
                                        
#                                         tax_contribution_for_fuel = dict_tax_scen[scenario][region][country][tax][tech][fuel][year_index]
#                                         if params_tier2['interpolate']:
#                                             tax_actual_0 = dict_scen[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel][index_initial_year_interpolation]
                                        
#                                         if activity_data == 0.0 and factor_value != 0.0:
#                                             file.write(f"\"Warning: No activity for {fuel}, but it has a factor assigned.\"\n")
#                                             count += 1
#                                         tax_adjustment = total_gap_for_tax_tech * factor_value
#                                         if float(activity_data) > 0.0:
#                                             unit_tax = (tax_adjustment + tax_contribution_for_fuel) / activity_data
#                                             unit_tax_percentage = (tax_adjustment + tax_contribution_for_fuel) / (activity_data*unit_convert_percentage)
                                            
#                                         else:
#                                             unit_tax = 0  # No activity to distribute the tax
#                                             unit_tax_percentage = 0
                                            
#                                         if params_tier2['interpolate'] and float(activity_amount_0) > 0.0:
#                                             unit_tax_0 = tax_actual_0 / activity_amount_0
#                                             unit_tax_percentage_0 = tax_actual_0 / (activity_amount_0*unit_convert_percentage)
#                                         else:
#                                             if params_tier2['interpolate']:
#                                                 if tax_actual_0 != 0 or float(tax_actual_0) != 0.0:
#                                                     unit_tax_0 = 0
#                                                     unit_tax_percentage_0 = 0
#                                                 else:
#                                                     unit_tax_0 = 0 # It's forcefully normalized 
#                                                     # Because tax_actual_0=0 and activity_amount_0=0 so,
#                                                     # 0/0 isn't posible, but if is assing 1, before when
#                                                     # apply the formula tax_actual_0=unit_tax_0*activity_amount
#                                                     # tax_actual_0=1*0=0
#                                                     unit_tax_percentage_0 = 0
                                        
#                                         values_list = ['' for year in time_vector]
#                                         values_list[year_index] = unit_tax
#                                         values_list_percentage = ['' for year in time_vector]
#                                         values_list_percentage[year_index] = unit_tax_percentage
#                                         incomes_list = ['' for _ in time_vector]
#                                         incomes_list[year_index] = tax_contribution_for_fuel
#                                         if tax in tax_exceptions and unit_tax != np.float64(0.0) and scenario == 'NETZERO':
#                                         # if tax in tax_exceptions and fuel in fuel_exceptions and unit_tax != np.float64(0.0) and scenario == 'No':
#                                             values_list[year_index] = 0
#                                             values_list_percentage[year_index] = 0
#                                             incomes_list[year_index] = 0
#                                         elif tax in tax_exceptions and unit_tax == np.float64(0.0) and scenario == 'NETZERO':
#                                         # elif tax in tax_exceptions and fuel in fuel_exceptions and unit_tax == np.float64(0.0) and scenario == 'No':
#                                             values_list[year_index] = unit_tax
#                                             values_list_percentage[year_index] = unit_tax_percentage
#                                             incomes_list[year_index] = tax_adjustment + tax_contribution_for_fuel
#                                         else:
#                                             values_list[year_index] = unit_tax
#                                             values_list_percentage[year_index] = unit_tax_percentage
#                                             incomes_list[year_index] = tax_adjustment + tax_contribution_for_fuel
                                        
#                                         # if scenario == 'NETZERO' and tech == 'CamionesA' and fuel == 'GASOLINA/ALCOHOL' and tax == 'IVA_Elec':
#                                         #     print(scenario)
#                                         #     print(tech)
#                                         #     print('unit_tax:',unit_tax)
#                                         #     print('unit_tax_percentage:',unit_tax_percentage)
#                                         #     print('factores:',tax_adjustment, tax_contribution_for_fuel,activity_data,unit_convert_percentage)
#                                         #     print('tax_adjustment:',total_gap_for_tax_tech,factor_value)
#                                         #     sys.exit(9)

                                        
#                                         if params_tier2['interpolate']:
#                                             values_list[index_initial_year_interpolation] = unit_tax_0
#                                             values_list_percentage[index_initial_year_interpolation] = unit_tax_percentage_0
#                                             incomes_list[index_initial_year_interpolation] = tax_actual_0
#                                             if tax in tax_exceptions and unit_tax_0 != np.float64(0.0) and scenario == 'NETZERO':
#                                                 values_list[index_initial_year_interpolation] = 0
#                                                 values_list_percentage[index_initial_year_interpolation] = 0
#                                             elif tax in tax_exceptions and unit_tax_0 == np.float64(0.0) and scenario == 'NETZERO':
#                                                 values_list[index_initial_year_interpolation] = unit_tax_0
#                                                 values_list_percentage[index_initial_year_interpolation] = unit_tax_percentage_0
#                                             else:
#                                                 values_list[index_initial_year_interpolation] = unit_tax_0
#                                                 values_list_percentage[index_initial_year_interpolation] = unit_tax_percentage_0
                                            
#                                             # for y in range(0,index_initial_year_interpolation):
#                                             #     values_list[y] = values_list[index_initial_year_interpolation]
#                                             #     values_list_percentage[y] = values_list_percentage[index_initial_year_interpolation]
#                                             #     incomes_list[y] = incomes_list[index_initial_year_interpolation]
#                                             # print(values_list)
#                                             values_list = interpolation_to_end(time_vector, \
#                                                 params_tier2['year_ini_inter'], values_list, 'interme', 'Population')
#                                             # print(values_list)
#                                             # sys.exit()
#                                             values_list_percentage = interpolation_to_end(time_vector, \
#                                                 params_tier2['year_ini_inter'], values_list_percentage, 'interme', 'Population')
#                                             incomes_list = interpolation_to_end(time_vector, \
#                                                 params_tier2['year_ini_inter'], incomes_list, 'interme', 'Population')

#                                         unit_taxes[scenario][region][country][tax][tech][fuel] = values_list
#                                         unit_taxes[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = incomes_list
#                                         unit_taxes_percentage[scenario][region][country][tax][tech][fuel] = values_list_percentage
#                                         unit_taxes_percentage[scenario][region][country][f'Transport Tax {tax} [$]'][tech][fuel] = incomes_list
#                                     file.write("-----------------------------------------------------------------------------\n")

#     return unit_taxes, unit_taxes_percentage, count


def integrate_unit_taxes_into_dict_scen(unit_taxes, dict_scen):
    """
    Integrates unit tax data from 'unit_taxes' into 'dict_scen' by adding them directly at the level of activities under modified names for each tax type.
    This integration allows for the streamlined use of unit tax data directly within scenario analyses without the need to navigate additional nested dictionaries.

    Args:
        unit_taxes (dict): Dictionary containing the unit taxes organized by scenario, region, country, tax, technology, and fuel.
        dict_scen (dict): Scenario dictionary that needs to be updated with the unit tax information.

    Returns:
        dict: The updated 'dict_scen' dictionary, which now includes unit taxes directly associated with each relevant entry, facilitating easier access and manipulation.
    """
    for scenario_case in dict_scen:
        for scenario in dict_scen[scenario_case]:
            for region in dict_scen[scenario_case][scenario]:
                for country in dict_scen[scenario_case][scenario][region]:
                    # Update dict_scen with unit taxes directly at the country level
                    dict_scen[scenario_case][scenario][region][country].update({
                        f"Unit Tax {tax}": unit_taxes[scenario][region][country][tax]
                        for tax in unit_taxes[scenario][region][country]
                    })
    
    return dict_scen

def changes_tax_keys_unit_taxes_dict(unit_taxes):
    """
    Renames the tax keys in the 'unit_taxes' dictionary by appending 'Unit Tax' to each tax key to clarify their purpose and differentiate them within the data structure.
    This renaming helps in identifying unit tax values directly and ensures consistency across different data manipulation or visualization functions.

    Args:
        unit_taxes (dict): Dictionary containing unit taxes organized by scenario, region, country, tax, technology, and fuel.

    Returns:
        dict: The updated 'unit_taxes' dictionary with modified tax names for clearer identification and access.
    """
    for scenario in unit_taxes:
        for region in unit_taxes[scenario]:
            for country in unit_taxes[scenario][region]:
                # Update the tax keys within the unit_taxes dictionary for clarity
                for tax in list(unit_taxes[scenario][region][country].keys()):
                    if not 'Transport' in tax:
                        new_key = f"Unit Tax {tax}"
                        unit_taxes[scenario][region][country][new_key] = unit_taxes[scenario][region][country].pop(tax)
    
    return unit_taxes


def take_factors_distribution(sheet_name, column_name, excel_name, scenario_list, regions_list, country_list):
    """
    Constructs a nested dictionary of factors by fuel and tax from an Excel file for specified scenarios, regions, and countries, 
    excluding the 'BAU' (Business As Usual) scenario. This setup facilitates easy access to factor data for various geographical 
    and scenario-based configurations in subsequent analyses.

    Args:
        sheet_name (str): The name of the Excel sheet from which to read the factors.
        excel_name (str): The path to the Excel file containing the necessary factor data.
        scenario_list (list): A list of scenarios for which the factors need to be configured.
        regions_list (list): A list of regions to be included in the factor distribution.
        country_list (list): A list of countries to be included under each region.

    Returns:
        tuple: Returns a tuple containing two elements:
            factors_by_tax (dict): A dictionary containing the factors associated directly with tax categories.
            factors_by_fuel (dict): A deeply nested dictionary structured by scenario, region, and country,
                                    where each entry contains the factor data relevant to that specific classification.
    """
    # Read the factors from the Excel file
    factors_by_tax, factors_by_fuel = read_factors(excel_name, sheet_name, column_name)

    # Construct the factors by fuel configuration for each scenario, region, and country
    factors_by_fuel = {
        scen: {
            region: {
                country: factors_by_fuel for country in country_list
            } for region in regions_list
        } for scen in scenario_list if scen != 'BAU'
    }

    return factors_by_tax, factors_by_fuel
