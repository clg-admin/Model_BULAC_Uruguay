# -*- coding: utf-8 -*-
"""
Created: Feb. 03, 2025
Last updated: Feb. 04, 2025

@author: Climate Lead Group; Luis Victor-Gallardo, Jairo Quirós-Tortós,
        Andrey Salazar-Vargas
Suggested citation: UNEP (2022). Is Natural Gas a Good Investment for Latin 
                                America and the Caribbean? From Economic to 
                                Employment and Climate Impacts of the Power
                                Sector. https://wedocs.unep.org/handle/20.500.11822/40923
"""

import pandas as pd
from scipy.stats import qmc
from openpyxl.styles import numbers

def generate_lhs_data():
    # 1. Read the Excel file and the specific sheet
    excel_file = "Taxes_Factors_Original.xlsx"
    sheet_name = "FACTORS"
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    inputs_excel = "data_inputs.xlsx"
    input_sheet = "2_general"
    df_inputs = pd.read_excel(inputs_excel, sheet_name=input_sheet)

    # Filter the row where 'Parameter' is 'number_comb'
    filter_condition = df_inputs['Parameter'] == 'number_comb'

    # Get the value from the 'Value' column in that row
    number_comb = df_inputs.loc[filter_condition, 'Value'].values[0]

    # 2. Set the number of rows to generate (equal to the number of rows in the original file)
    N = len(df)  # Use the number of rows from the original file

    # 3. Ask the user for the number of factor columns to generate
    num_factor_columns = number_comb

    # 4. Generate samples using Latin Hypercube Sampling (LHS)
    # Create an LHS generator with the dimension equal to the number of factor columns
    lhs_generator = qmc.LatinHypercube(d=num_factor_columns)

    # Generate uniform samples in the range [0, 1]
    samples = lhs_generator.random(n=N)

    # 5. Adjust the values so that the sum of each column is 100
    # Scale the samples so that the sum of each column is 100
    adjusted_samples = samples / samples.sum(axis=0)

    # 6. Create a new DataFrame with the generated values
    # Name the factor columns as FACTORS_1, FACTORS_2, etc.
    factor_column_names = [f"FACTORS_{i+1}" for i in range(num_factor_columns)]
    new_data = pd.DataFrame(adjusted_samples, columns=factor_column_names)

    # Add the TAX column (we will use the same tax names from the original file)
    new_data["TAX"] = df["TAX"].values

    # Reorder the columns so that TAX comes first
    new_data = new_data[["TAX"] + factor_column_names]

    # 7. Save the result to a new Excel file
    new_file = "Taxes_Factors.xlsx"
    with pd.ExcelWriter(new_file, engine='openpyxl') as writer:
        new_data.to_excel(writer, index=False, sheet_name='FACTORS')

        # Access the workbook and worksheet to apply percentage formatting
        workbook = writer.book
        worksheet = writer.sheets['FACTORS']

        # Apply percentage formatting to the factor columns
        for col_idx, col_name in enumerate(factor_column_names, start=2):  # Start from column 2 (skip 'TAX')
            for row_idx in range(2, len(new_data) + 2):  # Start from row 2 (skip header)
                cell = worksheet.cell(row=row_idx, column=col_idx)
                cell.number_format = numbers.FORMAT_PERCENTAGE

    print(f"LHS data generated and saved to {new_file} with percentage formatting.")

# Allow the script to be run directly
if __name__ == "__main__":
    generate_lhs_data()