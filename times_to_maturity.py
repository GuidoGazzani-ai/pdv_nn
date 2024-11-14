# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:46:41 2023

@author: Guido Gazzani
"""

import datetime

################################################## SPX FUNCTION

def extract_monthly_SPX(input_date):
    
    def is_third_friday(date):
        
    # Check if the given date is a Friday and is in the third week of the month.
        return date.weekday() == 4 and 15 <= date.day <= 21

    def days_to_next_third_fridays(start_date, num_months=15):
        days_needed = []
        current_date = start_date
    
        for _ in range(num_months):
            while not is_third_friday(current_date):
                current_date += datetime.timedelta(days=1)
            days_needed.append((current_date - start_date).days)
            current_date = (current_date.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)
    
        return days_needed
    
    def main(input_date):
        #input_date = input("Enter a date (YYYY-MM-DD): ")
        try:
            input_date = datetime.datetime.strptime(input_date, "%Y-%m-%d")
            days_needed = days_to_next_third_fridays(input_date)
    
            print("Days needed to reach the third Friday of each month for the next 2 years:")
            for i, days in enumerate(days_needed):
                current_date = input_date + datetime.timedelta(days=days)
                month_year = current_date.strftime("%B %Y")
                print(f"{month_year}: {days} days")
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
        return days_needed
    
    x=main(input_date)
    return [f'{num}.0days' for num in x]

################################################## VIX FUNCTION

def extract_monthly_VIX(input_date):
    
    def is_third_wednesday(date):
        # Check if the given date is a Wednesday and is in the third week of the month.
        return date.weekday() == 2 and 15 <= date.day <= 21

    def days_to_next_third_wednesdays(start_date, num_months=10):
        days_needed = []
        current_date = start_date

        for _ in range(num_months):
            while not is_third_wednesday(current_date):
                current_date += datetime.timedelta(days=1)
            days_needed.append((current_date - start_date).days)
            current_date = (current_date.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)

        return days_needed

    def main(input_date):
        #input_date = input("Enter a date (YYYY-MM-DD): ")
        try:
            input_date = datetime.datetime.strptime(input_date, "%Y-%m-%d")
            days_needed = days_to_next_third_wednesdays(input_date)

            print("Days needed to reach the third Wednesday of each month for the next 2 years:")
            for i, days in enumerate(days_needed):
                current_date = input_date + datetime.timedelta(days=days)
                month_year = current_date.strftime("%B %Y")
                print(f"{month_year}: {days} days")
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
        return days_needed

    x=main(input_date)
    return [f'{num}.0days' for num in x]
    
    

def maturities_joint_func(maturities_vix,maturities_spx):
    # Extract numbers before the dot and convert them to integers
    vix_numbers = [int(m.split('.')[0]) for m in maturities_vix]
    spx_numbers = [int(m.split('.')[0]) for m in maturities_spx]
    
    # Combine the two lists and sort them
    sorted_maturities = sorted(vix_numbers + spx_numbers)
    
    # Convert the sorted numbers back to strings with 'days' appended
    sorted_maturities_with_days = [f'{num}.0days' for num in sorted_maturities]
    return sorted_maturities_with_days

