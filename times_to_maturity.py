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

# =============================================================================
# 
# ####################################### ONLY SPX AND VIX MONTHLY OPTIONS ARE CONSIDERED
# 
# maturities_vix=[r'29.0days',r'57.0days',r'92.0days',r'120.0days',r'148.0days']
# maturities_spx=[r'31.0days',r'59.0days',r'87.0days',r'122.0days',r'150.0days',r'178.0days',r'241.0days',r'332.0days',r'345.0days']
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20201020'
# 
# maturities_vix=[r'19.0days',r'54.0days',r'82.0days',r'110.0days',r'138.0days',r'173.0days']
# maturities_spx=[r'21.0days',r'49.0days',r'84.0days',r'112.0days',r'140.0days',r'168.0days',r'294.0days',r'357.0days']
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20191129'
# 
# 
# maturities_vix=[r'22.0days',r'50.0days',r'85.0days',r'113.0days']
# maturities_spx=[r'24.0days',r'52.0days',r'80.0days',r'115.0days',r'143.0days',r'297.0days',r'352.0days']
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210427'
# 
# 
# maturities_vix=[r'15.0days',r'50.0days',r'78.0days',r'106.0days',r'141.0days',r'169.0days']
# maturities_spx=[r'17.0days',r'45.0days',r'80.0days',r'108.0days',r'136.0days',r'290.0days',r'353.0days']
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210601'
# 
# maturities_vix=[r'14.0days',r'49.0days',r'77.0days',r'105.0days',r'140.0days',r'168.0days',r'203.0days']
# maturities_spx=[r'16.0days',r'44.0days',r'79.0days',r'107.0days',r'135.0days',r'170.0days',r'289.0days',r'352.0days']
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210602'
# 
# maturities_vix=[r'13.0days',r'48.0days',r'76.0days',r'104.0days',r'139.0days',r'230.0days']
# maturities_spx=[r'15.0days',r'43.0days',r'78.0days',r'106.0days',r'134.0days',r'169.0days',r'232.0days',r'288.0days',r'351.0days']
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210603'
# 
# 
# maturities_vix=[r'12.0days',r'47.0days',r'75.0days',r'103.0days',r'138.0days',r'201.0days']
# maturities_spx=[r'14.0days',r'42.0days',r'77.0days',r'105.0days',r'133.0days',r'168.0days',r'231.0days',r'287.0days',r'350.0days']
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210604'
# 
# maturities_vix=[r'9.0days',r'44.0days',r'72.0days',r'100.0days',r'135.0days',r'163.0days']
# maturities_spx=[r'11.0days',r'39.0days',r'74.0days',r'102.0days',r'130.0days',r'165.0days',r'228.0days',r'284.0days',r'347.0days']
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210607'
# 
# 
# ###
# 
# maturities_vix=[r'8.0days',r'43.0days',r'71.0days',r'99.0days',r'134.0days',r'197.0days']
# maturities_spx=[r'10.0days',r'38.0days',r'73.0days',r'101.0days',r'129.0days',r'164.0days',r'227.0days',r'283.0days',r'346.0days']
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210608'
# 
# 
# maturities_vix=[r'7.0days',r'42.0days',r'70.0days',r'98.0days',r'133.0days',r'161.0days']
# maturities_spx=[r'9.0days',r'37.0days',r'72.0days',r'100.0days',r'128.0days',r'163.0days',r'226.0days',r'282.0days',r'345.0days']
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210609'
# 
# maturities_vix=[r'6.0days',r'41.0days',r'69.0days',r'97.0days',r'132.0days',r'160.0days']
# maturities_spx=[r'8.0days',r'36.0days',r'71.0days',r'99.0days',r'127.0days',r'162.0days',r'225.0days',r'281.0days',r'344.0days']
# maturities_joint=maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210610'
# 
# 
# maturities_vix=[r'5.0days',r'40.0days',r'68.0days',r'96.0days',r'131.0days',r'159.0days']
# maturities_spx=[r'7.0days',r'35.0days',r'70.0days',r'98.0days',r'126.0days',r'161.0days',r'224.0days',r'280.0days',r'343.0days']
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210611'
# 
# 
# maturities_vix=[r'2.0days',r'37.0days',r'65.0days',r'93.0days',r'128.0days',r'156.0days']
# maturities_spx=[r'4.0days',r'32.0days',r'67.0days',r'95.0days',r'123.0days',r'158.0days',r'221.0days',r'277.0days',r'340.0days']
# maturities_joint=maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210614'
# 
# 
# maturities_vix=[r'36.0days',r'64.0days',r'92.0days',r'127.0days',r'155.0days',r'183.0days']
# maturities_spx=[r'31.0days',r'66.0days',r'94.0days',r'122.0days',r'157.0days',r'185.0days',r'220.0days',r'276.0days',r'339.0days',r'367.0days']
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210615'
# 
# 
# 
# maturities_vix=extract_monthly_VIX('2021-06-16')[1:-4]+[r'189.0days']
# maturities_spx=extract_monthly_SPX('2021-06-16')[1:8]+extract_monthly_SPX('2021-06-16')[9:10]+extract_monthly_SPX('2021-06-16')[11:13]
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210616'
# 
# #################
# 
# maturities_vix=extract_monthly_VIX('2021-06-17')[:-5]+[r'188.0days']
# maturities_spx=extract_monthly_SPX('2021-06-17')[1:8]+extract_monthly_SPX('2021-06-17')[9:10]+extract_monthly_SPX('2021-06-17')[11:13]
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210617'
# 
# 
# 
# maturities_vix=extract_monthly_VIX('2021-06-18')[:-5]+[r'187.0days']
# maturities_spx=extract_monthly_SPX('2021-06-18')[1:8]+extract_monthly_SPX('2021-06-18')[9:10]+extract_monthly_SPX('2021-06-18')[11:13]
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210618'
# 
# 
# 
# maturities_vix=extract_monthly_VIX('2021-06-21')[:-5]+[r'184.0days']
# maturities_spx=extract_monthly_SPX('2021-06-21')[:8]+[r'297.0days']+extract_monthly_SPX('2021-06-21')[11:12]
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210621'
# 
# 
# 
# ################
# 
# 
# maturities_vix=extract_monthly_VIX('2021-06-22')[:-5]+[r'183.0days']
# maturities_spx=extract_monthly_SPX('2021-06-22')[:8]+[r'296.0days']+extract_monthly_SPX('2021-06-22')[11:12]
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210622'
# 
# 
# 
# maturities_vix=extract_monthly_VIX('2021-06-23')[:-5]+[r'182.0days']
# maturities_spx=extract_monthly_SPX('2021-06-23')[:8]+[r'295.0days']+extract_monthly_SPX('2021-06-23')[11:12]
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210623'
# 
# 
# 
# maturities_vix=extract_monthly_VIX('2021-06-24')[:-5]+[r'181.0days']
# maturities_spx=extract_monthly_SPX('2021-06-24')[:8]+[r'294.0days']+extract_monthly_SPX('2021-06-24')[11:12]
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210624'
# 
# 
# maturities_vix=extract_monthly_VIX('2021-06-25')[:-5]+[r'180.0days']
# maturities_spx=extract_monthly_SPX('2021-06-25')[:8]+[r'293.0days']+extract_monthly_SPX('2021-06-25')[11:12]
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210625'
# 
# ####################
# 
# 
# 
# maturities_vix=extract_monthly_VIX('2021-06-28')[:-5]+[r'177.0days']
# maturities_spx=extract_monthly_SPX('2021-06-28')[:8]+[r'292.0days']+extract_monthly_SPX('2021-06-28')[11:12]
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210628'
# 
# 
# 
# maturities_vix=extract_monthly_VIX('2021-06-29')[:-5]+[r'176.0days']
# maturities_spx=extract_monthly_SPX('2021-06-29')[:8]+[r'291.0days']+extract_monthly_SPX('2021-06-29')[11:12]
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210629'
# 
# 
# maturities_vix=extract_monthly_VIX('2021-06-30')[:-5]+[r'175.0days']
# maturities_spx=extract_monthly_SPX('2021-06-30')[:8]+[r'290.0days']+extract_monthly_SPX('2021-06-30')[11:12]
# maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
# maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
# day=r'/20210630'
# 
# =============================================================================




