# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 17:34:55 2023

@author: sarah
"""

import tkinter as tk
import pandas as pd

# Function to handle button click event
def handle_button_click():
    address_input = address_entry.get()
    match = dfOverbrook.loc[dfOverbrook['FULL_ADDRESS_EN'] == address_input]
    if not match.empty:
        result_label.config(text='The building information is: {}, {}, {}, Ward {}, and Minimum lotsize of {}'
                            .format(match.FULL_ADDRESS_EN.to_string(index=False, header=False),
                                    match.POSTAL_CODE.to_string(index=False, header=False),
                                    match.MUNICIPALITY.to_string(index=False, header=False),
                                    match.WARD.values,
                                    match.MINLOSIZELEFT.values))
    else:
        result_label.config(text='No matches found.')

# Create the main window
root = tk.Tk()
root.title('Building Information')

# Load the data
dfOverbrook = pd.read_csv('extractedOverbrook.csv')

# Create GUI widgets
address_label = tk.Label(root, text='Enter address:')
address_label.pack()
address_entry = tk.Entry(root)
address_entry.pack()
result_label = tk.Label(root, text='')
result_label.pack()
submit_button = tk.Button(root, text='Submit', command=handle_button_click)
submit_button.pack()

# Start the GUI event loop
root.mainloop()
