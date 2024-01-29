import argparse
from bokeh.models import TextInput, Button, Div, ColumnDataSource, Quad, Label, Select
from bokeh.layouts import column, row
from bokeh.plotting import curdoc, figure, show
from bokeh.models import Span

import requests
import joblib
import pandas as pd
import shap
import numpy as np

# Load the pre-trained model
model = joblib.load('lightgbm_model.joblib')

# Assuming 'X_test_app.csv' is in the same directory as your Bokeh server file
X_test_app = pd.read_csv('X_test_app.csv')

# Add a new column 'client_id' based on the DataFrame index
X_test_app['client_id'] = X_test_app.index

# Set up Bokeh widgets
text_input = TextInput(value="34", title="Client ID:")
variable_select = Select(title="Select Variable", options=[''] + ['AMT_CREDIT', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'ANNUITY_TO_INCOME_RATIO'], value='')
x_feature_select = Select(title="Select X-axis Feature", options=[''] + ['AMT_CREDIT', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'ANNUITY_TO_INCOME_RATIO'], value='')
y_feature_select = Select(title="Select Y-axis Feature", options=[''] + ['AMT_CREDIT', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'ANNUITY_TO_INCOME_RATIO'], value='')
button = Button(label="Get Prediction", button_type="success")
result_text = Div()

# Set up Gauge Plot
gauge_source = ColumnDataSource(data={'left': [0], 'right': [0], 'bottom': [0], 'top': [0]})

gauge = figure(height=300, width=600, tools='', x_range=(0, 1), y_range=(0, 1), title=None)
gauge_quad = gauge.quad(top='top', bottom='bottom', left='left', right='right', source=gauge_source)
gauge_label = Label(x=0.5, y=-0.1, text='', text_align='center', text_color='black')
gauge.add_layout(gauge_label)

gauge.axis.axis_label = None
gauge.axis.visible = False
gauge.grid.grid_line_color = None

# Set up Histogram Plot
hist_source = ColumnDataSource(data={'top': [], 'bottom': [], 'left': [], 'right': []})

histogram = figure(height=300, width=600, title="Histogram", tools='', x_axis_label="Variable", y_axis_label="Frequency")
hist_quad = histogram.quad(top='top', bottom='bottom', left='left', right='right', source=hist_source)

# Add a vertical line to mark the selected client's position on the histogram
selected_client_marker = Span(location=0, dimension='height', line_color='red', line_width=2)
histogram.add_layout(selected_client_marker)

# Set up Scatter Plot
scatter_source = ColumnDataSource(data={'x': [], 'y': []})

scatter_plot = figure(height=300, width=600, title="Scatter Plot", tools='', x_axis_label="X-axis", y_axis_label="Y-axis")
scatter_plot.circle('x', 'y', source=scatter_source, size=8, color='navy', alpha=0.6)

# Function to update the histogram based on the selected variable
def update_histogram():
    variable = variable_select.value

    # Check if a valid variable is selected
    if variable:
        hist, edges = np.histogram(X_test_app[variable], bins=30)
        hist_source.data = {'top': hist, 'bottom': [0] * len(hist), 'left': edges[:-1], 'right': edges[1:]}
        selected_client_marker.location = int(text_input.value)
    else:
        # If no variable is selected, clear the histogram
        hist_source.data = {'top': [], 'bottom': [], 'left': [], 'right': []}

# Function to update the scatter plot based on the selected features
def update_scatter_plot():
    x_feature = x_feature_select.value
    y_feature = y_feature_select.value

    # Check if valid features are selected
    if x_feature and y_feature:
        scatter_source.data = {'x': X_test_app[x_feature], 'y': X_test_app[y_feature]}
    else:
        # If no or only one feature is selected, clear the scatter plot
        scatter_source.data = {'x': [], 'y': []}

# Function to make the prediction request
def make_request(id):
    url = f"http://127.0.0.1:8000/predict_id?id={id}"  # Replace with your actual URL
    response = requests.get(url)
    return response.json()

# Function to update the Bokeh layout
def update():
    update_histogram()
    update_scatter_plot()

    client_id = int(text_input.value)
    result = make_request(client_id)
    result_text.text = f"True Prediction: {result['True Prediction']}\n" \
                       f"Binary Prediction: {result['Binary Prediction']}\n" \
                       f"Class: {result['Class']}"

    binary_prediction = result['Binary Prediction']
    credit_status = "CREDIT ACCEPTED" if binary_prediction == 1 else "CREDIT REFUSED"

    result_text.text += f"\nCredit Status: {credit_status}"

    gauge_source.data = {'left': [0], 'right': [result['True Prediction']], 'bottom': [0], 'top': [1]}
    gauge_quad.glyph.fill_color = 'green' if binary_prediction == 1 else 'red'
    gauge_label.text = credit_status

# Set up callback
button.on_click(update)
variable_select.on_change('value', lambda attr, old, new: update_histogram())
x_feature_select.on_change('value', lambda attr, old, new: update_scatter_plot())
y_feature_select.on_change('value', lambda attr, old, new: update_scatter_plot())

# Set up layout
layout = column(text_input, variable_select, x_feature_select, y_feature_select, button, result_text, row(gauge, histogram, scatter_plot))

# Add layout to the current document
curdoc().add_root(layout)

# Initial update of the histogram and scatter plot
update_histogram()
update_scatter_plot()

























