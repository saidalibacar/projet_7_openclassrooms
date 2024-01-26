# bokeh_server.py

import argparse
from bokeh.models import TextInput, Button, Div, ColumnDataSource, Quad, Label
from bokeh.layouts import column, row
from bokeh.plotting import curdoc, figure
import requests

def make_request(id):
    url = f"http://127.0.0.1:8000/predict_id?id={id}"  # Replace with your actual URL
    response = requests.get(url)
    return response.json()

def update():
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

# Set up command-line arguments
parser = argparse.ArgumentParser(description="Bokeh Server Application")
parser.add_argument('--id', type=int, default=34, help='The value for the "id" parameter')
args = parser.parse_args()

# Set up Bokeh widgets
text_input = TextInput(value=str(args.id), title="Client ID:")
button = Button(label="Get Prediction", button_type="success")
result_text = Div()

# Set up Gauge Plot
gauge_source = ColumnDataSource(data={'left': [0], 'right': [0], 'bottom': [0], 'top': [0]})

gauge = figure(height=200, width=400, tools='', x_range=(0, 1), y_range=(0, 1), title=None)
gauge_quad = gauge.quad(top='top', bottom='bottom', left='left', right='right', source=gauge_source)
gauge_label = Label(x=0.5, y=-0.1, text='', text_align='center', text_color='black')
gauge.add_layout(gauge_label)

gauge.axis.axis_label = None
gauge.axis.visible = False
gauge.grid.grid_line_color = None

# Set up callback
button.on_click(update)

# Set up layout
layout = column(text_input, button, result_text, row(gauge))

# Add layout to the current document
curdoc().add_root(layout)
















