import joblib
import pandas as pd
import requests
import shap
import numpy as np
import matplotlib.pyplot as plt
from bokeh.models import TextInput, Button, Div, ColumnDataSource, Label, Select, Span, Slider, HoverTool
from bokeh.layouts import column, row
from bokeh.plotting import curdoc, figure
from bokeh.models.widgets import Slider
from io import BytesIO
import base64
import mpld3
# Load the pre-trained model
model = joblib.load('lightgbm_model.joblib')

# Load test data
X_test_app = pd.read_csv('X_test_app.csv')
X_test_app['client_id'] = X_test_app.index

# Set up Bokeh widgets
text_input = TextInput(value="34", title="Client ID:")
variable_select = Select(title="Select Variable", options=[''] + X_test_app.columns.tolist(), value='')
x_feature_select = Select(title="Select X-axis Feature", options=[''] + X_test_app.columns.tolist(), value='')
y_feature_select = Select(title="Select Y-axis Feature", options=[''] + X_test_app.columns.tolist(), value='')
button = Button(label="Get Prediction", button_type="success")
result_text = Div()
# Replace TextInput with Slider for manual selection of client ID
client_id_slider = Slider(start=0, end=len(X_test_app) - 1, step=1, value=0, title="Select Client ID:")

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
bins_slider = Slider(start=1, end=50, value=30, step=1, title="Number of Bins")

hover_tool = HoverTool(
    tooltips=[
        ("Variable", "@left"),
        ("Frequency", "@top"),
    ],
    mode='vline'
)

histogram = figure(
    height=300, width=600, title="Histogram", tools=[hover_tool, 'pan', 'wheel_zoom', 'reset'],
    x_axis_label="Variable", y_axis_label="Frequency"
)
hist_quad = histogram.quad(top='top', bottom='bottom', left='left', right='right', source=hist_source)

selected_client_marker = Span(location=0, dimension='height', line_color='red', line_width=2)
histogram.add_layout(selected_client_marker)

# Set up Scatter Plot
scatter_source = ColumnDataSource(data={'x': [], 'y': []})
hover_tool_scatter = HoverTool(
    tooltips=[
        ("X", "@x"),
        ("Y", "@y"),
    ]
)

scatter_plot = figure(
    height=300, width=600, title="Scatter Plot", tools=[hover_tool_scatter, 'pan', 'wheel_zoom', 'reset'],
    x_axis_label="X-axis", y_axis_label="Y-axis", x_range=(0, 1)
)
scatter_plot.circle('x', 'y', source=scatter_source, size=8, color='navy', alpha=0.6)


def update_histogram():
    variable = variable_select.value
    bins = bins_slider.value

    if variable:
        hist, edges = np.histogram(X_test_app[variable], bins=bins)
        hist_source.data = {'top': hist, 'bottom': [0] * len(hist), 'left': edges[:-1], 'right': edges[1:]}
        selected_client_marker.location = int(text_input.value)
    else:
        hist_source.data = {'top': [], 'bottom': [], 'left': [], 'right': []}


def plot_global_feature_importance():
    features_for_shap = X_test_app.drop(columns=['client_id'])
    feature_names_list = features_for_shap.columns.tolist()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_for_shap)

    mean_abs_shap_values = np.mean(np.abs(shap_values[1]), axis=0)

    top_features_indices = np.argsort(mean_abs_shap_values)[-20:][::-1]
    top_features_names = [feature_names_list[i] for i in top_features_indices]
    top_features_values = mean_abs_shap_values[top_features_indices]

    global_importance_plot = figure(
        x_range=top_features_names, height=400, width=800, title="Global Feature Importance", tools='',
        toolbar_location=None, tooltips=[("Feature", "@x"), ("Importance", "@top")]
    )
    global_importance_plot.vbar(x=top_features_names, top=top_features_values, width=0.9, fill_color="navy", line_color="white")

    global_importance_plot.xaxis.major_label_orientation = "vertical"
    global_importance_plot.yaxis.axis_label = "Mean |SHAP Value|"

    return global_importance_plot


feature_names = X_test_app.columns


def plot_local_feature_importance(value):
    try:
        client_id = int(value)

        sample_instance = X_test_app[X_test_app['client_id'] == client_id].drop(columns=['client_id']).values.reshape(1, -1)
        class_index_to_visualize = 1

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_app.drop(columns=['client_id']))

        client_shap_values = shap_values[class_index_to_visualize][client_id, :]

        plt.figure(figsize=(10, 6))
        plt.bar(feature_names, client_shap_values)
        plt.title(f"Local Feature Importance for Client ID {client_id}")
        plt.xlabel("Feature")
        plt.ylabel("SHAP Value")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Display the plot in Bokeh using mpld3
        mpld3_fig = mpld3.fig_to_html(plt.gcf())

        # Clear any previous error messages
        local_feature_importance_div.text = ""
    except ValueError:
        local_feature_importance_div.text = "Please enter a valid integer for Client ID."



def update_scatter_plot():
    x_feature = x_feature_select.value
    y_feature = y_feature_select.value

    if x_feature and y_feature:
        scatter_source.data = {'x': X_test_app[x_feature], 'y': X_test_app[y_feature]}
    else:
        scatter_source.data = {'x': [], 'y': []}


def make_request(id):
    url = f"http://127.0.0.1:8000/predict_id?id={id}"
    response = requests.get(url)
    return response.json()


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


button.on_click(update)
variable_select.on_change('value', lambda attr, old, new: update_histogram())
x_feature_select.on_change('value', lambda attr, old, new: update_scatter_plot())
y_feature_select.on_change('value', lambda attr, old, new: update_scatter_plot())
text_input.on_change('value', lambda attr, old, new: plot_local_feature_importance())
bins_slider.on_change('value', lambda attr, old, new: update_histogram())
client_id_slider.on_change('value', lambda attr, old, new: plot_local_feature_importance(new))

global_importance_plot = plot_global_feature_importance()

# Set up layout
local_feature_importance_div = Div()
local_feature_importance_button = Button(label="Plot Local Feature Importance", button_type="success")
layout = column(
    text_input, variable_select, bins_slider, x_feature_select, y_feature_select, button, result_text,
    row(gauge, histogram, scatter_plot), global_importance_plot, local_feature_importance_button, local_feature_importance_div
)

local_feature_importance_button.on_click(plot_local_feature_importance)

# Add layout to the current document
curdoc().add_root(layout)

# Initial update of the histogram and scatter plot
update_histogram()
update_scatter_plot()
plot_global_feature_importance()
plot_local_feature_importance()























