from bokeh.io import curdoc
from bokeh.models import TextInput, Button
from bokeh.layouts import column
import pytest
import numpy as np
import pandas as pd
import joblib
from io import BytesIO
from matplotlib.figure import Figure
import shap
from bokeh.models import ColumnDataSource, Select, Slider, HoverTool, Span
from bokeh.plotting import figure
from bokeh.layouts import row

def update_text_input(text_input):
    text_input.value = "Updated"

@pytest.fixture
def bokeh_app():
    # Your Bokeh app code here
    def modify_doc(doc):
        text_input = TextInput(value="Hello", title="Label:", name="Label")
        button = Button(label="Update", name="Update")

        button.on_click(lambda: update_text_input(text_input))

        layout = column(text_input, button)
        doc.add_root(layout)

    # Create a Bokeh server application
    curdoc().clear()
    modify_doc(curdoc())
    
    # Return the modified Bokeh app
    return curdoc()

def test_bokeh_app(bokeh_app):
    # Test the Bokeh app without a WebDriver
    layout = bokeh_app.roots[0]  # Assuming the layout is the first and only root
    text_input = layout.select_one({'name': 'Label'})
    assert text_input.value == "Hello"

    update_text_input(text_input)

    # Ensure the button logic updates the text input
    assert text_input.value == "Updated"