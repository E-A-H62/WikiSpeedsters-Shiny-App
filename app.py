from shiny import App, render, ui

app_ui = ui.page_fluid(
    ui.h2("Hello Shiny!"),
    ui.input_slider("n", "Number of bins:", 1, 50, 25),
    ui.output_plot("plot")
)

def server(input, output, session):
    import matplotlib.pyplot as plt
    import numpy as np

    @output
    @render.plot
    def plot():
        x = np.random.randn(1000)
        fig, ax = plt.subplots()
        ax.hist(x, bins=input.n())
        return fig

app = App(app_ui, server)

