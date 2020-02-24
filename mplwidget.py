import matplotlib
matplotlib.use('Qt5Agg')
# matplotlib.rcParams['backend.qt4'] = 'PySide2'

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MplWidget(FigureCanvas):

    def __init__(self, parent=None):
        # super(MplWidget, self).__init__(Figure())
        super().__init__(Figure())

        self.setParent(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
