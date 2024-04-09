from keras.utils.vis_utils import plot_model
import pydot
import pydotplus
import graphviz
import visualkeras
from model import MakeModel

model=MakeModel()
# Tableau :
plot_model(model, to_file='model_plot2.png', show_shapes=True, show_layer_names=True)

# 3D Shape :
visualkeras.layered_view(model, legend=True, to_file='model_layered2.png')
