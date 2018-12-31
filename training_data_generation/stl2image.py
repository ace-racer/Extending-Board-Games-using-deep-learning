from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import math
import cv2
import os

GENERATED_FILES_BASE_LOC = "H:\\AR-ExtendingOnlineGames\\3d_pieces\\images"

# Create a new plot
figure = pyplot.figure(frameon=False)
#figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

# Load the STL files and add the vectors to the plot
image_mesh = mesh.Mesh.from_file('H:\\AR-ExtendingOnlineGames\\3d_pieces\\Chess_Set_-_Print_Friendly\\files\\King.stl')
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(image_mesh.vectors))

# Auto scale to the mesh size
scale = image_mesh.points.flatten(-1)
axes.auto_scale_xyz(scale, scale, scale)
axes.set_axis_off()

num_images_required = 6

# 1 unit rotation
unit_rotation = 360/num_images_required

for itr in range(num_images_required):
    rotation_angle = unit_rotation * itr
    axes.view_init(azim=rotation_angle)
    file_location = os.path.join(GENERATED_FILES_BASE_LOC, "img" + str(itr) + ".jpg")
    pyplot.savefig(file_location)
    print("Generated image...")
    # pyplot.show()
