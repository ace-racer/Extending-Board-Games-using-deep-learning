from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import math
import cv2
import os

GENERATED_FILES_BASE_LOC = "H:\\AR-ExtendingOnlineGames\\3d_pieces\\images"

# Create a new plot
# figure = pyplot.figure(frameon=False)
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

# Load the STL files and add the vectors to the plot
image_mesh = mesh.Mesh.from_file('H:\\AR-ExtendingOnlineGames\\3d_pieces\\Chess_Set_-_Print_Friendly\\files\\King.stl')
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(image_mesh.vectors))

# Auto scale to the mesh size
scale = image_mesh.points.flatten(-1)
axes.auto_scale_xyz(scale, scale, scale)
# axes.set_axis_off()

# Show the plot to the screen
# pyplot.show()
num_images_required = 3

# 1 unit rotation in radians
unit_rotation_radian = (2*math.pi) / num_images_required

# check 90 degree rotation
# image_mesh.rotate([0, 0, 1], math.pi/2)
pyplot.show()
# file_location = os.path.join(GENERATED_FILES_BASE_LOC, "img" + str(itr) + ".jpg")
#pyplot.savefig(file_location)
#print("Generated image...")

""" for itr in range(1, num_images_required + 1):
    rotation_angle_radian = unit_rotation_radian * itr
    image_mesh.rotate([0, 0, 1], rotation_angle_radian)
    file_location = os.path.join(GENERATED_FILES_BASE_LOC, "img" + str(itr) + ".jpg")
    pyplot.savefig(file_location)
    print("Generated image...")
    # pyplot.show() """
