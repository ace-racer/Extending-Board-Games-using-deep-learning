from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from matplotlib.ticker import NullLocator
import math
import cv2
import os

NUM_IMAGES_FOR_ELEVATION_ANGLE = 4 # 360 * 4
NUM_ELEVATION_ANGLES = 5 # 50
STL_FILES_LOCATION = "H:\\AR-ExtendingOnlineGames\\3d_pieces\\Chess_Set_-_Print_Friendly\\files"
GENERATED_FILES_BASE_LOC = "H:\\AR-ExtendingOnlineGames\\3d_pieces\\images"

# constants
stl_file_names = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]
MIN_ELEVATION_ANGLE = 10
MAX_ELEVATION_ANGLE = 75

# calculated values
total_images = NUM_IMAGES_FOR_ELEVATION_ANGLE * NUM_ELEVATION_ANGLES

for stl_file_name in stl_file_names:
    print("Generating images for {0}.".format(stl_file_name))
    complete_stl_file_location = os.path.join(STL_FILES_LOCATION, stl_file_name + ".stl")
    # Create a new plot
    figure = pyplot.figure(frameon=False)
    axes = mplot3d.Axes3D(figure)

    # Load the STL files and add the vectors to the plot
    image_mesh = mesh.Mesh.from_file(complete_stl_file_location)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(image_mesh.vectors))

    # Auto scale to the mesh size
    scale = image_mesh.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)
    axes.set_axis_off()

    elevation_angle_range = MAX_ELEVATION_ANGLE - MIN_ELEVATION_ANGLE
    unit_elevation_angle = elevation_angle_range / NUM_ELEVATION_ANGLES
    current_elevation_angle = MIN_ELEVATION_ANGLE
    elevation_loop_counter = 0

    current_piece_images_location = os.path.join(GENERATED_FILES_BASE_LOC, stl_file_name)
    if not os.path.exists(current_piece_images_location):
        os.makedirs(current_piece_images_location)

    while current_elevation_angle <= MAX_ELEVATION_ANGLE:
        unit_rotation_for_elevation = 360 / NUM_IMAGES_FOR_ELEVATION_ANGLE

        for itr in range(NUM_IMAGES_FOR_ELEVATION_ANGLE):
            rotation_angle = unit_rotation_for_elevation * itr
            axes.view_init(azim=rotation_angle, elev=current_elevation_angle)
            image_name = "e{0}_a{1}".format(round(current_elevation_angle, 2), round(rotation_angle, 2))
            file_location = os.path.join(current_piece_images_location, image_name + ".jpg")
            pyplot.savefig(file_location, bbox_inches = 'tight', pad_inches = 0)

            image_number = elevation_loop_counter * NUM_IMAGES_FOR_ELEVATION_ANGLE + itr
            print("Generated image {0}/{1} => {2}.".format(image_number, total_images, image_name))
        
        elevation_loop_counter += 1
        current_elevation_angle += unit_elevation_angle
