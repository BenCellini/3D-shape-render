
import numpy as np
import matplotlib as mpl
import pyvista as pv
from scipy.spatial.transform import Rotation as R


class Cylinder3D:
    def __init__(self, size=1.0, color_gradient_axis='z', cmap=None):
        """ Render a 3D cylinder.
        """

        self.size = size
        self.color_gradient_axis = color_gradient_axis

        if cmap is None:
            self.cmap = mpl.colors.ListedColormap([(128/255, 200/255, 255/255)])
        else:
            self.cmap = cmap

        self.axis_index_map = {'x': 0, 'y': 1, 'z': 2}

    def render(self, x=0, y=0, z=0, roll=0.0, pitch=0.0, yaw=0.0, zoom=3.0, show=False):
        """ Render the 3D cylinder.
        """

        # Create a cylinder (aligned with Z)
        cylinder = pv.Cylinder(center=(x, y, z),
                               direction=(0, 0, 1),
                               radius=0.2*self.size,
                               height=self.size,
                               resolution=1000,
                               capping=False)

        # Get the coordinates of each point along the set axis
        color_points = cylinder.points[:, self.axis_index_map[self.color_gradient_axis]]

        # Add Z as a scalar array to the mesh
        cylinder[self.color_gradient_axis] = color_points

        # Convert RPY to rotation matrix (ZYX = yaw-pitch-roll)
        r = R.from_euler('XYZ', [yaw, pitch, roll], degrees=True)
        rotation_matrix = r.as_matrix()

        # Apply transformation
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        cylinder.transform(transform, inplace=True)

        # Plot from a fixed front-facing camera (along +X)
        plotter = pv.Plotter(off_screen=True, window_size=(300, 300))
        plotter.add_mesh(cylinder,
                         lighting=False,
                         scalars=self.color_gradient_axis,
                         cmap=self.cmap,
                         show_edges=False,
                         show_scalar_bar=False)

        plotter.camera_position = [
            (zoom, 0, 0),  # camera position (looking along +X)
            (0, 0, 0),  # focal point
            (0, 0, 1)  # view-up direction
        ]

        # Render and capture RGB image as numpy array
        rgb_image = plotter.screenshot(return_img=True)
        gray_image = pv.Texture(rgb_image).to_grayscale().to_array()

        if show:
            plotter.show()

        return rgb_image, gray_image