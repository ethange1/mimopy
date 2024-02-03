import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class Array:
    """Base class for array objects.

    Parameters
    ----------
    num_antennas : int
        Number of antennas in the array.
    coordinates : array_like
        Coordinates of the antennas. The shape of the array must be (num_antennas, 3).
    weights : array_like, optional
        Weights of the antennas. If not given, all antennas are assumed to have
        unit weight.
    """

    def __init__(self, num_antennas, coordinates="[0, 0, 0]"):
        self.num_antennas = num_antennas
        self.coordinates = np.array(coordinates)
        self.weights = np.ones(num_antennas)
        self.marker = "o"
        self.power = 1
        self.name = "Array"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __len__(self):
        return self.num_antennas

    @classmethod
    def initialize_ula(cls, num_antennas, ax="x", array_center=[0, 0, 0], **kwargs):
        """Empties the array and creates a half-wavelength spaced,
        uniforom linear array along the desired axis centered at the origin.

        Parameters
        ----------
        num_antennas : int
            Number of antennas in the array.
        ax : char, optional
            Axis along which the array is to be created.
            Takes value 'x', 'y' or 'z'. Default is 'x'.
        """
        if ax == "x":
            coordinates = (
                np.array(
                    [
                        np.arange(num_antennas),
                        np.zeros(num_antennas),
                        np.zeros(num_antennas),
                    ]
                ).T
                / 2
            )
        elif ax == "y":
            coordinates = (
                np.array(
                    [
                        np.zeros(num_antennas),
                        np.arange(num_antennas),
                        np.zeros(num_antennas),
                    ]
                ).T
                / 2
            )
        elif ax == "z":
            coordinates = (
                np.array(
                    [
                        np.zeros(num_antennas),
                        np.zeros(num_antennas),
                        np.arange(num_antennas),
                    ]
                ).T
                / 2
            )
        else:
            raise ValueError("ax must be 'x', 'y' or 'z'")
        coordinates = cls._translate_coordinates(coordinates)
        coordinates = cls._translate_coordinates(coordinates, array_center)
        ula = cls(num_antennas, coordinates)
        for kwarg in kwargs:
            ula.__setattr__(kwarg, kwargs[kwarg])
        return ula

    @classmethod
    def initialize_upa(cls, num_rows, num_cols, plane="xy", array_center=None):
        """Empties the array and creates a half-wavelength spaced,
        uniform plannar array in the desired plane.

        Parameters
        ----------
        num_rows : int
            Number of rows in the array. (along the first axis)
        num_col : int
            Number of columns in the array.(along the second axis)
        plane : str, optional
            Plane in which the array is to be created or the axis orthogonal to the plane.
            Takes value 'xy', 'yz' or 'xz'. Default is 'xy'.
        """

        if plane == "xy":
            coordinates = np.array(
                [
                    np.tile(np.arange(num_cols), num_rows),
                    np.repeat(np.arange(num_rows), num_cols),
                    np.zeros(num_rows * num_cols),
                ]
            ).T
        elif plane == "yz":
            coordinates = np.array(
                [
                    np.zeros(num_rows * num_cols),
                    np.tile(np.arange(num_cols), num_rows),
                    np.repeat(np.arange(num_rows), num_cols),
                ]
            ).T
        elif plane == "xz":
            coordinates = np.array(
                [
                    np.tile(np.arange(num_cols), num_rows),
                    np.zeros(num_rows * num_cols),
                    np.repeat(np.arange(num_rows), num_cols),
                ]
            ).T
        else:
            raise ValueError("plane must be 'xy', 'yz' or 'xz'")
        coordinates = cls._translate_coordinates(coordinates)
        return cls(num_rows * num_cols, coordinates).translate()

    @classmethod
    def from_file(cls, filename):
        """Load an array from a file.

        Parameters
        ----------
        filename : str
            Name of the file to load the array from.
        """
        raise NotImplementedError

    def to_file(self, filename):
        """Save the array to a file.

        Parameters
        ----------
        filename : str
            Name of the file to save the array to.
        """
        np.save(filename, [self.coordinates, self.weights, self.marker])

    def reset(self):
        """reset weights to 1"""
        self.weights = np.ones(self.num_antennas)

    def set_marker(self, marker):
        """Set the marker of the antennas.

        Parameters
        ----------
        marker : str
            Marker of the antennas.
        """
        # TODO: transmitter and receiver should be different markers
        self.marker = marker

    def set_weights(self, weights, index=None):
        """Set the weights of the antennas.

        Parameters
        ----------
        weights : array_like or float
            Weights of the antennas.
            If an array is given, the shape of the array must match the length of coordinates given.
            If a float is given, all antennas are changed to the same weight. and coordinates are ignored.
        index : array_like, optional
            Indices of the antennas whose weight is to be changed. If not given, the
            weights of all antennas are passed.
        """
        weights = np.asarray(weights).reshape(-1)
        if index is None:
            if isinstance(weights, float):
                self.weights = weights * np.ones(self.num_antennas)
            else:
                self.weights = weights
        else:
            self.weights[index] = weights

    def get_weights(self, coordinates=None):
        """Get the weights of the antennas.

        Parameters
        ----------
        coordinates : array_like
            Coordinates of the antennas whose weight is to be changed. If not
            given, the coordinates of all antennas are passed.
        """
        if coordinates is None:
            return self.weights
        else:
            indices = self._match_coordinates(coordinates)
            print(indices)
            if len(indices) == 0:
                raise ValueError("No matching coordinates found")
            return self.weights[indices]

    def _match_coordinates(self, coordinates):
        """Match the given coordinates to the coordinates of the array.

        Parameters
        ----------
        coordinates : array_like
            Coordinates of the antennas to be matched. The shape of the array must be (num_antennas, 3).
        """

        # match each coordinate to with the coordinate in the array and return the indices
        indices = []
        coordinates = np.reshape(coordinates, (-1, 3))
        indices = np.where((coordinates[:, None] == self.coordinates).all(axis=2))[1]
        return indices

    ############################
    #  Antenna Manipulation
    ############################

    def add_elements(self, coordinates):
        """Add antennas to the array.

        Parameters
        ----------
        coordinates : array_like
            Coordinates of the antennas to be added. The shape of the array must be (num_antennas, 3).
        """
        self.coordinates = np.concatenate((self.coordinates, coordinates))
        self.num_antennas += coordinates.shape[0]
        self.weights = np.concatenate((self.weights, np.ones(coordinates.shape[0])))

    def remove_elements(self, coordinates):
        """Remove antennas from the array.

        Parameters
        ----------
        coordinates : array_like
            Coordinates of the antennas to be removed. The shape of the array must be (num_antennas, 3).
        """
        indices = self._match_coordinates(coordinates)
        self.coordinates = np.delete(self.coordinates, indices, axis=0)
        self.weights = np.delete(self.weights, indices, axis=0)
        self.num_antennas -= len(indices)

    def remove_elements_by_idx(self, indices):
        """Remove antennas from the array.

        Parameters
        ----------
        indices : array_like
            Indices of the antennas to be removed. The shape of the array must be (num_antennas, 3).
        """
        self.coordinates = np.delete(self.coordinates, indices, axis=0)
        self.weights = np.delete(self.weights, indices, axis=0)
        self.num_antennas -= len(indices)

    @staticmethod
    def _translate_coordinates(coordinates, shift=None):
        """Shift all elements of the array by the given coordinates.

        Parameters
        ----------
        coordinates: array_like
            Coordinates of the array to be shifted.
        shift: array_like, optional
            Coordinates by which the array is to be shifted. If not given, the
            array is centered at the origin.
        """
        if shift is None:
            shift = -np.mean(coordinates, axis=0)
        shift = np.asarray(shift).reshape(1, -1)
        coordinates += shift
        return coordinates

    def translate(self, coordinates=None):
        """Shift all elements of the array by the given coordinates.

        Parameters
        ----------
        coordinates: array_like, optional
            Coordinates by which the array is to be shifted. If not given, the
            array is centered at the origin.
        """
        if coordinates is None:
            coordinates = -np.mean(self.coordinates, axis=0)
        self.coordinates += coordinates
        return coordinates

    def _rotate(self, coordinates, x_angle, y_angle, z_angle):
        """Rotate the array by the given angles.

        Parameters
        ----------
        x_angle : float
            Angle of rotation about the x-axis in radians.
        y_angle : float
            Angle of rotation about the y-axis in radians.
        z_angle : float
            Angle of rotation about the z-axis in radians.
        """
        rotation_matrix = np.array(
            [
                [
                    np.cos(y_angle) * np.cos(z_angle),
                    np.cos(z_angle) * np.sin(x_angle) * np.sin(y_angle)
                    - np.cos(x_angle) * np.sin(z_angle),
                    np.cos(x_angle) * np.cos(z_angle) * np.sin(y_angle)
                    + np.sin(x_angle) * np.sin(z_angle),
                ],
                [
                    np.cos(y_angle) * np.sin(z_angle),
                    np.cos(x_angle) * np.cos(z_angle)
                    + np.sin(x_angle) * np.sin(y_angle) * np.sin(z_angle),
                    -np.cos(z_angle) * np.sin(x_angle)
                    + np.cos(x_angle) * np.sin(y_angle) * np.sin(z_angle),
                ],
                [
                    -np.sin(y_angle),
                    np.cos(y_angle) * np.sin(x_angle),
                    np.cos(x_angle) * np.cos(y_angle),
                ],
            ]
        )

        translate_coordinates = self.translate()  # center the array at the origin
        self.coordinates = np.dot(coordinates, rotation_matrix)  # rotate the array
        self.translate(
            -translate_coordinates
        )  # translate the array back to its original position

        return np.dot(coordinates, rotation_matrix)

    def rotate(self, x_angle=0.0, y_angle=0.0, z_angle=0.0, inplace=True):
        """Rotate the array by the given angles.

        Parameters
        ----------
        x_angle : float
            Angle of rotation about the x-axis in radians.
        y_angle : float
            Angle of rotation about the y-axis in radians.
        z_angle : float
            Angle of rotation about the z-axis in radians.
        inplace : bool, optional
            If True, the array is rotated in-place. If False, a new array is
            returned. Default is True.
        """

        if inplace:
            self._rotate(self.coordinates, x_angle, y_angle, z_angle)
            return self
        else:
            new_array = self.copy()
            new_array._rotate(new_array.coordinates, x_angle, y_angle, z_angle)
            return new_array

    def get_array_center(self):
        """Returns the center of the array."""
        return np.mean(self.coordinates, axis=0)

    ############################
    # Get Array Properties
    ############################

    def get_array_response(self, az=0, el=0):
        """Returns the array response vector at a given azimuth and elevation.

        This response is simply the phase shifts experienced by the elements
        on an incoming wavefront from the given direction, normalied to the first
        element in the array

        Parameters
        ----------
        az : float
            Azimuth angle in radians.
        el : float
            Elevation angle in radians.
        """

        # calculate the distance of each element from the first element
        dx = self.coordinates[:, 0] - self.coordinates[0, 0]
        dy = self.coordinates[:, 1] - self.coordinates[0, 1]
        dz = self.coordinates[:, 2] - self.coordinates[0, 2]
        dx = dx.reshape(-1, 1)
        dy = dy.reshape(-1, 1)
        dz = dz.reshape(-1, 1)

        array_response_vector = np.exp(
            1j
            * 2
            * np.pi
            * (
                dx * np.sin(az) * np.cos(el)
                + dy * np.cos(az) * np.cos(el)
                + dz * np.sin(el)
            )
        )
        return array_response_vector

    def get_array_gain(self, az, el, db=True):
        """Returns the array gain at a given azimuth and elevation in dB.

        Parameters
        ----------
        az : float
            Azimuth angle in radians.
        el : float
            Elevation angle in radians.
        db : bool, optional

            If True, the gain is returned in dB. Default is True.
        """

        array_response_vector = self.get_array_response(az, el)
        gain = (self.weights.conj().T @ array_response_vector) ** 2
        if db:
            return 10 * np.log10(np.abs(gain + 1e-9))
        return gain

    def get_conjugate_beamformer(self, az, el):
        """Returns the conjugate beamformer at a given azimuth and elevation.

        Parameters
        ----------
        az : float
            Azimuth angle in radians.
        el : float
            Elevation angle in radians.
        """
        array_response_vector = self.get_array_response(az, el)
        conjugate_beamformer = np.conj(array_response_vector)
        return conjugate_beamformer

    def get_array_pattern_azimuth(self, el, num_points=360, range=360):
        """Returns the array pattern at a given elevation.

        Parameters
        ----------
        el : float
            Elevation angle in radians.
        num_points : int, optional
            Number of points at which the pattern is to be calculated.
            Default is 360.
        range : float, optional
            Range of azimuth angles in degrees. Default is 360.
        """
        az = np.linspace(-range / 2, range / 2, num_points) * np.pi / 180
        return self.get_array_response(az, el)

    ############################
    # Plotting
    ############################

    def plot_gain(
        self,
        el=0,
        az_range=178,
        az_center=0,
        num_points=178,
        weights=None,
        polar=False,
        db=True,
        ax=None,
        **kwargs,
    ):
        """Plot the array pattern at a given elevation.

        Parameters
        ----------
        el : float, optional
            Elevation angle in degrees. Default is 0.
        az_range : float, optional
            Range of azimuth angles in degrees. Default is 360.
        az_center : float, optional
            Center of azimuth angles in degrees. Default is 0.
        num_points : int, optional
            Number of points at which the pattern is to be calculated.
            Default is 360.
        **kwargs : optional
            matplotlib.pyplot.fig arguments.
        """
        if weights is not None:
            orig_weights = self.get_weights()
            self.set_weights(weights)
        az = (
            np.linspace(az_center - az_range / 2, az_center + az_range / 2, num_points)
            * np.pi
            / 180
        )
        array_response = np.zeros(num_points)
        for i in range(num_points):
            array_response[i] = self.get_array_gain(az[i], el * np.pi / 180, db=db)
        # if "label" not in kwargs:
        # kwargs["label"] = r"Elevation = {} deg".format(el)

        if ax == None:
            fig, ax = plt.subplots()
        if polar:
            ax = plt.gca(projection="polar")
            ax.plot(az, array_response, **kwargs)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_rlabel_position(0)
            ax.set_rticks([-20, -10, 0])
            # ax.set_rlim(-20, 0)
        else:
            ax.plot(az * 180 / np.pi, array_response)
            # ax.set_ylim(-(max(array_response)), max(array_response) + 10)
        ax.set_xlabel("Azimuth (deg)")
        ax.set_ylabel("Gain (dB)")
        title = f"Array Pattern at Elevation = {el} deg, Max Gain = {np.max(np.real(array_response)):.2f} dB"
        ax.set_title(title)
        ax.grid(True)
        if weights is not None:
            self.set_weights(orig_weights)
        if ax is None:
            plt.show()
        return ax

    def plot_gain_3d(
        self,
        az_range=360,
        el_range=180,
        az_center=0,
        el_center=0,
        num_points=360,
        polar=False,
        ax=None,
        **kwargs,
    ):
        """Plot the array pattern.

        Parameters
        ----------
        az_range : float, optional
            Range of azimuth angles in degrees. Default is 360.
        el_range : float, optional
            Range of elevation angles in degrees. Default is 180.
        az_center : float, optional
            Center of azimuth angles in degrees. Default is 0.
        el_center : float, optional
            Center of elevation angles in degrees. Default is 0.
        num_points : int, optional
            Number of points at which the pattern is to be calculated.
            Default is 360.
        **kwargs : optional
            matplotlib.pyplot.plot arguments.
        """
        az = (
            np.linspace(az_center - az_range / 2, az_center + az_range / 2, num_points)
            * np.pi
            / 180
        )
        el = (
            np.linspace(el_center - el_range / 2, el_center + el_range / 2, num_points)
            * np.pi
            / 180
        )
        az_grid, el_grid = np.meshgrid(az, el)
        array_response = np.zeros((num_points, num_points), dtype=complex)
        for i in range(num_points):
            for j in range(num_points):
                array_response[i, j] = self.get_array_gain(az_grid[i, j], el_grid[i, j])
        array_response = np.abs(array_response)
        array_response = 20 * np.log10(array_response / np.max(array_response))
        fig = plt.figure(**kwargs)

        if polar:
            # TODO: fix polar plot
            if ax is None:
                ax = plt.gca(projection="polar")
            ax.plot(
                az_grid,
                el_grid,
                array_response,
                # cmap=cm.coolwarm,
                linewidth=0,
            )
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_rlabel_position(0)
            ax.set_rticks([-20, -10, 0])
            ax.set_rlim(-20, 0)
        else:
            if ax is None:
                ax = fig.gca()
            ax.plot_surface(
                az_grid * 180 / np.pi,
                el_grid * 180 / np.pi,
                array_response,
                cmap=cm.coolwarm,
                linewidth=0,
            )
            ax.set_xlabel("Azimuth (deg)")
            ax.set_ylabel("Elevation (deg)")
            ax.set_zlabel("Gain (dB)")
        if ax is None:
            plt.show()
        return ax

    def plot_array_3d(self):
        """Plot the array."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            self.coordinates[:, 0],
            self.coordinates[:, 1],
            self.coordinates[:, 2],
            marker=self.marker,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    def plot_array(self, plane="xy", ax=None):
        """Plot the array in 2D projection

        Parameters
        ----------
        plane : str, optional
            Plane in which the array is to be projected.
            Takes value 'xy', 'yz' or 'xz'. Default is 'xy'.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The matplotlib axes object.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if plane == "xy":
            ax.scatter(
                self.coordinates[:, 0], self.coordinates[:, 1], marker=self.marker
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        elif plane == "yz":
            ax.scatter(
                self.coordinates[:, 1], self.coordinates[:, 2], marker=self.marker
            )
            ax.set_xlabel("y")
            ax.set_ylabel("z")
        elif plane == "xz":
            ax.scatter(
                self.coordinates[:, 0], self.coordinates[:, 2], marker=self.marker
            )
            ax.set_xlabel("x")
            ax.set_ylabel("z")
        else:
            raise ValueError("plane must be 'xy', 'yz' or 'xz'")
        ax.grid(True)
        ax.set_title(r"Array Projection in {}-plane".format(plane))

        if ax is None:
            plt.show()
        return ax
