##############################################################################
# MDTraj: A Python Library for Loading, Saving, and Manipulating
#         Molecular Dynamics Trajectories.
# Copyright 2012-2015 Stanford University and the Authors
#
# Authors: Robert McGibbon
# Contributors: Kyle A Beauchamp, Jason Swails
#
# MDTraj is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with MDTraj. If not, see <http://www.gnu.org/licenses/>.
##############################################################################



from __future__ import print_function, division
import numpy as np
from mdtraj.utils import ensure_type
from mdtraj.utils.six.moves import range


__all__ = ['compute_center_of_mass']



def compute_center_of_mass(traj):
    """Compute the center of mass for each frame.

    Parameters
    ----------
    traj : Trajectory
        Trajectory to compute center of mass for

    Returns
    -------
    com : np.ndarray, shape=(n_frames, 3)
         Coordinates of the center of mass for each frame
    """

    com = np.zeros((traj.n_frames, 3))
    masses = np.array([a.element.mass for a in traj.top.atoms])
    masses /= masses.sum()

    for i, x in enumerate(traj.xyz):
        com[i, :] = x.astype('float64').T.dot(masses)
    return com


##############################################################################
# pure python implementation of the core routines
##############################################################################


def _distance(xyz, pairs):
    "Distance between pairs of points in each frame"
    delta = np.diff(xyz[:, pairs], axis=2)[:, :, 0]
    return (delta ** 2.).sum(-1) ** 0.5


def _displacement(xyz, pairs):
    "Displacement vector between pairs of points in each frame"
    value = np.diff(xyz[:, pairs], axis=2)[:, :, 0]
    assert value.shape == (xyz.shape[0], pairs.shape[0], 3), 'v.shape %s, xyz.shape %s, pairs.shape %s' % (str(value.shape), str(xyz.shape), str(pairs.shape))
    return value


def _reduce_box_vectors(vectors):
    """Make sure box vectors are in reduced form."""
    (bv1, bv2, bv3) = vectors
    bv3 -= bv2*round(bv3[1]/bv2[1]);
    bv3 -= bv1*round(bv3[0]/bv1[0]);
    bv2 -= bv1*round(bv2[0]/bv1[0]);
    return (bv1, bv2, bv3)


def _distance_mic(xyz, pairs, box_vectors, orthogonal):
    """Distance between pairs of points in each frame under the minimum image
    convention for periodic boundary conditions.

    The computation follows scheme B.9 in Tukerman, M. "Statistical
    Mechanics: Theory and Molecular Simulation", 2010.

    This is a slow pure python implementation, mostly for testing.
    """
    out = np.empty((xyz.shape[0], pairs.shape[0]), dtype=np.float32)
    for i in range(len(xyz)):
        bv1, bv2, bv3 = _reduce_box_vectors(box_vectors[i].T)

        for j, (a,b) in enumerate(pairs):
            r12 = xyz[i,b,:] - xyz[i,a,:]
            r12 -= bv3*round(r12[2]/bv3[2]);
            r12 -= bv2*round(r12[1]/bv2[1]);
            r12 -= bv1*round(r12[0]/bv1[0]);
            dist = np.linalg.norm(r12)
            if not orthogonal:
                for ii in range(-1, 2):
                    v1 = bv1*ii
                    for jj in range(-1, 2):
                        v12 = bv2*jj + v1
                        for kk in range(-1, 2):
                            new_r12 = r12 + v12 + bv3*kk
                            dist = min(dist, np.linalg.norm(new_r12))
            out[i, j] = dist
    return out


def _displacement_mic(xyz, pairs, box_vectors, orthogonal):
    """Displacement vector between pairs of points in each frame under the
    minimum image convention for periodic boundary conditions.

    The computation follows scheme B.9 in Tukerman, M. "Statistical
    Mechanics: Theory and Molecular Simulation", 2010.

    This is a very slow pure python implementation, mostly for testing.
    """
    out = np.empty((xyz.shape[0], pairs.shape[0], 3), dtype=np.float32)
    for i in range(len(xyz)):
        bv1, bv2, bv3 = _reduce_box_vectors(box_vectors[i].T)
        hinv = np.linalg.inv(np.array([bv1, bv2, bv3]).T)

        for j, (a,b) in enumerate(pairs):
            r12 = xyz[i,b,:] - xyz[i,a,:]
            r12 -= bv3*round(r12[2]/bv3[2]);
            r12 -= bv2*round(r12[1]/bv2[1]);
            r12 -= bv1*round(r12[0]/bv1[0]);
            min_disp = r12
            dist2 = (r12*r12).sum()
            if not orthogonal:
                for ii in range(-1, 2):
                    v1 = bv1*ii
                    for jj in range(-1, 2):
                        v12 = bv2*jj+v1
                        for kk in range(-1, 2):
                            tmp = r12 + v12 + bv3*kk
                            new_dist2 = (tmp*tmp).sum()
                            if new_dist2 < dist2:
                                dist2 = new_dist2
                                min_disp = tmp
            out[i, j] = min_disp

    return out
