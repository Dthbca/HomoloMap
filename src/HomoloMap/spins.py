"""Deprecated compatibility module; use HomoloMap.stats and transforms."""
from pathlib import Path
import warnings
warnings.warn(
    "HomoloMap.spins is deprecated; import from HomoloMap.stats or "
    "HomoloMap.transforms instead.", DeprecationWarning, stacklevel=2
)
import tempfile
import os
import numpy as np
import pandas as pd
import nibabel as nib
import subprocess
import uuid
from scipy import optimize
from scipy.spatial import distance_matrix, cKDTree
from sklearn.utils.validation import check_random_state
from neuromaps import stats
from HomoloMap.datasets import fetch_parc, fetch_fslr

from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from itertools import combinations
from tqdm import tqdm


PARCIGNORE = [
    'unknown', 'corpuscallosum', 'Background+FreeSurfer_Defined_Medial_Wall',
    '???', 'Unknown', 'Medial_wall', 'Medial wall', 'medial_wall', None
]


def get_gd_parc_centroids(mesh, parc, label):
    """
    Calculate the geodesic centroid of a specific parcel label on a cortical mesh.

    Parameters:
    ----------
    mesh : str
        Path to the cortical mesh file.
    parc : nibabel.GiftiImage
        Parcel data loaded as a GIFTI image.
    label : int
        The specific parcel label for which the centroid is calculated.

    Returns:
    -------
    inds : int
        Index of the vertex closest to the geodesic centroid.
    vertices[inds] : np.ndarray
        Coordinates of the geodesic centroid vertex.
    """
    # Extract parcel data and find vertices corresponding to the given label
    parc_data = parc.agg_data()
    label_vertices = np.where(parc_data == label)[0]
    metric_data = np.zeros(shape=parc_data.shape)
    metric_data[label_vertices] = 1

    # Create a GIFTI image for the ROI metric
    data_array = nib.gifti.GiftiDataArray(
        data=metric_data,
        datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
        intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE']
    )
    metric_img = nib.gifti.GiftiImage()
    metric_img.add_gifti_data_array(data_array)

    # Save the ROI metric to a temporary file
    output_file = f'/tmp/roi_metric_{uuid.uuid4().hex[:8]}.func.gii'
    nib.save(metric_img, output_file)

    # Compute geodesic distances using wb_command
    roi_metric = output_file
    tmp_fname = f'/tmp/surface_disc_{uuid.uuid4().hex[:8]}.dconn.nii'
    cmd = ['wb_command', '-surface-geodesic-distance-all-to-all',
           mesh, tmp_fname, '-roi', roi_metric]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    # Load the geodesic distance matrix and find the closest vertex
    disc = nib.load(tmp_fname)
    disc = np.asarray(disc.dataobj)
    inds = label_vertices[disc.sum(axis=0).argmin()]
    vertices, _ = nib.load(mesh).agg_data()

    # Clean up temporary files
    for f in [output_file, tmp_fname]:
        if os.path.exists(f):
            os.remove(f)

    return inds, vertices[inds]


def get_parcel_centroids(surf, parc=None, method='surface', drop=PARCIGNORE):
    """
    Calculate centroids for parcels on a cortical surface.

    Parameters:
    ----------
    surf : str
        Path to the cortical surface file.
    parc : nibabel.GiftiImage, optional
        Parcel data loaded as a GIFTI image. Default is None.
    method : str, optional
        Method for centroid calculation ('average', 'surface', 'geodesic').
        Default is 'surface'.
    drop : list, optional
        List of parcel labels to ignore. Default is PARCIGNORE.

    Returns:
    -------
    centroids : np.ndarray
        Array of centroid coordinates for each parcel.
    """
    # Validate the method parameter
    methods = ['average', 'surface', 'geodesic']
    if method not in methods:
        raise ValueError('Provided method for centroid calculation {} is '
                         'invalid. Must be one of {}'.format(methods, methods))
    if drop is None:
        drop = PARCIGNORE

    centroids = []

    # Load surface vertices and faces
    vertices, faces = nib.load(surf).agg_data()
    if parc is not None:
        labels = parc.agg_data()
        labeltable = parc.labeltable.get_labels_as_dict()

        if not labeltable:
            warnings.warn("Labeltable is empty. Using numeric labels directly.")
            labeltable = {lab: str(lab) for lab in np.unique(labels)}
            for lab in np.unique(labels):
                if lab <= 0:
                    labeltable[lab] = "unknown"

        # Iterate over unique parcel labels
        for lab in np.unique(labels):
            if labeltable.get(lab) in drop:
                continue

            mask = labels == lab
            if method in ('average', 'surface'):
                # Calculate the average position of vertices in the parcel
                roi = np.atleast_2d(vertices[mask].mean(axis=0))
                if method == 'surface':  # Find the closest vertex on the surface
                    idx = np.argmin(distance_matrix(vertices, roi), axis=0)[0]
                    roi = vertices[idx]
            elif method == 'geodesic':
                # Calculate the geodesic centroid
                _, roi = get_gd_parc_centroids(surf, parc, lab)

            centroids.append(roi)
    else:
        centroids.append(vertices)

    return np.row_stack(centroids)


def _gen_rotation(seed=None):
    """
    Generate a random rotation matrix for 3D coordinates.

    Parameters:
    ----------
    seed : int or None, optional
        Seed for the random number generator to ensure reproducibility. Default is None.

    Returns:
    -------
    rotate_l : np.ndarray
        A 3x3 rotation matrix that can be used to rotate 3D coordinates.
    """

    rs = check_random_state(seed)
    # generate rotation for left
    rotate_l, temp = np.linalg.qr(rs.normal(size=(3, 3)))
    rotate_l = rotate_l @ np.diag(np.sign(np.diag(temp)))
    if np.linalg.det(rotate_l) < 0:
        rotate_l[:, 0] = -rotate_l[:, 0]

    return rotate_l


def gen_spinsamples(coords,  n_rotate=1000, check_duplicates=True,
                    method='Alexander-Bloch', seed=None, verbose=False,
                    return_cost=False):
    """
    Parameters
    ----------
    coords : (N, 3) array_like
        X, Y, Z coordinates of `N` nodes/parcels/regions/vertices defined on a
        sphere
    n_rotate : int, optional
        Number of rotations to generate. Default: 1000
    check_duplicates : bool, optional
        Whether to check for and attempt to avoid duplicate resamplings. A
        warnings will be raised if duplicates cannot be avoided. Setting to
        True may increase the runtime of this function! Default: True
    method : {'Alexander-Bloch', 'vasa', 'hungarian'}, optional
        Method by which to match non- and rotated coordinates. Specifying
        'Alexander-Bloch' will use the method described in Alexander-Bloch et al.(2018).
        'On testing for spatial correspondence between maps of human brain
        structure and function'. NeuroImage, 178, 540-51. Specifying 'vasa'
        will use the method described in Váša et al.(2018). 'Adolescent
        tuning of association cortex in human structural brain networks'.
        Cerebral Cortex, 28(1), 281-294.. Specifying 'hungarian' will
        use the Hungarian algorithm to minimize the global cost of
        reassignment (will dramatically increase runtime).
        Default: 'Alexander-Bloch'
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Default: None
    verbose : bool, optional
        Whether to print occasional status messages. Default: False
    return_cost : bool, optional
        Whether to return cost array (specified as Euclidean distance) for each
        coordinate for each rotation Default: True

    Returns
    -------
    spinsamples : (N, `n_rotate`) numpy.ndarray
        Resampling matrix to use in permuting data based on supplied `coords`.
    cost : (N, `n_rotate`,) numpy.ndarray
        Cost (specified as Euclidean distance) of re-assigning each coordinate
        for every rotation in `spinsamples`. Only provided if `return_cost` is
        True.
    """

    methods = ['Alexander-Bloch', 'vasa', 'hungarian']
    if method not in methods:
        raise ValueError('Provided method "{}" invalid. Must be one of {}.'
                         .format(method, methods))

    seed = check_random_state(seed)
    coords = np.asanyarray(coords)

    # check supplied coordinate shape
    if coords.shape[-1] != 3 :
        raise ValueError('Provided `coords` must be of shape (N, 3), not {}'
                         .format(coords.shape))

    # empty array to store resampling indices
    spinsamples = np.zeros((len(coords), n_rotate), dtype=int)
    cost = np.zeros((len(coords), n_rotate))
    inds = np.arange(len(coords), dtype=int)

    # generate rotations and resampling array!
    msg, warned = '', False
    for n in range(n_rotate):
        count, duplicated = 0, True

        if verbose:
            msg = 'Generating spin {:>5} of {:>5}'.format(n, n_rotate)
            print(msg, end='\r', flush=True)

        while duplicated and count < 500:
            count, duplicated = count + 1, False
            resampled = np.zeros(len(coords), dtype='int32')

            # rotate each hemisphere separately
            rot = _gen_rotation(seed=seed)
            coor = coords
            if len(coor) == 0:
                continue

            if method == 'vasa':
                dist = distance_matrix(coor, coor @ rot)
                # min of max a la Vasa et al., 2018
                col = np.zeros(len(coor), dtype='int32')
                for _ in range(len(dist)):
                    # find parcel whose closest neighbor is farthest away
                    # overall; assign to that
                    row = dist.min(axis=1).argmax()
                    col[row] = dist[row].argmin()
                    cost[inds[row], n] = dist[row, col[row]]
                    # set to -inf and inf so they can't be assigned again
                    dist[row] = -np.inf
                    dist[:, col[row]] = np.inf

            elif method == 'hungarian':
                dist = distance_matrix(coor, coor @ rot)
                row, col = optimize.linear_sum_assignment(dist)
                cost[:, n] = dist[row, col]

            elif method == 'Alexander-Bloch':
                dist, col = cKDTree(coor @ rot).query(coor, 1)
                cost[:, n] = dist

            resampled = inds[col]

            # if we want to check for duplicates ensure that we don't have any
            if check_duplicates:
                if np.any(np.all(resampled[:, None] == spinsamples[:, :n], 0)):
                    duplicated = True
                # if our "spin" is identical to the input then that's no good
                elif np.all(resampled == inds):
                    duplicated = True

        if count == 500 and not warned:
            warnings.warn('Duplicate rotations used. Check resampling array '
                          'to determine real number of unique permutations.',
                          stacklevel=2)
            warned = True

        spinsamples[:, n] = resampled

    if verbose:
        print(' ' * len(msg) + '\b' * len(msg), end='', flush=True)

    if return_cost:
        return spinsamples, cost

    return spinsamples


def spin_data(data=None,atlas='FGC',n_spins=1000,seed=1234,method='Alexander-Bloch',return_ind = False):
    """
    Generate spin permutations for spatial data.

    Parameters
    ----------
    data : pd.DataFrame or None, optional
        DataFrame containing spatial data to be permuted. Rows should correspond
        to brain regions, and columns to features. If None, only the spin indices
        will be returned. Default: None.
    atlas : str, optional
        Brain atlas to use for fetching parcellation data. Default: 'FGC'.
    n_spins : int, optional
        Number of spin permutations to generate. Default: 1000.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility. Default: 1234.
    method : {'Alexander-Bloch', 'vasa', 'hungarian'}, optional
        Method for matching non-rotated and rotated coordinates:
        - 'Alexander-Bloch': Uses the method described in [ST1]_.
        - 'vasa': Uses the method described in [ST4]_.
        - 'hungarian': Uses the Hungarian algorithm to minimize the global cost
          of reassignment (increases runtime significantly).
        Default: 'Alexander-Bloch'.
    return_ind : bool, optional
        If True, return only the spin indices. If False, return the permuted data.
        Default: False.

    Returns
    -------
    spins : np.ndarray
        If `return_ind` is True, returns an array of spin indices with shape
        (n_regions, n_spins). Otherwise, returns the permuted data.
    """

    parc = fetch_parc(key=atlas)
    surf =  fetch_fslr(surf='sphere',return_path=True)
    coords = get_parcel_centroids(surf, parc=parc, method='surface')

    labels = np.unique(parc.agg_data())
    labels = labels[labels > 0]

    spins = gen_spinsamples(coords, n_rotate=n_spins, method= method, seed=seed)

    if (data is None) | return_ind:
        return spins
    else:
        if isinstance(data, pd.DataFrame):
            data = data.reindex(labels).values
        return np.squeeze(data[spins])


def get_gd_disc(mesh, vertex_number, radius=None,return_dist=False):
    """
    Get a disc of vertices that are `radius` mm apart from the seed
    `vertex_number` on the `mesh`.

    Parameters
    ----------
    mesh : str
        Path to the cortical mesh file.
    vertex_number : int
        Index of the seed vertex on the mesh.
    radius : int, optional
        Radius of the geodesic disc in millimeters. If None, the entire geodesic
        distance map is computed. Default: None.
    return_dist : bool, optional
        If True, return both the geodesic distances and the geodesic mask.
        If False, return only the geodesic mask. Default: False.

    Returns
    -------
    gd_mask : np.ndarray (bool)
        Boolean mask indicating vertices within the geodesic disc.
    gd : np.ndarray, optional
        Geodesic distances from the seed vertex to all other vertices on the mesh.
        Only returned if `return_dist` is True.
    """
    tmp_fname = f'/tmp/gd_{vertex_number}_{uuid.uuid4().hex[:8]}.shape.gii'
    cmd = ['wb_command', '-surface-geodesic-distance',
        mesh, str(vertex_number), tmp_fname]
    if radius is not None:
        cmd.extend(['-limit', str(radius)])
    subprocess.run(cmd,check=True)
    # read the output gifti file into a numpy array
    gd = nib.load(tmp_fname).agg_data()
    gd_mask = gd >= 0

    if os.path.exists(tmp_fname):
        os.remove(tmp_fname)

    if return_dist:
        return gd, gd_mask
    else:
        return gd_mask


def disc_smooth(surface_data, smooth_disc_radius, approach='euclidean'):
    """
    Smooth surface data by averaging values within a geodesic or Euclidean disc.

    Parameters
    ----------
    surface_data : np.ndarray
        A 1D or 2D array of surface data with shape (32492, ...) where 32492 corresponds
        to the number of vertices in the cortical surface.
    smooth_disc_radius : float
        Radius of the disc (in mm) used for smoothing.
    approach : str, optional
        The approach used to define the disc:
        - 'euclidean': Use Euclidean distance to define the disc.
        - 'geodesic': Use geodesic distance on the cortical surface to define the disc.
        Default: 'euclidean'.

    Returns
    -------
    smoothed_surface_data : np.ndarray
        The smoothed surface data with the same shape as `surface_data`.
    discs : dict or np.ndarray
        If `approach` is 'euclidean', returns a precomputed distance matrix (discs).
        If `approach` is 'geodesic', returns an empty dictionary.
    """
    
    assert surface_data.shape[0] == 32492
    discs = {}
    smoothed_surface_data = {}
    # load inflated mesh
    inflated_mesh = fetch_fslr(surf='inflated').agg_data()
    if approach == 'euclidean':
        # calculate Euclidean distance of all pairs of vertices
        ed_matrix = distance_matrix(inflated_mesh[0], inflated_mesh[0])
        # create a matrix of discs for each row/column seed vertex
        discs = ed_matrix < smooth_disc_radius
    # loop through vertices and calculate the smoothed value by
    # taking the average of the disc surrounding it
    smoothed_surface_data = np.zeros_like(surface_data)
    for vertex in range(32492):
        if np.isnan(surface_data[vertex]).all():
            # only do smoothing if the seed is not NaN
            smoothed_surface_data[vertex] = np.NaN
        else:
            if approach == 'euclidean':
                disc = discs[vertex, :].astype('bool')
            else:
                disc = get_gd_disc(fetch_fslr(surf='inflated',return_path=True), vertex, smooth_disc_radius)
            smoothed_surface_data[vertex] = np.nanmean(surface_data[disc], axis=0)
    return smoothed_surface_data, discs


def smooth_single_roi(roi, parc_data, roi_disc, radius, method='mean', sigma=5):
    """
    Smooth data for a single region of interest (ROI) using mean or Gaussian smoothing.

    Parameters
    ----------
    roi : str or int
        The region of interest (ROI) to smooth.
    parc_data : pd.DataFrame
        DataFrame containing parcellation data. Rows correspond to ROIs, and columns correspond to features.
    roi_disc : pd.DataFrame
        Geodesic distance matrix for ROIs. Rows and columns correspond to ROIs.
    radius : float
        Radius (in mm) within which to smooth the data.
    method : {'mean', 'gaussian'}, optional
        Smoothing method to use:
        - 'mean': Average the values within the specified radius.
        - 'gaussian': Apply a Gaussian kernel to weight the values based on distance.
        Default: 'mean'.
    sigma : float, optional
        Standard deviation for the Gaussian kernel. Only used if `method='gaussian'`.
        Default: 5.

    Returns
    -------
    smoothed : np.ndarray
        Smoothed data for the specified ROI.
    """
    
    disc = roi_disc.loc[roi, :].values
    labels = roi_disc.index.values
    
    
    mask = (disc < radius) & (disc > 0)
    neighbor_labels = labels[mask]
    neighbor_disc = disc[mask]
    
    current_data = parc_data.loc[roi, :].values
    
    if method == 'mean':
        if len(neighbor_labels) > 0:
            neighbor_data = parc_data.loc[neighbor_labels, :].values
            smoothed = (np.sum(neighbor_data, axis=0) + current_data) / (len(neighbor_labels) + 1)
        else:
            smoothed = current_data 
            
    elif method == 'gaussian':    
        if len(neighbor_labels) > 0:
            all_names = np.append(neighbor_labels, roi)
            all_distances = np.append(neighbor_disc, 0.0) 
            
            weights = np.exp(- (all_distances ** 2) / (2 * sigma ** 2))
            weights = weights / np.sum(weights)  
            
            neighbor_data = parc_data.loc[all_names, :].values
            smoothed = np.sum(neighbor_data * weights[:, np.newaxis], axis=0)
        else:
            smoothed = current_data  
    else:
        raise ValueError("method must be 'mean' or 'gaussian'")
    
    return smoothed

def get_parcel_geodist(mesh, parc, n_jobs=-1):
    """
    Calculate the geodesic distance matrix for parcel labels.

    Parameters:
    ----------
    mesh : str
        Path to the cortical mesh file.
    parc : nibabel.GiftiImage
        Parcel data loaded as a GIFTI image.
    n_jobs : int, optional
        Number of parallel jobs to use. Default is -1 (use all available cores).

    Returns:
    -------
    geodist_matrix : pd.DataFrame
        DataFrame containing the geodesic distance matrix for parcel labels.
    """
    # Extract unique parcel labels, excluding background (label 0)
    labels = np.trim_zeros(np.unique(parc.agg_data()))

    # Compute centroids for each parcel
    results = Parallel(n_jobs=n_jobs)(
        delayed(get_gd_parc_centroids)(mesh, parc, label) for label in labels
    )
    centroids = [result[0] for result in results]

    # Compute geodesic distances for each centroid
    results = Parallel(n_jobs=n_jobs)(
        delayed(get_gd_disc)(mesh, centroid, None, True) for centroid in centroids
    )

    # Build the geodesic distance matrix
    geodist_matrix = np.asarray([results[i][0][centroids] for i in range(len(results))])
    geodist_matrix = pd.DataFrame(geodist_matrix, index=labels, columns=labels)

    return geodist_matrix


def parc_smooth(parc_data, roi_disc=None, radius=6, method='gaussian', sigma=5,
                mesh = None, parc = None,n_jobs=-1):
    """
    Smooth parcellation data by averaging or applying a Gaussian kernel within a specified radius.

    Parameters
    ----------
    parc_data : pd.DataFrame
        DataFrame containing parcellation data. Rows correspond to regions of interest (ROIs),
        and columns correspond to features.
    roi_disc : pd.DataFrame or None, optional
        Precomputed geodesic distance matrix for ROIs. If None, it will be computed using
        `mesh` and `parc`. Default: None.
    radius : float, optional
        Radius (in mm) within which to smooth the data. Default: 6.
    method : {'mean', 'gaussian'}, optional
        Smoothing method to use:
        - 'mean': Average the values within the specified radius.
        - 'gaussian': Apply a Gaussian kernel to weight the values based on distance.
        Default: 'gaussian'.
    sigma : float, optional
        Standard deviation for the Gaussian kernel. Only used if `method='gaussian'`.
        Default: 5.
    mesh : str or None, optional
        Path to the cortical mesh file. Required if `roi_disc` is not provided. Default: None.
    parc : nibabel.GiftiImage or None, optional
        Parcellation data loaded as a GIFTI image. Required if `roi_disc` is not provided.
        Default: None.
    n_jobs : int, optional
        Number of parallel jobs to use for smoothing. Default: -1 (use all available cores).

    Returns
    -------
    smoothed_data : pd.DataFrame
        DataFrame containing the smoothed parcellation data with the same shape as `parc_data`.
    """

    if roi_disc is None:
        if (mesh is None) or (parc is None):
            raise ValueError('meah and parc must provided when no distance matrix prepared')
        else:
            roi_disc = get_parcel_geodist(mesh, parc, n_jobs=-1)
        
    roi_disc = roi_disc.loc[parc_data.index,parc_data.index]

    results = Parallel(n_jobs=n_jobs)(
        delayed(smooth_single_roi)(roi, parc_data, roi_disc, radius, method, sigma)
        for roi in parc_data.index
    )

    smoothed_data = np.zeros_like(parc_data.values)
    for i,data in enumerate(results):
        smoothed_data[i, :] = data

    return pd.DataFrame(smoothed_data, index=parc_data.index, columns=parc_data.columns)


def get_reg_r_sq(X, y, adjust=True, model_type='linear'):
    """
    Calculate R-squared or adjusted R-squared for a regression model.

    Parameters:
    ----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target variable.
    adjust : bool, optional
        Whether to calculate adjusted R-squared. Default is True.
    model_type : str, optional
        Type of regression model ('linear', 'random_forest', 'svr'). Default is 'linear'.

    Returns:
    -------
    float
        R-squared or adjusted R-squared value.
    """
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42)
    elif model_type == 'svr':
        model = SVR()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    model.fit(X, y)
    yhat = model.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * \
        (len(y) - 1) / (len(y) - X.shape[1] - 1)
    if adjust:
        return adjusted_r_squared
    else:
         return r_squared

def get_reg_r_pval(X, y, spins, nspins,model_type='linear'):
    """
    Calculate the empirical p-value for the R-squared of a regression model
    using spin permutations.

    Parameters
    ----------
    X : np.ndarray
        Input features (predictor variables) with shape (n_samples, n_features).
    y : np.ndarray
        Target variable with shape (n_samples,).
    spins : np.ndarray
        Spin permutation indices with shape (n_samples, nspins). Each column
        represents a permutation of the rows of `X`.
    nspins : int
        Number of spin permutations to use.
    model_type : str, optional
        Type of regression model to use. Options are:
        - 'linear': Linear regression (default).
        - 'random_forest': Random forest regression.
        - 'svr': Support vector regression.

    Returns
    -------
    pval : float
        Empirical p-value for the R-squared value of the regression model.
    """

    emp = get_reg_r_sq(X, y,model_type=model_type)
    null = np.zeros((nspins, ))
    for s in range(nspins):
        null[s] = get_reg_r_sq(X[spins[:, s], :], y,model_type=model_type)
    return (1 + sum(null > emp))/(nspins + 1)




def get_dominance_stats(X, y, use_adjusted_r_sq=True, verbose=False, n_jobs=-1):
    """
    Return the dominance analysis statistics for multilinear regression.
    Parameters
    ----------
    X : (N, M) array_like
        Input data
    y : (N,) array_like
        Target values
    use_adjusted_r_sq : bool, optional
        Whether to use adjusted r squares. Default: True
    verbose : bool, optional
        Whether to print debug messages. Default: False
    n_jobs : int, optional
        The number of jobs to run in parallel. Default: 1

    Returns
    -------
    model_metrics : dict
        The dominance metrics, currently containing `individual_dominance`,
        `partial_dominance`, `total_dominance`, and `full_r_sq`.
    model_r_sq : dict
        Contains all model r squares
    """
    # this helps to remove one element from a tuple
    def remove_ret(tpl, elem):
        lst = list(tpl)
        lst.remove(elem)
        return tuple(lst)

    # helper function to compute r_sq for a given idx_tuple
    def compute_r_sq(idx_tuple):
        return idx_tuple, get_reg_r_sq(X[:, idx_tuple],
                                       y,
                                       adjust=use_adjusted_r_sq)

    # generate all predictor combinations in list (num of predictors) of lists
    n_predictor = X.shape[-1]
    # n_comb_len_group = n_predictor - 1
 
    predictor_combs = [list(combinations(range(n_predictor), i))
                       for i in range(1, n_predictor + 1)]
    if verbose:
        print(f"[Dominance analysis] Generated \
              {len([v for i in predictor_combs for v in i])} combinations")

    model_r_sq = dict()
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_r_sq)(idx_tuple)
        for len_group in tqdm(predictor_combs,
                              desc='num-of-predictor loop',
                              disable=not verbose)
        for idx_tuple in tqdm(len_group,
                              desc='insider loop',
                              disable=not verbose))

    # extract r_sq from results
    for idx_tuple, r_sq in results:
        model_r_sq[idx_tuple] = r_sq

    if verbose:
        print(f"[Dominance analysis] Acquired {len(model_r_sq)} r^2's")

    # getting all model metrics
    model_metrics = dict([])

    # individual dominance
    individual_dominance = []
    for i_pred in range(n_predictor):
        individual_dominance.append(model_r_sq[(i_pred,)])
    individual_dominance = np.array(individual_dominance).reshape(1, -1)
    model_metrics["individual_dominance"] = individual_dominance

    # partial dominance
    partial_dominance = [[] for _ in range(n_predictor - 1)]
    for i_len in range(n_predictor - 1):
        i_len_combs = list(combinations(range(n_predictor), i_len + 2))
        for j_node in range(n_predictor):
            j_node_sel = [v for v in i_len_combs if j_node in v]
            reduced_list = [remove_ret(comb, j_node) for comb in j_node_sel]
            diff_values = [
                model_r_sq[j_node_sel[i]] - model_r_sq[reduced_list[i]]
                for i in range(len(reduced_list))]
            partial_dominance[i_len].append(np.mean(diff_values))

    # save partial dominance
    partial_dominance = np.array(partial_dominance)
    model_metrics["partial_dominance"] = partial_dominance
    # get total dominance
    total_dominance = np.mean(
        np.r_[individual_dominance, partial_dominance], axis=0)
    # test and save total dominance
    assert np.allclose(total_dominance.sum(),
                       model_r_sq[tuple(range(n_predictor))]), \
           "Sum of total dominance is not equal to full r square!"
    model_metrics["total_dominance"] = total_dominance
    # save full r^2
    model_metrics["full_r_sq"] = model_r_sq[tuple(range(n_predictor))]

    return model_metrics, model_r_sq


try:
    import shap
except ImportError:
    shap = None
def shap_stats(X,y,model_type = 'linear'):
    """
    Calculate SHAP (SHapley Additive exPlanations) values for a regression model.

    Parameters
    ----------
    X : np.ndarray
        Input features (predictor variables) with shape (n_samples, n_features).
    y : np.ndarray
        Target variable with shape (n_samples,).
    model_type : str, optional
        Type of regression model to use. Options are:
        - 'linear': Linear regression (default).
        - 'random_forest': Random forest regression.
        - 'svr': Support vector regression.

    Returns
    -------
    shap_summary : pd.DataFrame
        DataFrame summarizing the mean absolute SHAP values for each feature.
        Columns: ["Mean |SHAP Value|"].
    shap_values : np.ndarray
        Array of SHAP values with shape (n_samples, n_features), representing
        the contribution of each feature to the model's predictions.
    """

    if shap is None:
        raise ImportError("SHAP is not installed. Install with: pip install shap")
    if model_type == 'linear':
        model = LinearRegression()
        explainer_class = shap.LinearExplainer
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42)
        explainer_class = shap.TreeExplainer
    elif model_type == 'svr':
        model = SVR()
        explainer_class = shap.KernelExplainer
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # 拟合模型
    model.fit(X, y)

    # 创建 SHAP Explainer
    if (model_type == 'svr')|(model_type == 'linear'):
        explainer = explainer_class(model, X)
    else:
        explainer = explainer_class(model)

    # 计算 SHAP 值
    shap_values = explainer.shap_values(X)
    # Summarize SHAP values
    shap_summary = pd.DataFrame(
        data=np.mean(np.abs(shap_values), axis=0),
        columns=["Mean |SHAP Value|"]
    )

    return shap_summary, shap_values
