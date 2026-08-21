import numpy as np
import rasterio
import torch
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree, cKDTree
from sklearn.neighbors import NearestNeighbors


class DistanceFromTarget:

    def __init__(self, coords: tuple, target_cell_mask: torch.Tensor):
        y_coords = np.asarray(coords[0])
        x_coords = np.asarray(coords[1])
        self.coords = np.stack([y_coords, x_coords], axis=1)

        # 1. Store the target mask as a 1D numpy array so we can reference it later
        self.target_mask_np = target_cell_mask.view(-1).cpu().numpy()
        self.coords_target = self.coords[self.target_mask_np]  # Shape: (N_targets, 2)

    def calculate_min_distances(
            self, valid_cell_mask: torch.Tensor, normalize: bool = False
    ) -> torch.Tensor:
        # 2. Flatten the input mask to 1D internally
        valid_cell_mask_1d = valid_cell_mask.view(-1)
        valid_mask_np = valid_cell_mask_1d.cpu().numpy()

        # 3. EXCLUSION STEP: Filter out target cells from the valid mask
        # calc_mask_np is True ONLY for cells that are valid AND NOT targets
        calc_mask_np = valid_mask_np & ~self.target_mask_np

        # 4. Extract coordinates ONLY for the non-overlapping valid cells
        coords_calc = self.coords[calc_mask_np]  # Shape: (N_calc, 2)

        # 5. Handle safety edge cases
        if len(self.coords_target) == 0 or len(coords_calc) == 0:
            calc_distances = np.full(len(coords_calc), np.inf)
        else:
            tree = KDTree(self.coords_target)
            # Query closest target ONLY for non-overlapping cells
            calc_distances, _ = tree.query(coords_calc)

        if normalize and len(calc_distances) > 0:
            max_dist = np.nanmax(calc_distances)
            if max_dist > 0:
                calc_distances /= max_dist

        # 6. Initialize a 1D result tensor filled with NaNs
        result_1d = torch.full_like(
            valid_cell_mask_1d, float("nan"), dtype=torch.float32
        )

        # 7. Assign calculated distances ONLY to non-overlapping valid slots
        # Overlapping cells (and invalid cells) remain as NaN
        result_1d[calc_mask_np] = (
            torch.from_numpy(calc_distances).float().to(valid_cell_mask.device)
        )

        # 8. Restore original input tensor shape
        return result_1d.view(valid_cell_mask.shape)


def dispersal_distances_threshold_coords_kdtree(
        lambda_0: float, coords: tuple, threshold=3
):
    lat_flat = np.asarray(coords[0])
    lon_flat = np.asarray(coords[1])
    length = len(lat_flat)
    exp_rate = 1.0 / lambda_0

    # 1. Build spatial tree
    points = np.column_stack((lat_flat, lon_flat))
    tree = cKDTree(points)

    # 2. Find all pairs within the Chebyshev distance (p=np.inf)
    # This exactly mimics your `abs(lat_1-lat_2) <= threshold` logic
    pairs = tree.query_pairs(r=threshold, p=np.inf, output_type="ndarray")

    # Extract indices
    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]

    # 3. Calculate actual Euclidean distances for valid pairs only
    dists = np.sqrt(
        (lat_flat[i_idx] - lat_flat[j_idx]) ** 2
        + (lon_flat[i_idx] - lon_flat[j_idx]) ** 2
    )
    decay_data = np.exp(-exp_rate * dists)

    # 4. query_pairs only returns i < j (one side of the matrix).
    # We mirror the data for symmetry, and add the diagonal (i == j, distance 0).
    rows = np.concatenate([i_idx, j_idx, np.arange(length)])
    cols = np.concatenate([j_idx, i_idx, np.arange(length)])

    diag_data = np.ones(length, dtype=np.float32)  # exp(0) = 1
    data = np.concatenate([decay_data, decay_data, diag_data])

    # 5. Build and return sparse matrix directly
    return csr_matrix((data.astype(np.float32), (rows, cols)), shape=(length, length))


def scipy_sparse_to_torch(
        sparse_matrix: csr_matrix,
        device: torch.device,
) -> torch.Tensor:
    """Convert scipy CSR sparse matrix to PyTorch sparse CSR tensor.

    Args:
        sparse_matrix: SciPy CSR sparse matrix.
        device: Target PyTorch device.

    Returns:
        PyTorch sparse CSR tensor on the appropriate device.
        Note: MPS does not support sparse CSR tensors, so sparse
        matrices are kept on CPU for MPS devices.
    """
    sparse_matrix = sparse_matrix.tocsr()
    crow_indices = torch.from_numpy(sparse_matrix.indptr.astype(np.int64))
    col_indices = torch.from_numpy(sparse_matrix.indices.astype(np.int64))
    values = torch.from_numpy(sparse_matrix.data.astype(np.float32))

    # MPS doesn't support sparse tensors - keep on CPU
    sparse_device = "cpu" if device.type == "mps" else device

    return torch.sparse_csr_tensor(
        crow_indices,
        col_indices,
        values,
        size=sparse_matrix.shape,
        device=sparse_device,
        dtype=torch.float32,
    )


def dispersal_distances_threshold_coords(lambda_0: float, coords: tuple, threshold=3):
    return dispersal_distances_threshold_coords_kdtree(
        lambda_0=lambda_0, coords=coords, threshold=threshold
    )


def save_dispersal_distances(
        lambda_0: float, coords: tuple, threshold=3, filename: str | None = None
):
    m = dispersal_distances_threshold_coords(lambda_0, coords, threshold)
    if filename is None:
        return m
    else:
        sparse.save_npz(filename, m)


def load_dispersal_distances(filename: str):
    return sparse.load_npz(filename)


def flatten_grid(array_3d, mask=None):
    """
    array_3d: shape (channels, x, y)
    Returns:
        data_2d: shape (channels, valid_cells)
        coords: tuple of (x_indices, y_indices)
        original_shape: (x, y) to help reconstruction
    """
    # 1. Create a mask from the first channel where data is NOT NA
    # Change np.isnan to (array_3d[0] != mask_value) if using a specific fill value
    if mask is not None:
        mask[mask == 0] = np.nan
        mask = ~np.isnan(mask)
    else:
        mask = ~np.isnan(array_3d[0])

    # 2. Get the x, y coordinates of the valid cells
    # np.where returns a tuple of (array_of_x, array_of_y)
    coords = np.where(mask)

    # 3. Extract the data
    # Slicing with a mask on the spatial dimensions (axis 1 and 2)
    # We loop through channels to keep the (channels, valid_cells) structure
    data_2d = array_3d[:, mask]

    return data_2d, coords, array_3d.shape[1:]


def reconstruct_grid(data_2d, coords, original_spatial_shape):
    """
    data_2d: shape (channels, valid_cells)
    coords: tuple of (x_indices, y_indices)
    original_spatial_shape: (x, y)
    """
    channels = data_2d.shape[0]
    x_dim, y_dim = original_spatial_shape

    # 1. Initialize an array full of NAs
    reconstructed = np.full((channels, x_dim, y_dim), np.nan)

    # 2. Map the 2D data back using the coordinate indices
    # NumPy's advanced indexing makes this very efficient
    reconstructed[:, coords[0], coords[1]] = data_2d

    return reconstructed


def compute_convolution_matrix(
        coords: tuple[np.ndarray, np.ndarray], radius: int = 2
) -> csr_matrix:
    """Compute a row-normalized sparse convolution matrix for spatial averaging.

    Creates an adjacency matrix where each cell is connected to its neighbors
    within the specified radius (using Chebyshev/chessboard distance). The matrix
    is row-normalized so that multiplying a vector by this matrix computes
    the local average within each cell's neighborhood.

    Args:
        coords: Tuple of (x_indices, y_indices) for valid cells.
        radius: Neighborhood radius in cells (default 2 = 5x5 window).

    Returns:
        Sparse CSR matrix of shape (n_cells, n_cells), transposed for
        right-multiplication: result = values @ conv_matrix.
    """
    # Stack coords into (n_valid, 2)
    points = np.column_stack(coords)

    # Find all points within distance (Chebyshev distance for square window)
    # A 5x5 window means a radius of 2
    nn = NearestNeighbors(radius=radius, metric="chebyshev")
    nn.fit(points)
    adj = nn.radius_neighbors_graph(points, radius=radius, mode="connectivity")

    # Ensure it is a CSR matrix for fast multiplication
    adj = adj.tocsr()

    # Row-normalize using diagonal matrix multiplication (efficient for sparse)
    # This ensures each row sums to 1, computing neighborhood averages
    # and preventing edge bleeding for cells with fewer neighbors
    row_sums = np.array(adj.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero for isolated cells
    inv_row_sums = sparse.diags(1.0 / row_sums)
    normalized_adj = inv_row_sums @ adj

    # Transpose for right-multiplication convention: values @ conv_matrix
    return normalized_adj.T


def calculate_delta(
        map_present: np.ndarray, map_future: np.ndarray, n_steps: int | float
):
    delta = (map_future - map_present) / n_steps
    return delta


def extract_regional_centroids(
        tif_path: str, normalize: bool = True, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Reads a region map TIFF and returns a 3D PyTorch tensor of regional centroids.

    Args:
        tif_path: Path to the .tif file containing integer region IDs and NaNs.
        normalize: If True, scales coordinates to [0, 1] relative to map size.
        device: Target device for the tensor operations and output ('cpu' or 'cuda').

    Returns:
        A PyTorch tensor of shape (2, Height, Width) on the specified device.
        Channel 0: X-coordinate of the region's centroid for every cell in that region.
        Channel 1: Y-coordinate of the region's centroid for every cell in that region.
    """
    target_device = torch.device(device)

    # 1. Load the TIFF file using rasterio, then immediately convert to PyTorch
    with rasterio.open(tif_path) as src:
        # Read band 1 as float array to cleanly support NaN values
        np_map = src.read(1).astype(np.float32)

        if src.nodata is not None:
            np_map[np_map == src.nodata] = np.nan

        # Move the tensor to your ML execution device right away
        region_map = torch.from_numpy(np_map).to(target_device)

    H, W = region_map.shape

    # 2. Initialize a 3D tensor (2 channels, Height, Width) filled with NaNs
    centroid_layers = torch.full(
        (2, H, W), float("nan"), dtype=torch.float32, device=target_device
    )

    # 3. Find unique region IDs, filtering out the NaNs
    valid_pixels = region_map[~torch.isnan(region_map)]
    unique_regions = torch.unique(valid_pixels)

    # 4. Calculate centroids and broadcast using PyTorch mechanisms
    for region_id in unique_regions:
        # Create a PyTorch boolean mask
        region_mask = region_map == region_id

        # torch.nonzero(..., as_tuple=True) mimics np.where() perfectly
        y_indices, x_indices = torch.nonzero(region_mask, as_tuple=True)

        if x_indices.numel() == 0:
            continue

        # CRITICAL: PyTorch requires coordinate indices to be explicitly converted to
        # floats before computing a mean, otherwise it raises a RuntimeError.
        centroid_x = x_indices.float().mean()
        centroid_y = y_indices.float().mean()

        # Optional: Normalize coordinates between 0.0 and 1.0
        if normalize:
            centroid_x /= W
            centroid_y /= H

        # In-place broadcast the scalar centroids back to the correct region slots
        centroid_layers[0, region_mask] = centroid_x
        centroid_layers[1, region_mask] = centroid_y

    return centroid_layers
