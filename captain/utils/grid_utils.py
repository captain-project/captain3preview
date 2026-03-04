import numpy as np
import torch
from numba import jit
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


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


@jit(nopython=True)
def _dispersal_distances_threshold_coords(lambda_0: float, coords: tuple, threshold=3):
    length = len(coords[0])
    lat_flat, lon_flat = coords
    exp_rate = 1.0 / lambda_0
    dumping_dist = np.zeros((length, length), dtype=np.float32)
    for i in range(0, length):
        for n in range(0, length):
            if (
                    abs(lat_flat[i] - lat_flat[n]) <= threshold
                    and abs(lon_flat[i] - lon_flat[n]) <= threshold
            ):
                # relative dispersal probability: always 1 at distance = 0
                # the actual number of offspring is modulated by growth_rate
                # print(i, j, n, m)
                dumping_dist[i, n] = np.exp(
                    -exp_rate
                    * np.sqrt(
                        (lat_flat[i] - lat_flat[n]) ** 2
                        + (lon_flat[i] - lon_flat[n]) ** 2
                    )
                )

    return dumping_dist


def dispersal_distances_threshold_coords(lambda_0: float, coords: tuple, threshold=3):
    return csr_matrix(
        _dispersal_distances_threshold_coords(lambda_0, coords, threshold)
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
    mask = ~np.isnan(array_3d[0]) if mask is None else ~np.isnan(mask)

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
