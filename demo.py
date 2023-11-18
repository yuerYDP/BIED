import numpy as np
import cv2
import os


def circular_avg_kernel(radius):
    """
        circular_avg_kernel(dtype=float)

            Return a circular mean filter.

            Parameters
            ----------
            radius: int or float
                Half of filter size, e.g., ``3`` or ``2.5``.

            Returns
            -------
            out: ndarray
                A two-dimension array.
        """
    diameter = int(2 * radius)
    if diameter % 2 == 0:
        diameter += 1
    centre = diameter // 2
    kernel = np.zeros([diameter, diameter], dtype=np.float32)
    for i in range(diameter):
        for j in range(diameter):
            if (i - centre) ** 2 + (j - centre) ** 2 <= radius ** 2:
                kernel[i, j] = 1
    return kernel / kernel.sum()
def zero_kernel_centre(kernel, centre_radius):
    """
        zero_kernel_centre(kernel, dtype=float)

        Set the kernel center to 0.

        Parameters
        ----------
        kernel: ndarray
            A two-dimension array.
        radius: int or float
            Half of center size, e.g., ``3`` or ``2.5``.
        Returns
        -------
        out: ndarray
            A two-dimension array.
    """
    if centre_radius == 0:
        return kernel
    centre = kernel.shape[0] // 2
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if (i - centre) ** 2 + (j - centre) ** 2 <= centre_radius ** 2:
                kernel[i, j] = 0
    return kernel


def get_gaussian_kernel_size(sigma):
    """
        get_gaussian_kernel(dtype=float)

        Return an integer.

        Parameters
        ----------
        sigma: float
            A floating point number greater than 0 to determine the size of the kernel.

        Returns
        -------
        out: int
            An integer representing the size of the Gaussian kernel.
    """
    threshold = 1e-3
    radius = np.sqrt(-np.log(np.sqrt(2 * np.pi) * sigma * threshold) * (2 * sigma * sigma))
    radius = int(np.ceil(radius))
    kernel_size = radius * 2 + 1
    return kernel_size
def gaussian_gradient_kernel(sigma, theta, seta):
    """
        gaussian_gradient_kernel(dtype=float, dtype=float, dtype=float)

        Return the Gaussian gradient with orientation theta and spatial aspect ratio seta.

        Parameters
        ----------
        sigma: float
            A floating point number greater than 0 to determine the size of the kernel.
        theta: float
            A floating point number to determine the orientation of the kernel.
        seta: float
            A floating point number to determine the spatial aspect ratio of the kernel.

        Returns
        -------
        out: ndarray
            A two-dimension gaussian gradient kernel.
    """
    k_size = get_gaussian_kernel_size(sigma)
    kernel = np.zeros([k_size, k_size], dtype=np.float32)
    sqr_sigma = sigma ** 2
    width = k_size // 2
    for i in range(k_size):
        for j in range(k_size):
            y1 = i - width
            x1 = j - width
            x = x1 * np.cos(theta) + y1 * np.sin(theta)
            y = - x1 * np.sin(theta) + y1 * np.cos(theta)
            kernel[i, j] = - x * np.exp(-(x ** 2 + y ** 2 * seta ** 2) / (2 * sqr_sigma)) / (np.pi * sqr_sigma)
    return kernel
def ellipse_gaussian_kernel(sigma_x, sigma_y, theta):
    """
    ellipse_gaussian_kernel(dtype=float, dtype=float, dtype=float)

    Return an ellipse gaussian kernel with orientation theta.

    Parameters
    ----------
    sigma_x: float
        A floating point number indicating the scale in the X direction.
    sigma_y: float
        A floating point number indicating the scale in the Y direction.
    theta: float
        A floating point number to determine the orientation of the kernel.

    Returns
    -------
        out: ndarray
            A two-dimension ellipse gaussian kernel.
    """
    max_sigma = max(sigma_x, sigma_y)
    kernel_size = get_gaussian_kernel_size(max_sigma)

    centre_x = kernel_size // 2
    centre_y = kernel_size // 2
    sqr_sigma_x = sigma_x ** 2
    sqr_sigma_y = sigma_y ** 2

    a = np.cos(theta) ** 2 / (2 * sqr_sigma_x) + np.sin(theta) ** 2 / (2 * sqr_sigma_y)
    b = - np.sin(2 * theta) / (4 * sqr_sigma_x) + np.sin(2 * theta) / (4 * sqr_sigma_y)
    c = np.sin(theta) ** 2 / (2 * sqr_sigma_x) + np.cos(theta) ** 2 / (2 * sqr_sigma_y)

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x = i - centre_x
            y = j - centre_y
            kernel[i, j] = np.exp(- (a * x ** 2 + 2 * b * x * y + c * y ** 2))

    return kernel / kernel.sum()
def late_inhi_kernel(sigma_x, sigma_y, delta_degree, theta):
    """
    late_inhi_kernel(dtype=float, dtype=float, dtype=int, dtype=float)

    This function first call ellipse_gaussian_kernel() with orientation theta to get the orient_kernel.
    Then rotate the orient_kernel, substract the orient_kernel, and compute the maximal non-zero value at each location.
    At last, return the lateral inhibition kernel.

    Parameters
    ----------
    sigma_x: float
        A floating point number indicating the scale of the ellipse gaussian kernel in the X direction.
    sigma_y: float
        A floating point number indicating the scale of the ellipse gaussian kernel in the Y direction.
    delta_degree: int
        An integer to determine the number of rotation.
    theta: float
        A floating point number to determine the orientation of the ellipse gaussian kernel.

    Returns
    -------
    out: ndarray
        A two-dimension kernel.
    """
    orient_kernel = ellipse_gaussian_kernel(sigma_x, sigma_y, theta=theta)

    n_degree = 180 // delta_degree
    orient_kernels = []
    for idx in range(3 * n_degree):
        k = ellipse_gaussian_kernel(sigma_x, sigma_y, theta=idx * delta_degree / 3 / 180 * np.pi)
        k = k - orient_kernel
        k[k < 0] = 0
        orient_kernels.append(k)

    kernel = np.max(np.array(orient_kernels), axis=0)
    return kernel / kernel.sum()
def texture_gradient_kernel(theta, radius):
    """
    texture_boundary_kernel(dtype=int, dtype=float)

    Return the texture gradient kernel with orientation theta.

    Parameters
    ----------
    theta: int
        An integer to determine the orientation of the texture gradient kernel.
    radius: float
        A floating piont number determining the size of the kernel.

    Returns
    -------
    out: ndarray
        A two-dimension kernel.
    """
    kernel = circular_avg_kernel(radius)
    center = int(np.floor(radius))

    # 1. top half circle
    top_half_kernel = np.copy(kernel)
    top_half_kernel[center:, :] = 0
    M = cv2.getRotationMatrix2D((kernel.shape[0] // 2, kernel.shape[1] // 2), -theta, 1)
    top_half_kernel = cv2.warpAffine(top_half_kernel, M, (kernel.shape[1], kernel.shape[0]))

    # 2. bottom half circle
    bottom_half_kernel = np.copy(kernel)
    bottom_half_kernel[:center + 1, :] = 0
    M = cv2.getRotationMatrix2D((kernel.shape[0] // 2, kernel.shape[1] // 2), -theta, 1)
    bottom_half_kernel = cv2.warpAffine(bottom_half_kernel, M, (kernel.shape[1], kernel.shape[0]))

    result_kernel = top_half_kernel - bottom_half_kernel
    result_kernel = result_kernel - kernel / radius
    return result_kernel


def compute_local_contrast(gray_img, radius):
    """
    compute_local_contrast(ndarray, dtype=float)

    Get the local contrast of the grayscale image.

    Parameters
    ----------
    gray_img: ndarray
        Grayscale image.
    radius:
        Area radius for computing the local contrast.

    Returns
    -------
    out: ndarray
        A two_dimension array representing the local contrast of the grayscale image.
    """
    avg_kernel = circular_avg_kernel(radius)
    mean_img = cv2.filter2D(gray_img, ddepth=-1, kernel=avg_kernel, borderType=cv2.BORDER_REFLECT)
    variance_img = np.power(gray_img - mean_img, 2)
    mean_variance_img = cv2.filter2D(variance_img, ddepth=-1, kernel=avg_kernel, borderType=cv2.BORDER_REFLECT)
    std_deviation_img = np.sqrt(mean_variance_img)
    local_contrast = np.sqrt(std_deviation_img)
    return local_contrast
def RetinaLGN(input_img):
    """
    RetinaLGN(ndarray)

    Simulate the retina and lateral geniculate nucleus to encode the image into several parallel channels:
    rg, by, im_rg, im_by, luminance, luminance contrast.

    Parameters:
    input_img: ndarray
        Primary image with shape (H, W, 3).

    Returns:
    out: ndarray
        Parallel channels with shape (H, W, 6).
    """
    sqrt_img = np.sqrt(input_img)

    channels = np.zeros([input_img.shape[0], input_img.shape[1], 6], dtype=np.float32)
    channels[:, :, 0] = sqrt_img[:, :, 2] - sqrt_img[:, :, 1]
    channels[:, :, 1] = sqrt_img[:, :, 0] - 0.5 * (sqrt_img[:, :, 2] + sqrt_img[:, :, 1])
    channels[:, :, 2] = sqrt_img[:, :, 2] - 0.5 * sqrt_img[:, :, 1]
    channels[:, :, 3] = sqrt_img[:, :, 0] - 0.5 * 0.5 * (sqrt_img[:, :, 2] + sqrt_img[:, :, 1])
    # lu = np.sum(sqrt_img, axis=2) / 3
    lu = 0.114 * sqrt_img[:, :, 0] + 0.587 * sqrt_img[:, :, 1] + 0.299 * sqrt_img[:, :, 2]
    channels[:, :, 4] = lu
    lc = compute_local_contrast(lu, radius=2.5)
    channels[:, :, 5] = lc
    return channels


def V1_surround_modulation(orientation_edge):
    """
    V1_surround_modulation(ndarray)

    Design a new surround modulation mechanism to modulate the response of the classical receptive field.

    Parameters
    ----------
    orientation_edge: ndarray
        A ndarray with shape (H, W, num_orientation).

    Returns
    -------
    out: ndarray
        A ndarray with shape (H, W, num_orientation).
    """
    num_orientation = orientation_edge.shape[2]

    same_faci_kernels = []
    late_inhi_kernels = []
    for i in range(num_orientation):
        sigma_x = 0.3
        sigma_y = 2.0
        same_faci_k = ellipse_gaussian_kernel(sigma_x, sigma_y, i / num_orientation * np.pi)
        same_faci_k = zero_kernel_centre(same_faci_k, 0.5)
        same_faci_kernels.append(same_faci_k)

        sigma_x = 0.3
        sigma_y = 2.0
        late_inhi_k = late_inhi_kernel(sigma_x, sigma_y, 15, i / num_orientation * np.pi)
        late_inhi_k = zero_kernel_centre(late_inhi_k, 0.5)
        late_inhi_kernels.append(late_inhi_k)

    sum_edge = np.sum(orientation_edge, axis=2)
    avg_edge = sum_edge / num_orientation
    full_inhi_radius = same_faci_kernels[0].shape[0] // 2
    full_inhi_k = circular_avg_kernel(full_inhi_radius)
    full_inhi = cv2.filter2D(avg_edge, ddepth=-1, kernel=full_inhi_k, borderType=cv2.BORDER_CONSTANT)
    orientation_edge_sm = np.zeros(orientation_edge.shape, dtype=np.float32)
    for i in range(num_orientation):
        d_edge = orientation_edge[:, :, i]
        same_faci_k = same_faci_kernels[i]
        same_faci = cv2.filter2D(d_edge, ddepth=-1, kernel=same_faci_k)
        late_inhi_k = late_inhi_kernels[i]
        late_inhi = cv2.filter2D(d_edge, ddepth=-1, kernel=late_inhi_k)

        temp = d_edge + same_faci - 1.0 * late_inhi - 0.8 * full_inhi
        temp[temp < 0] = 0
        orientation_edge_sm[:, :, i] = temp

    return orientation_edge_sm
def V1_texture_boundary(texture_info):
    """
    V1_texture_boundary(ndarray)

    Get texture boundaries based on texture information.

    Parameters
    ----------
    texture_info: ndarray
        A ndarray with shape (H, W).

    Returns
    -------
    out: ndarray
        A ndarray with shape (H, W).
    """
    radiuses = [3.5, 5.5, 8.5, 12.5, 17.5]

    texture_boundary = np.zeros([texture_info.shape[0], texture_info.shape[1], 24], dtype=np.float32)
    for radius in radiuses:
        n_directions = 24
        texture_boundary_24s = np.zeros([texture_info.shape[0], texture_info.shape[1], 24], dtype=np.float32)
        for idx in range(n_directions):
            theta = idx * 15
            texture_boundary_k = texture_gradient_kernel(theta, radius)
            temp = cv2.filter2D(texture_info, ddepth=-1, kernel=texture_boundary_k, borderType=cv2.BORDER_REFLECT)

            dis = - texture_boundary_k.shape[0] // 2 // 2
            M = np.float32([[1, 0, dis * np.cos(theta / 180 * np.pi + np.pi / 2)],
                            [0, 1, dis * np.sin(theta / 180 * np.pi + np.pi / 2)]])
            temp_result = cv2.warpAffine(temp, M, (temp.shape[1], temp.shape[0]))

            texture_boundary_24s[:, :, idx] = temp_result

        texture_boundary[:, :, :] += texture_boundary_24s

    return texture_boundary
def V1_get_orient_edge(crf_response, pri_shape):
    """
    V1_get_orient_edge(ndarray, pri_shape)

    Parameters
    ----------
    crf_response: ndarray
        A ndarray with shape (H, W, num_orientation).
    pri_shape: tuple of ints
        Shape is used to resize to the primary image size.

    Returns
    -------
    out: ndarray
        A two-dimension array with pri_shape.
    """
    rf_response = V1_surround_modulation(crf_response)
    rf_response = cv2.resize(rf_response, (pri_shape[1], pri_shape[0]), interpolation=cv2.INTER_LINEAR)
    sc_edge = np.max(rf_response, axis=2)

    texture_boundary_24 = V1_texture_boundary(np.sum(crf_response, axis=2))
    texture_boundary_24 = cv2.resize(texture_boundary_24, (pri_shape[1], pri_shape[0]), interpolation=cv2.INTER_LINEAR)
    texture_boundary = np.max(texture_boundary_24, axis=2)

    max_val = sc_edge.max()
    sc_edge[sc_edge > max_val / 7] = max_val / 7
    sc_edge /= sc_edge.max()

    max_val = texture_boundary.max()
    texture_boundary[texture_boundary > max_val / 7] = max_val / 7
    texture_boundary /= texture_boundary.max()

    edge = sc_edge * texture_boundary
    return edge
def V1(chs, num_orientation, rf_size_levels):
    """
    V1(ndarray, dtype=int, dtype=int)

    Get the response of the primary visual cortex.

    Parameters
    ----------
    chs: ndarray
        Input channels.
    num_orientation: int
        The number of orientations of simple cells.
    rf_size_levels: int
        Levels of simple cells with different receptive field sizes.

    Returns
    -------
    out: ndarray
        A ndarray with shape (H, W, num_ch, num_orientation).
    """
    edge_response = np.zeros([chs.shape[0], chs.shape[1], chs.shape[2], rf_size_levels], dtype=np.float32)

    for level in range(rf_size_levels):
        for idx_c in range(chs.shape[2]):
            input_img = chs[:, :, idx_c]

            pri_shape = input_img.shape
            input_img = cv2.resize(input_img, (0, 0), fx=1 / np.power(2, level), fy=1 / np.power(2, level))

            crf_response = np.zeros([input_img.shape[0], input_img.shape[1], num_orientation], dtype=np.float32)
            for idx in range(num_orientation):
                theta = idx / num_orientation * np.pi
                crf_k = gaussian_gradient_kernel(1, theta, 0.5)
                crf = np.abs(cv2.filter2D(input_img, ddepth=-1, kernel=crf_k, borderType=cv2.BORDER_REFLECT))
                crf_response[:, :, idx] = crf

            edge = V1_get_orient_edge(crf_response, pri_shape)
            edge_response[:, :, idx_c, level] = edge

    return edge_response


def V2_ep_modulation(ch_edge):
    """
    V2_ep_modulation(ndarray)

    Simulate endpoint neurons to modulate edge response.

    Parameters
    ----------
    ch_edge: ndarray
        A two-dimension array with shape (H, W).

    Returns
    -------
    out: ndarray
        A two-dimension array with shape (H, W).
    """
    line_num_k = modeldata.l5_kernels.shape[2]
    line_l5_kernels = modeldata.l5_kernels
    line_response = np.zeros([ch_edge.shape[0], ch_edge.shape[1], line_num_k], dtype=np.float32)
    for i in range(line_num_k):
        line_response[:, :, i] = cv2.filter2D(ch_edge, ddepth=-1, kernel=line_l5_kernels[:, :, i], borderType=cv2.BORDER_CONSTANT)

    line_l3_kernels = modeldata.l3_kernels
    line_ep_left_kernels = modeldata.ep_left_kernels
    line_ep_right_kernels = modeldata.ep_right_kernels
    line_ep_inhi_response = np.zeros([ch_edge.shape[0], ch_edge.shape[1], line_num_k], dtype=np.float32)
    for i in range(line_num_k):
        line_ep_left = cv2.filter2D(ch_edge, ddepth=-1, kernel=line_ep_left_kernels[:, :, i], borderType=cv2.BORDER_CONSTANT)
        line_ep_right = cv2.filter2D(ch_edge, ddepth=-1, kernel=line_ep_right_kernels[:, :, i], borderType=cv2.BORDER_CONSTANT)
        line_ep_inhi = np.abs(line_ep_left - line_ep_right)
        line_ep_inhi_response[:, :, i] = cv2.filter2D(line_ep_inhi, ddepth=-1, kernel=line_l3_kernels[:, :, i], borderType=cv2.BORDER_CONSTANT)

    for i in range(line_num_k):
        line_response[:, :, i] = line_response[:, :, i] - 2 * line_ep_inhi_response[:, :, i]

    line_cc_edge = np.max(line_response, axis=2)
    line_cc_edge[line_cc_edge < 0] = 0

    return line_cc_edge
def V2(V1_response):
    """
    V2(ndarray)

    Call V2_ep_modulation() to modulate edge response.

    Parameters
    ----------
    V1_response: ndarray
        A ndarray with shape (H, W, num_orientation).

    Returns
    -------
    out: ndarray
        A ndarray with shape (H, W, num_orientation).
    """
    dims = V1_response.shape
    result = np.zeros([dims[0], dims[1], dims[2]], dtype=np.float32)
    for level in range(dims[3]):
        single_size_edges = np.zeros([dims[0], dims[1], dims[2]], dtype=np.float32)
        for idx_c in range(dims[2]):
            edge = V1_response[:, :, idx_c, level]
            single_size_edges[:, :, idx_c] = V2_ep_modulation(edge)
            # single_size_edges[:, :, idx_c] = edge
        result[:, :, :] += single_size_edges / (level + 1)

    return result


def V4(V2_response):
    """
    V4(ndarray)

    Sum the edge response of all channels.

    Parameters
    ----------
    V2_response: ndarray
        A ndarray with shape (H, W, num_channel)

    Returns
    -------
    out: ndarray
        A two-dimension with shape (H, W).
    """
    V4_response = np.sum(V2_response, axis=2)
    V4_response = V4_response / V4_response.max()
    return V4_response


def BIED(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    input_img = img / img.max()

    channels = RetinaLGN(input_img)
    V1_response = V1(channels, 12, 4)
    V2_response = V2(V1_response)
    edge = V4(V2_response)

    return edge


class ModelData():
    def __init__(self):
        data_dir = os.path.join(os.path.abspath(os.curdir), "model_data")
        self.l5_kernels = np.load(os.path.join(data_dir, "line_l5.npy"))
        self.l3_kernels = np.load(os.path.join(data_dir, "line_l3.npy"))
        self.ep_left_kernels = np.load(os.path.join(data_dir, "ep_left_l3.npy"))
        self.ep_right_kernels = np.load(os.path.join(data_dir, "ep_right_l3.npy"))


modeldata = ModelData()
if __name__ == "__main__":
    pic_path = os.path.join(os.path.abspath(os.curdir), "example_1.jpg")
    edge = BIED(pic_path)
    result_path = os.path.join(os.path.abspath(os.curdir), "example_1_result.png")
    cv2.imwrite(result_path, edge * 255)