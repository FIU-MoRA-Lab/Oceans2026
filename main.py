from utils import read_pkl, get_interpolator

path = "swe_data/fort_myers_result.pkl"
data = read_pkl(path)

# Extract speed components u and v: shape (T, 2, nx, ny)
speed_matrix = data["solution"][:, 1:3, :, :]
print("Speed matrix shape:", speed_matrix.shape)

# Create the interpolator
interp = get_interpolator(speed_matrix, config=data["config"])

# Test interpolator at a coordinate (lon, lat) and time (t)
lon = -81.85
lat = 26.65
t = 100.0

res = interp(lon, lat, t)
print(f"Interpolated speed at lon={lon}, lat={lat}, t={t} is: {res}")

# Find a coordinate where speed is maximum (non-zero)
import numpy as np
t_idx, c_idx, x_idx, y_idx = np.unravel_index(np.argmax(np.abs(speed_matrix)), speed_matrix.shape)
config = data["config"]
time_coords = np.linspace(0, config["t_final"], speed_matrix.shape[0])
x_coords = np.linspace(config["lon_range"][0], config["lon_range"][1], config["nx"])
y_coords = np.linspace(config["lat_range"][0], config["lat_range"][1], config["ny"])

max_lon = x_coords[x_idx]
max_lat = y_coords[y_idx]
max_t = time_coords[t_idx]

res_max = interp(max_lon, max_lat, max_t)
expected = speed_matrix[t_idx, :, x_idx, y_idx]
print(f"\nMax velocity index is at: t_idx={t_idx}, coordinate_idx={c_idx}, x_idx={x_idx}, y_idx={y_idx}")
print(f"Coordinates: lon={max_lon}, lat={max_lat}, t={max_t}")
print(f"Interpolated: {res_max}")
print(f"Expected:     {expected}")


