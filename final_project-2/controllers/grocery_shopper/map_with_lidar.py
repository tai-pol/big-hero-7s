import numpy as np
import math
from scipy.ndimage import binary_dilation

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)


lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis



def globalcoords_to_map_coords(x_coor, y_coor) -> tuple[any, any]:
    """super important function takes in the global coords and returns them converted to map space"""
    try:
        res = ((x_coor + 15) * 10, (y_coor + 15) * 10)
        
        if res[0] < 0 or res[0] > 360:
            raise TypeError
        
        if res[1] < 0 or res[1] > 360:
            raise TypeError
    except Exception as e:
        print(e)
        return 1, 1
    
    return res

def map_coords_to_global_coords(x, y) -> tuple[any, any]:
    """takes in map coords and converts them to global coords for the Webots world"""
    try:
        res = ((x/10) - 15, (y/10) - 15)
        
        if res[0] < -15 or res[0] > 15:
            raise TypeError
        
        if res[1] < -8 or res[1] > 8:
            raise TypeError
    except Exception as e:
        print(e)
        return 1, 1
    
    return res

def get_points_between(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """
    Returns a numpy array of (x,y) points that lie on a straight line between (x0,y0) and (x1,y1).
    Uses vectorized operations for better performance.
    """
    # Calculate the number of points needed
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    n_points = max(dx, dy) + 1
    
    # Create linearly spaced points for both x and y
    x_points = np.linspace(x0, x1, n_points, dtype=int)
    y_points = np.linspace(y0, y1, n_points, dtype=int)
    
    # Combine into array of points
    points = np.column_stack((x_points, y_points))
    
    # Remove duplicate points
    return np.unique(points, axis=0)



def make_lidar_map(pose_x, pose_y, pose_theta, curr_map, lidar_sensor_readings, display):
    
    # how quickly the maps are made
    amount_add_per_cycle = .010
    explored_add_per_cycle = -.010
    
    # Convert robot position to display coordinates
    robot_x, robot_y = globalcoords_to_map_coords(pose_x, pose_y)

    
    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]
        exceeded_distance = False

        if rho > LIDAR_SENSOR_MAX_RANGE:
            rho = LIDAR_SENSOR_MAX_RANGE
            exceeded_distance = True
        
        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = math.cos(alpha)*rho
        ry = -math.sin(alpha)*rho

        t = pose_theta + np.pi/2.
        wx =  math.cos(t)*rx - math.sin(t)*ry 
        wy =  math.sin(t)*rx + math.cos(t)*ry 
        
        wx *= 10
        wy *= 10
        
        wx += robot_x
        wy += robot_y
            
        try:
            
            if not exceeded_distance:
                # Always update the obstacle point
                grey_val = min(curr_map[int(wx)][int(wy)] + amount_add_per_cycle, 1.0)
                curr_map[int(wx)][int(wy)] = grey_val
            
            # Only trace the line every 5th reading
            if i % 30 == 0:
                points = get_points_between(int(robot_x), int(robot_y), int(wx), int(wy))
                
                # Update map values for all points except the obstacle point (already done)
                for x, y in points:
                    if x != int(wx) or y != int(wy):  # Skip the obstacle point
                        grey_val = max(curr_map[x][y] + explored_add_per_cycle, -1.0)
                        curr_map[x][y] = grey_val
  
                
        except Exception as e:
            print(e)

                
            

    # Draw the robot's current pose on the 360x360 display
    display.setColor(int(0xFF0000))
    display.drawPixel(360-abs(int(pose_x)), abs(int(pose_y)))
    
    
    
def filter_lidar_map(lidar_map: np.ndarray):
    
    obstacle_cutoff = 0.95
    explored_cuttoff = -0.25
    
    size_x, size_y = lidar_map.shape    
    
    new_map = np.zeros_like(lidar_map)
    
    # for x in range(size_x):
    #     for y in range(size_y):
    #         val = lidar_map[x, y]
    #         if val > obstacle_cutoff:
    #             new_map[x, y] = 2  # Obstacle
    #         elif val < explored_cuttoff:
    #             new_map[x, y] = 1  # Explored
    #         else:
    #             new_map[x, y] = 0  # Unknown
    
    # optimized filtering code is here
    new_map = np.where(lidar_map > obstacle_cutoff, 2, np.where(lidar_map < explored_cuttoff, 1, 0))
    
    return new_map



def display_map(display, lidar_map_filtered):
    for x in range(lidar_map_filtered.shape[0]):
        for y in range(lidar_map_filtered.shape[1]):
            val = lidar_map_filtered[x, y]
            if val == 1:
                display.setColor(0x0000FF)  # Blue
            elif val == 2:
                display.setColor(0xFF0000)  # Red
            else:
                display.setColor(0x000000)  # Black
            display.drawPixel(x, y)
            


# in progress function if we need it
def expand_pixels(input_arr: np.ndarray, box_size: int = 3) -> np.ndarray:
    """
    Expands each non-zero pixel value into a square box in a new array.

    For each pixel with a non-zero value 'v' at location (r, c) in the
    input_arr, this function sets all pixels within a square box of
    size 'box_size' centered at (r, c) to the value 'v' in the output_arr.
    If boxes overlap, the value from the pixel processed later in the
    iteration (row-major order) will overwrite previous values.

    Args:
        input_arr: The 2D NumPy array to process.
        box_size: The side length of the square box to expand into.
                  Should ideally be an odd number for a clear center.
                  Defaults to 3 (for a 3x3 box).

    Returns:
        A new 2D NumPy array with the expanded pixel values.
    """
    if not isinstance(input_arr, np.ndarray) or input_arr.ndim != 2:
        raise ValueError("input_arr must be a 2D NumPy array.")
    if not isinstance(box_size, int) or box_size < 1:
        raise ValueError("box_size must be a positive integer.")
    # if box_size % 2 == 0:
    #     print(f"Warning: box_size ({box_size}) is even. "
    #           "The center is ambiguous; using integer division for radius.")

    rows, cols = input_arr.shape
    # Create a new array to store the results, initialized to zeros
    output_arr = np.zeros_like(input_arr)

    # Calculate radius for slicing
    radius = (box_size - 1) // 2
    radius_ceil = box_size // 2 # Needed for end slice index if box_size is even

    # --- Vectorized Approach using Dilation ---

    # 1. Define the structuring element (neighborhood shape)
    # A square box of size (box_size x box_size)
    structure = np.ones((box_size, box_size), dtype=bool)

    # 2. Create boolean masks for locations of 1s and 2s
    mask_1 = (input_arr == 1)
    mask_2 = (input_arr == 2)

    # 3. Dilate the masks - this expands the True regions
    dilated_mask_1 = binary_dilation(mask_1, structure=structure)
    dilated_mask_2 = binary_dilation(mask_2, structure=structure)

    # 4. Combine the results, giving priority to 2s
    # Start with zeros
    # Place 1s where the dilated mask for 1 is True
    output_arr[dilated_mask_1] = 1
    # Place 2s where the dilated mask for 2 is True (this overwrites any 1s)
    output_arr[dilated_mask_2] = 2

    return output_arr


    # --- Pass 1: Expand all the 1s ---
    for r in range(rows):
        for c in range(cols):
            value = input_arr[r, c]
            if value == 1: # Only process 1s in this pass
                # Calculate bounds, clamping to array edges
                r_start = max(0, r - radius)
                r_end = min(rows, r + radius_ceil + 1)
                c_start = max(0, c - radius)
                c_end = min(cols, c + radius_ceil + 1)

                # Stamp the value 1 onto the output array
                output_arr[r_start:r_end, c_start:c_end] = 1

    # --- Pass 2: Expand all the 2s (will overwrite 1s where boxes overlap) --
    for r in range(rows):
        for c in range(cols):
            value = input_arr[r, c]
            if value == 2: # Only process 2s in this pass
                # Calculate bounds, clamping to array edges
                r_start = max(0, r - radius)
                r_end = min(rows, r + radius_ceil + 1)
                c_start = max(0, c - radius)
                c_end = min(cols, c + radius_ceil + 1)

                # Stamp the value 2 onto the output array
                output_arr[r_start:r_end, c_start:c_end] = 2

    return output_arr

# def expand_pixels_v2(input_arr: np.ndarray, box_size: int = 3) -> np.ndarray:
#     rows, cols = input_arr.shape
#     # Create a new array to store the results, initialized to zeros
#     output_arr = np.zeros_like(input_arr)
    
#     kernel = np.ones((box_size, box_size))
    
# kernel = np.ones((10, 10))  # Define a 3x3 kernel of ones
# map = convolve2d(filtered_map, kernel, mode='same', boundary='wrap')  # Convolve the map with the kernel
# map = map > 0  # Convert the convolved map to binary, where 1 represents an obstacle or its configuration space
# plt.imshow(map, cmap='gray', origin="lower")
# plt.show()


# old code to draw as mapping
# display_grey = abs(grey_val)
# color = (display_grey*256**2 + display_grey*256 + display_grey)*255
# display.setColor(int(color))
# display.drawPixel(int(wx), int(wy))