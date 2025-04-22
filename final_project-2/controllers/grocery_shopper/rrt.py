import numpy as np
import matplotlib.pyplot as plt
import math
import random

###############################################################################
## Base Code
###############################################################################
class Node:
    """
    Node for RRT/RRT* Algorithm. This is what you'll make your graph with!
    """
    def __init__(self, pt, parent=None):
        self.point = pt # n-Dimensional point
        self.parent = parent # Parent node
        self.path_from_parent = [] # List of points along the way from the parent node (for visualization)
        self.cost = 0.0 # length of path from start node to itself

frontiers = [] # is explored and has at least one neighbor that is unexplored
unknown = [] # map value of 0
explored = [] # map value of 1
obstacles = [] # map value of 2
node_list = [] # list of nodes being navigated to in the map
current_position = ()

# for testing purposes
testing_map = np.zeros(shape=[360,360])
testing_map

# for testing purposes - NOT FINISHED YET!
def visualize_2D_graph(state_bounds, obstacles, nodes, goal_point=None, filename=None):
    '''
    @param state_bounds Array of min/max for each dimension
    @param obstacles Locations and radii of spheroid obstacles
    @param nodes List of vertex locations
    @param edges List of vertex connections
    '''

    fig = plt.figure()
    plt.xlim(state_bounds[0,0], state_bounds[0,1])
    plt.ylim(state_bounds[1,0], state_bounds[1,1])

    for obs in obstacles:
        plt.plot(obs, color='red')
        # plot_circle(obs[0][0], obs[0][1], obs[1])

    goal_node = None
    for node in nodes:
        if node.parent is not None:
            node_path = np.array(node.path_from_parent)
            plt.plot(node_path[:,0], node_path[:,1], '-b')
        if goal_point is not None and np.linalg.norm(node.point - np.array(goal_point)) <= 1e-5:
            goal_node = node
            plt.plot(node.point[0], node.point[1], 'k^')
        else:
            plt.plot(node.point[0], node.point[1], 'ro')

    plt.plot(nodes[0].point[0], nodes[0].point[1], 'ko')

    if goal_node is not None:
        cur_node = goal_node
        while cur_node is not None: 
            if cur_node.parent is not None:
                node_path = np.array(cur_node.path_from_parent)
                plt.plot(node_path[:,0], node_path[:,1], '--y')
                cur_node = cur_node.parent
            else:
                break

    if goal_point is not None:
        plt.plot(goal_point[0], goal_point[1], 'gx')


    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()

def map_update(map):
    frontiers.clear()
    unknown.clear()
    explored.clear()
    obstacles.clear()
    rows, cols = map.shape
    for i in range(rows):
        for j in range(cols):
            if map[i, j] == 1: # if the current element is explored
                explored.append((i, j))

                # get valid neighbors
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        neighbori, neighborj = i+di, j+dj
                        if di == 0 and dj == 0:
                            continue
                        if 0 <= neighbori < rows and 0 <= neighborj < cols:
                            neighbors.append(map[neighbori, neighborj])

                if 0 in neighbors: # if a neighbor is unknown
                    frontiers.append((i, j))
            if map[i, j] == 0: # if the current element is unknown
                unknown.append((i, j))
            if map[i, j] == 2: # if the current element is an obstacle
                obstacles.append((i, j))
    
def get_random_frontier_vertex():
    return frontiers[np.random.randint(0, len(frontiers)-1)]

def get_random_valid_vertex():
    return explored[np.random.randint(0, len(explored)-1)]

def point_is_valid(point, map):
    '''
    @param point: n-dimensional array representing a point, should be a set of integers
    @param map: 2D array that holds the map values
    @return bool value that represents whether the point is valid (a free space) or not
    '''
    if map[point[0], point[1]] == 1: # if the point is an explored, free space
        return True
    else:
        return False

# END BASE CODE
#####################################


def get_nearest_vertex(node_list, q_point):
    '''
    @param node_list: List of Node objects
    @param q_point: n-dimensional array representing a point, should be a set of integers
    @return Node in node_list with closest node.point to query q_point
    '''

    min_node = None
    min_dist = math.inf

    for node in node_list:
        distance = np.linalg.norm(np.array(node.point)-np.array(q_point))

        if distance < min_dist:
            min_dist = distance
            min_node = node

    return min_node

    raise NotImplementedError


def steer(from_point, to_point, delta_q):
    '''
    @param from_point: n-Dimensional array (point) where the path to "to_point" is originating from (e.g., [1.,2.])
    @param to_point: n-Dimensional array (point) indicating destination (e.g., [0., 0.])
    @param delta_q: Max path-length to cover, possibly resulting in changes to "to_point" (e.g., 0.2)
    @returns path: list of points leading from "from_point" to "to_point" (inclusive of endpoints)  (e.g., [ [1.,2.], [1., 1.], [0., 0.] ])
    '''

    path = []

    from_point = np.array(from_point)
    to_point = np.array(to_point)
    dist = np.linalg.norm(to_point-from_point)

    if dist > delta_q:
        to_point = from_point + ((to_point - from_point)/dist)*delta_q
    
    path = np.linspace(from_point, to_point, 10, True)

    for point in path:
        point = np.round(point, 0) # rounds the point to the nearest (x, y) indices in the map - needed for checking obstacles later

    return path


# use this method to retrieve waypoints for the robot to navigate to a frontier
def rrt_star(state_bounds, obstacles, point_is_valid, starting_point, goal_point, k, delta_q):
    '''
    @param state_bounds: matrix of min/max values for each dimension (e.g., [[0,1],[0,1]] for a 2D 1m by 1m square)
    @param state_is_valid: function that maps states (N-dimensional Real vectors) to a Boolean (indicating free vs. forbidden space)
    @param k: Number of points to sample
    @param delta_q: Maximum distance allowed between vertices
    @returns List of RRT* graph nodes and the path from the starting node to the goal node
    '''

    node_list = []
    starting_node = Node(starting_point, parent=None)
    node_list.append(starting_node) # Add Node at starting point with no parent
    path_to_source = []

    def get_nearby_nodes(center_point):
        nearby = []
        for node in node_list:
            if math.dist(center_point, node.point) <= delta_q:
                nearby.append(node)

        return nearby

    for i in range(k):
        q_rand = []
        rand_num = random.random()
        if goal_point is not None and rand_num < 0.05:
            q_rand = goal_point
        else:
            q_rand = get_random_valid_vertex()
        
        # finding closest node and then calling steer function to get the location/path of new node
        q_near = get_nearest_vertex(node_list, q_rand)
        path_to_new_node = steer(q_near.point, q_rand, delta_q)
        new_node_point = path_to_new_node[len(path_to_new_node)-1]

        nearby_nodes = get_nearby_nodes(new_node_point)

        min_cost = math.inf
        min_cost_node = None

        # finding the nearby node that has the smallest cost
        for curr in nearby_nodes:
            if curr.cost < min_cost:
                min_cost = curr.cost
                min_cost_node = curr

        # if the smallest cost in nearby nodes is smaller than the original q_near cost, then make this smallest cost node the new parent of q_new
        if min_cost < q_near.cost:
            new_parent = min_cost_node
        else:
            new_parent = q_near

        # find the new path to the new parent
        path_to_new_node = steer(new_parent.point, new_node_point, delta_q)

        # checking that the node and path are in valid spaces
        if not all(point_is_valid(x) for x in path_to_new_node):
            continue
        
        # creating the new node and appending to node list
        q_new = Node(new_node_point, new_parent)
        q_new.path_from_parent = path_to_new_node
        q_new.cost = q_new.parent.cost + math.dist(q_new.point, q_new.parent.point)
        node_list.append(q_new)

        # rewiring the nearby neighbors with the new node
        for curr in nearby_nodes:
            # if the shortest path from start node to curr is through the new node, then rewire
            if q_new.cost + math.dist(q_new.point, curr.point) < curr.cost:
                path_to_q_new = steer(q_new.point, curr.point, delta_q)
                # check to make sure the new path is actually valid
                if all(point_is_valid(x) for x in path_to_q_new):
                    # if path is valid, then create the new edge from curr to new node
                    curr.parent = q_new
                    curr.path_from_parent = path_to_q_new
                    curr.cost = q_new.cost + math.dist(q_new.point, curr.point)

        # return if we're close enough to the goal point
        if goal_point is not None and math.dist(new_node_point, goal_point) <= 1e-5:
            cur_node = q_new

            # populate path_to_source with nodes from source node to q_new
            while cur_node is not starting_node: 
                path_to_source.append(cur_node)
                if cur_node.parent is not None:
                    cur_node = cur_node.parent
                else:
                    break

            return node_list, path_to_source.reverse()

    return node_list, path_to_source.reverse()