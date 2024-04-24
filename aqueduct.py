#import matplotlib.pyplot as plt

def read_grid(filename):
    with open(filename, 'r') as file:
        lines = file.read().strip().split('\n')
        grid_size = tuple(map(int, lines[0].split(', ')))
        heights = {(int(x), int(y)): int(h) for h, x, y in (line.split(', ') for line in lines[1:grid_size[0] * grid_size[1] + 1])}
        baths = [tuple(map(int, line.split(', '))) for line in lines[grid_size[0] * grid_size[1] + 1:]]
        return grid_size, heights, baths
    

def calculate_weight(current, target, heights):
    return max(-1, 1 + heights[target] - heights[current])


def create_graph(grid_size, heights):
    graph = {}
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            current_node = (x, y)
            neighbors = []
            # right node
            if (y + 1) < grid_size[1]:
                neighbor_node = (x, y + 1)
                weight = calculate_weight(current_node, neighbor_node, heights)
                neighbors.append((neighbor_node, weight))
            # left node
            if (y - 1) >= 0:
                neighbor_node = (x, y - 1)
                weight = calculate_weight(current_node, neighbor_node, heights)
                neighbors.append((neighbor_node, weight))
            # bottom node
            if (x + 1) < grid_size[0]:
                neighbor_node = (x + 1, y)
                weight = calculate_weight(current_node, neighbor_node, heights)
                neighbors.append((neighbor_node, weight))
            # top node
            if (x - 1) >= 0:
                neighbor_node = (x - 1, y)
                weight = calculate_weight(current_node, neighbor_node, heights)
                neighbors.append((neighbor_node, weight))
            
            graph[current_node] = neighbors
    return graph

# regular bellman ford algorithm
def bellman_ford(graph, source, target):
    distance = {node: float('inf') for node in graph}
    predecessor = {node: None for node in graph}
    distance[source] = 0

    for _ in range(len(graph) - 1):
        for node in graph:
            for neighbor, weight in graph[node]:
                if distance[node] + weight < distance[neighbor]:
                    distance[neighbor] = distance[node] + weight
                    predecessor[neighbor] = node

    for node in graph:
        for neighbor, weight in graph[node]:
            if distance[node] + weight < distance[neighbor]:
                raise ValueError("Graph has a negative cycle")
    
    path = []
    current = target
    while current is not None:
        path.insert(0, current)
        current = predecessor[current]

    if distance[target] == float('inf'):
        raise ValueError(f"no path from {source} to {target}")

    return path, distance[target]


def shortest_paths(graph, source):
    # use bellman ford to find shortest paths from source to all nodes
    distance = {node: float('inf') for node in graph}
    predecessor = {node: None for node in graph}
    distance[source] = 0
    for _ in range(len(graph) - 1):
        for node in graph:
            for neighbor, weight in graph[node]:
                if distance[node] + weight < distance[neighbor]:
                    distance[neighbor] = distance[node] + weight
                    predecessor[neighbor] = node
    return distance, predecessor

def shortest_path_to_baths(graph, source, baths):
    # find shortest paths from source to all bathhouses
    distance, _ = shortest_paths(graph, source)
    return {bath: distance[bath] for bath in baths}

# using bitwise operations to represent the state of visited bathhouses and for more efficient memory usage.
# This way it doesn't take a super long time to run.
def min_cost_to_visit_all_baths(graph, baths):
    n = len(baths)
    dp = {(visited, current): float('inf') for visited in range(1 << n) for current in range(n)}
    dp[(1, 0)] = 0
    shortest = {i: shortest_path_to_baths(graph, baths[i], baths) for i in range(n)}
    for visited in range(1, 1 << n):
        for current in range(n):
            if not (visited & (1 << current)):
                continue
            for next in range(n):
                if visited & (1 << next):
                    continue
                dp[(visited | (1 << next), next)] = min(dp[(visited | (1 << next), next)], dp[(visited, current)] + shortest[current][baths[next]])
    return min(dp[(1 << n) - 1, i] for i in range(n))

# helper function to help visualize the problem for me. I will comment this out when I submit the code.
# def plot_grid(heights, baths):
#     fig, ax = plt.subplots(figsize=(10, 10),constrained_layout = True)
#     fig.patch.set_facecolor('black')
#     ax.set_facecolor('black')
#     radius = 0.2
#     offset = 0.1  # Offset for the new lines
#     for (x, y), h in heights.items():
#         circle = plt.Circle((y, x), radius, color='white', fill=False)
#         ax.add_artist(circle)
#         ax.text(y, x, str(h), color='white', ha='center', va='center', bbox=dict(facecolor='black', edgecolor='black', boxstyle='round,pad=0.2'), fontsize=12)
#         if (x, y) in baths:
#             color = 'red' if baths.index((x, y)) != 0 else 'blue'  # Change color for the first bathhouse
#             circle = plt.Circle((y, x), radius, color=color, fill=False, linewidth=2)
#             ax.add_artist(circle)
#     for (x, y), h in heights.items():
#         if (x+1, y) in heights:
#             ax.arrow(y-offset, x+radius, 0, 1-2*radius, head_width=0.03, head_length=0.05, fc='w', ec='w')  # New arrow
#             ax.arrow(y+offset, x+1-radius, 0, -1+2*radius, head_width=0.03, head_length=0.05, fc='w', ec='w')  # New arrow
#             weight1 = calculate_weight((x, y), (x+1, y), heights)
#             weight2 = calculate_weight((x+1, y), (x, y), heights)
#             ax.text(y-offset, x+0.5, str(weight1), color='green', ha='center', va='center', bbox=dict(facecolor='black', edgecolor='black', boxstyle='round,pad=0.2'), fontsize=12)  # New text
#             ax.text(y+offset, x+0.5, str(weight2), color='green', ha='center', va='center', bbox=dict(facecolor='black', edgecolor='black', boxstyle='round,pad=0.2'), fontsize=12)  # New text
#         if (x, y+1) in heights:
#             ax.arrow(y+radius, x-offset, 1-2*radius, 0, head_width=0.03, head_length=0.05, fc='w', ec='w')  # New arrow
#             ax.arrow(y+1-radius, x+offset, -1+2*radius, 0, head_width=0.03, head_length=0.05, fc='w', ec='w')  # New arrow
#             weight1 = calculate_weight((x, y), (x, y+1), heights)
#             weight2 = calculate_weight((x, y+1), (x, y), heights)
#             ax.text(y+0.5, x-offset, str(weight1), color='green', ha='center', va='center', bbox=dict(facecolor='black', edgecolor='black', boxstyle='round,pad=0.2'), fontsize=12)  # New text
#             ax.text(y+0.5, x+offset, str(weight2), color='green', ha='center', va='center', bbox=dict(facecolor='black', edgecolor='black', boxstyle='round,pad=0.2'), fontsize=12)  # New text
#     ax.set_aspect(0.5)  
#     plt.gca().invert_yaxis()
#     plt.show()

# helper function to print the graphs
def print_graph(graph):
    for node, edges in graph.items():
        print(f"Node {node}:")
        for edge in edges:
            neighbor, weight = edge
            print(f"  -> Connects to {neighbor} with weight {weight}")
        print()

    
grid_size, heights, baths = read_grid("grid.txt")
graph = create_graph(grid_size, heights)
ans = min_cost_to_visit_all_baths(graph, baths)
# uncomment below, plot_grid, and import matplotlib.pyplot as plt at the top of the file to visualize the problem. (did this just incase you don't have matplotlib installed)
# plot_grid(heights, baths) 
with open("pathLength.txt", "w") as file:
    file.write(str(ans))

