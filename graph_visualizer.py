import pygame
import random
import numpy as np
import math
from collections import defaultdict
import heapq
import time
import sys

# Initialize Pygame
pygame.init()
pygame.mixer.init(44100, -16, 2, 512)

# Constants
WIDTH = 1000
HEIGHT = 800
NODE_RADIUS = 20
NUM_NODES = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

# Algorithm Information
ALGORITHM_INFO = {
    "prims": {
        "name": "Prim's Algorithm",
        "description": """Prim's algorithm builds a minimum spanning tree by iteratively selecting the lowest-weight edge that connects a new vertex to the growing tree. It starts from an arbitrary vertex and grows the tree one vertex at a time.

Key characteristics:
- Greedy algorithm
- Handles only positive weights
- Efficient for dense graphs""",
        "time": "O(E log V)",
        "space": "O(V)",
        "use": "Minimum Spanning Tree"
    },
    "dijkstra": {
        "name": "Dijkstra's Algorithm",
        "description": """Dijkstra's algorithm finds the shortest path from a source vertex to all other vertices. It maintains a set of unvisited vertices and repeatedly selects the unvisited vertex with the smallest tentative distance.

Key characteristics:
- Greedy algorithm
- Cannot handle negative weights
- Optimal for shortest paths""",
        "time": "O(E log V)",
        "space": "O(V)",
        "use": "Single-Source Shortest Path"
    },
    "bellman": {
        "name": "Bellman-Ford Algorithm",
        "description": """The Bellman-Ford algorithm finds shortest paths from a source vertex to all other vertices, even with negative edge weights. It relaxes all edges V-1 times and can detect negative cycles.

Key characteristics:
- Dynamic programming approach
- Handles negative weights
- Can detect negative cycles""",
        "time": "O(VE)",
        "space": "O(V)",
        "use": "Shortest Path (handles negative weights)"
    },
    "kruskal": {
        "name": "Kruskal's Algorithm",
        "description": """Kruskal's algorithm builds a minimum spanning tree by considering edges in ascending order of weight. It uses a disjoint set data structure to detect cycles.

Key characteristics:
- Greedy algorithm
- Efficient for sparse graphs
- Uses Union-Find structure""",
        "time": "O(E log E)",
        "space": "O(V)",
        "use": "Minimum Spanning Tree"
    }
}

class Graph:
    def __init__(self):
        self.nodes = {}  # position of nodes
        self.edges = defaultdict(list)  # adjacency list with weights
        self.generate_random_graph()
        
    def generate_random_graph(self):
        # Generate random node positions
        self.nodes = {}
        margin = NODE_RADIUS * 3
        for i in range(NUM_NODES):
            while True:
                x = random.randint(margin, WIDTH - margin)
                y = random.randint(margin, HEIGHT - margin)
                # Check if position is far enough from other nodes
                valid = True
                for pos in self.nodes.values():
                    if math.dist((x, y), pos) < NODE_RADIUS * 4:
                        valid = False
                        break
                if valid:
                    self.nodes[i] = (x, y)
                    break
        
        # Generate random edges
        self.edges = defaultdict(list)
        # Ensure graph is connected
        nodes_list = list(range(NUM_NODES))
        random.shuffle(nodes_list)
        for i in range(NUM_NODES - 1):
            weight = random.randint(1, 10)
            self.edges[nodes_list[i]].append((nodes_list[i + 1], weight))
            self.edges[nodes_list[i + 1]].append((nodes_list[i], weight))
        
        # Add some random extra edges
        for _ in range(NUM_NODES * 2):
            a = random.randint(0, NUM_NODES - 1)
            b = random.randint(0, NUM_NODES - 1)
            if a != b and not any(edge[0] == b for edge in self.edges[a]):
                weight = random.randint(1, 10)
                self.edges[a].append((b, weight))
                self.edges[b].append((a, weight))

class GraphVisualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Graph Algorithm Visualization")
        self.font = pygame.font.Font(None, 24)
        self.graph = Graph()
        self.current_algorithm = None
        self.node_colors = {}
        self.edge_colors = {}
        self.path_progression = []
        self.execution_time = 0
        self.start_time = 0
    
    def play_sound(self, weight):
        try:
            frequency = 200 + weight * 50
            duration = 50
            sample_rate = 44100
            samples = int(duration * sample_rate / 1000.0)
            
            t = np.linspace(0, duration/1000.0, samples, False)
            wave = np.sin(2.0 * np.pi * frequency * t)
            
            attack = int(samples * 0.1)
            decay = int(samples * 0.2)
            envelope = np.ones(samples)
            envelope[:attack] = np.linspace(0, 1, attack)
            envelope[-decay:] = np.linspace(1, 0, decay)
            wave = wave * envelope
            
            wave += 0.3 * np.sin(2.0 * np.pi * (frequency/2) * t) * envelope
            wave += 0.2 * np.sin(2.0 * np.pi * (frequency*2) * t) * envelope
            
            wave = wave / np.max(np.abs(wave))
            wave = np.int16(wave * 32767 * 0.5)
            stereo = np.column_stack((wave, wave))
            stereo = np.ascontiguousarray(stereo)
            
            sound = pygame.sndarray.make_sound(stereo)
            sound.play()
        except Exception as e:
            pass

    def show_info_screen(self):
        if not self.current_algorithm:
            return
            
        info = ALGORITHM_INFO[self.current_algorithm]
        running = True
        
        while running:
            self.screen.fill(BLACK)
            
            # Draw title
            title_font = pygame.font.Font(None, 48)
            title = title_font.render(info["name"], True, WHITE)
            self.screen.blit(title, (50, 50))
            
            # Draw description
            desc_font = pygame.font.Font(None, 24)
            lines = info["description"].split('\n')
            y = 120
            for line in lines:
                text = desc_font.render(line, True, WHITE)
                self.screen.blit(text, (50, y))
                y += 30
            
            # Draw progression
            y += 30
            progression_title = desc_font.render("Path Progression:", True, YELLOW)
            self.screen.blit(progression_title, (50, y))
            y += 30
            for step in self.path_progression[-10:]:  # Show last 10 steps
                text = desc_font.render(step, True, WHITE)
                self.screen.blit(text, (50, y))
                y += 25
            
            # Draw performance metrics
            y += 30
            metrics = [
                f"Input Size: {len(self.graph.nodes)} nodes, {sum(len(edges) for edges in self.graph.edges.values())//2} edges",
                f"Execution Time: {self.execution_time:.3f} seconds",
                f"Time Complexity: {info['time']}",
                f"Space Complexity: {info['space']}"
            ]
            
            for metric in metrics:
                text = desc_font.render(metric, True, GREEN)
                self.screen.blit(text, (50, y))
                y += 25
            
            # Draw instructions
            instructions = desc_font.render("Press SPACE to return to visualization", True, WHITE)
            self.screen.blit(instructions, (50, HEIGHT - 50))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        running = False
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()

    def draw_graph(self):
        self.screen.fill(BLACK)
        
        # Draw edges
        for start, edges in self.graph.edges.items():
            start_pos = self.graph.nodes[start]
            for end, weight in edges:
                end_pos = self.graph.nodes[end]
                color = self.edge_colors.get((min(start, end), max(start, end)), WHITE)
                pygame.draw.line(self.screen, color, start_pos, end_pos, 2)
                
                # Draw weight
                mid_x = (start_pos[0] + end_pos[0]) // 2
                mid_y = (start_pos[1] + end_pos[1]) // 2
                text = self.font.render(str(weight), True, color)
                self.screen.blit(text, (mid_x, mid_y))
        
        # Draw nodes
        for node, pos in self.graph.nodes.items():
            color = self.node_colors.get(node, WHITE)
            pygame.draw.circle(self.screen, color, pos, NODE_RADIUS)
            text = self.font.render(str(node), True, BLACK)
            text_rect = text.get_rect(center=pos)
            self.screen.blit(text, text_rect)
        
        pygame.display.flip()

    def wait_for_user(self):
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_i:
                        self.show_info_screen()
                    elif event.key == pygame.K_SPACE:
                        waiting = False
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                    # Add number key handling for direct algorithm switching
                    elif event.key == pygame.K_1:
                        self.node_colors = {}
                        self.edge_colors = {}
                        self.prims_algorithm()
                    elif event.key == pygame.K_2:
                        self.node_colors = {}
                        self.edge_colors = {}
                        self.dijkstra_algorithm()
                    elif event.key == pygame.K_3:
                        self.node_colors = {}
                        self.edge_colors = {}
                        self.bellman_ford_algorithm()
                    elif event.key == pygame.K_4:
                        self.node_colors = {}
                        self.edge_colors = {}
                        self.kruskal_algorithm()
            
            # Update the instruction text to include algorithm switching option
            text = self.font.render("Press SPACE to continue, I for algorithm info, or 1-4 to run another algorithm", True, WHITE)
            text_rect = text.get_rect(center=(WIDTH//2, HEIGHT - 30))
            # Clear the bottom area before drawing new text
            pygame.draw.rect(self.screen, BLACK, (0, HEIGHT - 50, WIDTH, 50))
            self.screen.blit(text, text_rect)
            pygame.display.flip()

    def prims_algorithm(self, start=0):
        self.current_algorithm = "prims"
        self.node_colors = {start: GREEN}
        self.edge_colors = {}
        self.path_progression = []
        self.start_time = time.time()
        
        visited = {start}
        edges = []
        for neighbor, weight in self.graph.edges[start]:
            heapq.heappush(edges, (weight, start, neighbor))
        
        while edges and len(visited) < len(self.graph.nodes):
            weight, u, v = heapq.heappop(edges)
            
            if v in visited:
                continue
                
            visited.add(v)
            self.node_colors[v] = GREEN
            self.edge_colors[(min(u, v), max(u, v))] = GREEN
            self.path_progression.append(f"Added edge {u}-{v} with weight {weight}")
            self.play_sound(weight)
            self.draw_graph()
            pygame.time.wait(500)
            
            for neighbor, w in self.graph.edges[v]:
                if neighbor not in visited:
                    heapq.heappush(edges, (w, v, neighbor))
        
        self.execution_time = time.time() - self.start_time
        self.wait_for_user()

    def dijkstra_algorithm(self, start=0):
        self.current_algorithm = "dijkstra"
        self.node_colors = {start: BLUE}
        self.edge_colors = {}
        self.path_progression = []
        self.start_time = time.time()
        
        distances = {node: float('inf') for node in self.graph.nodes}
        distances[start] = 0
        pq = [(0, start)]
        visited = set()
        
        while pq:
            dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            self.node_colors[current] = BLUE
            self.path_progression.append(f"Visited node {current} with distance {dist}")
            
            for neighbor, weight in self.graph.edges[current]:
                if neighbor not in visited:
                    new_dist = dist + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor))
                        self.edge_colors[(min(current, neighbor), max(current, neighbor))] = BLUE
                        self.play_sound(weight)
                        self.draw_graph()
                        pygame.time.wait(500)
        
        self.execution_time = time.time() - self.start_time
        self.wait_for_user()

    def bellman_ford_algorithm(self, start=0):
        self.current_algorithm = "bellman"
        self.node_colors = {start: YELLOW}
        self.edge_colors = {}
        self.path_progression = []
        self.start_time = time.time()
        
        distances = {node: float('inf') for node in self.graph.nodes}
        distances[start] = 0
        
        for i in range(len(self.graph.nodes) - 1):
            self.path_progression.append(f"Iteration {i+1}")
            for u in self.graph.nodes:
                for v, weight in self.graph.edges[u]:
                    if distances[u] + weight < distances[v]:
                        distances[v] = distances[u] + weight
                        self.node_colors[v] = YELLOW
                        self.edge_colors[(min(u, v), max(u, v))] = YELLOW
                        self.path_progression.append(f"Updated distance to node {v}: {distances[v]}")
                        self.play_sound(weight)
                        self.draw_graph()
                        pygame.time.wait(500)
        
        self.execution_time = time.time() - self.start_time
        self.wait_for_user()

    def kruskal_algorithm(self):
        self.current_algorithm = "kruskal"
        self.node_colors = {}
        self.edge_colors = {}
        self.path_progression = []
        self.start_time = time.time()
        
        # Create edge list and sort by weight
        edges = []
        seen_edges = set()
        for u in self.graph.nodes:
            for v, weight in self.graph.edges[u]:
                if (min(u, v), max(u, v)) not in seen_edges:
                    edges.append((weight, u, v))
                    seen_edges.add((min(u, v), max(u, v)))
        edges.sort()
        
        # Initialize disjoint set
        parent = {node: node for node in self.graph.nodes}
        
        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]
        
        def union(u, v):
            parent[find(u)] = find(v)
        
        # Process edges
        for weight, u, v in edges:
            if find(u) != find(v):
                union(u, v)
                self.node_colors[u] = RED
                self.node_colors[v] = RED
                self.edge_colors[(min(u, v), max(u, v))] = RED
                self.path_progression.append(f"Added edge {u}-{v} with weight {weight}")
                self.play_sound(weight)
                self.draw_graph()
                pygame.time.wait(500)
        
        self.execution_time = time.time() - self.start_time
        self.wait_for_user()

def main():
    visualizer = GraphVisualizer()
    running = True
    
    while running:
        visualizer.screen.fill(BLACK)
        menu_font = pygame.font.Font(None, 36)
        texts = [
            "Select an algorithm:",
            "1. Prim's Algorithm",
            "2. Dijkstra's Algorithm",
            "3. Bellman-Ford Algorithm",
            "4. Kruskal's Algorithm",
            "Q. Quit"
        ]
        
        for i, text in enumerate(texts):
            text_surface = menu_font.render(text, True, WHITE)
            visualizer.screen.blit(text_surface, (WIDTH//4, HEIGHT//4 + i*40))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_1:
                    visualizer.graph = Graph()
                    visualizer.prims_algorithm()
                elif event.key == pygame.K_2:
                    visualizer.graph = Graph()
                    visualizer.dijkstra_algorithm()
                elif event.key == pygame.K_3:
                    visualizer.graph = Graph()
                    visualizer.bellman_ford_algorithm()
                elif event.key == pygame.K_4:
                    visualizer.graph = Graph()
                    visualizer.kruskal_algorithm()
    
    pygame.quit()

if __name__ == "__main__":
    main()