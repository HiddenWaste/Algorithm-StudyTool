import pygame
import sys
import random
from sorting_visualizer import SortingVisualizer  # From our sorting file
import time
from graph_visualizer import GraphVisualizer, Graph  # Import both classes

# Constants
WIDTH = 1000
HEIGHT = 800
WHITE = (255, 255, 255)
NODE_RADIUS = 20  # Add this
MAX_VAL = 500    # Add this
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)

# Setup display
#screen = pygame.display.set_mode((WIDTH, HEIGHT))

class AlgorithmQuiz:
    def __init__(self, sorting_vis, graph_vis):
        self.sorting_vis = sorting_vis
        self.graph_vis = graph_vis
        self.correct_answers = 0
        self.current_type = None
        self.current_answer = None
        self.font = pygame.font.Font(None, 36)
        
        # Algorithm descriptions for theoretical quiz
        self.algorithm_descriptions = {
            'merge': """This algorithm uses a divide-and-conquer strategy. It repeatedly breaks down the array into smaller subarrays until each has only one element, then merges them back together in sorted order.
            
Key characteristics:
- Time complexity: O(n log n)
- Stable sort
- Not in-place""",
            
            'quick': """This algorithm picks a 'pivot' element and partitions other elements into two sub-arrays according to whether they are less than or greater than the pivot.
            
Key characteristics:
- Average case O(n log n)
- Not stable
- In-place sorting""",
            
            'selection': """This algorithm divides the input into a sorted and unsorted region, and repeatedly selects the smallest element from the unsorted region to add to the sorted region.
            
Key characteristics:
- Time complexity: O(n²)
- Not stable
- In-place sorting""",
            
            'insertion': """This algorithm builds the final sorted array one item at a time, by repeatedly inserting a new element into the sorted portion of the array.
            
Key characteristics:
- Time complexity: O(n²)
- Stable sort
- In-place
- Good for small data sets""",
            
            'bubble': """This algorithm repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order.
            
Key characteristics:
- Time complexity: O(n²)
- Stable sort
- In-place sorting""",
            
            'prims': """This algorithm builds a minimum spanning tree by always adding the lowest-weight edge that connects a new vertex to the growing tree.
            
Key characteristics:
- Greedy algorithm
- Good for dense graphs
- Time complexity: O(E log V)""",
            
            'dijkstra': """This algorithm finds the shortest path between nodes in a graph by maintaining a set of unvisited nodes and repeatedly selecting the unvisited node with the smallest tentative distance.
            
Key characteristics:
- Cannot handle negative weights
- Greedy algorithm
- Time complexity: O(E log V)""",
            
            'bellman': """This algorithm computes shortest paths from a source vertex to all other vertices and can handle negative edge weights.
            
Key characteristics:
- Can detect negative cycles
- Dynamic programming approach
- Time complexity: O(VE)""",
            
            'kruskal': """This algorithm finds a minimum spanning tree by considering edges in ascending order of weight and adding them if they don't create a cycle.
            
Key characteristics:
- Uses Union-Find data structure
- Good for sparse graphs
- Time complexity: O(E log E)""",
        }

    def run_theoretical_quiz(self, screen):
        correct_needed = 7
        self.correct_answers = 0
        
        while self.correct_answers < correct_needed:
            # Select random algorithm and its description
            algorithm = random.choice(list(self.algorithm_descriptions.keys()))
            description = self.algorithm_descriptions[algorithm]
            
            # Generate options
            if algorithm in ['merge', 'quick', 'selection', 'insertion', 'bubble']:
                all_options = ['Merge Sort', 'Quick Sort', 'Selection Sort', 'Insertion Sort', 'Bubble Sort']
                correct_index = ['merge', 'quick', 'selection', 'insertion', 'bubble'].index(algorithm)
            else:
                all_options = ["Prim's Algorithm", "Dijkstra's Algorithm", "Bellman-Ford Algorithm", "Kruskal's Algorithm"]
                correct_index = ['prims', 'dijkstra', 'bellman', 'kruskal'].index(algorithm)
            
            correct_option = all_options[correct_index]
            other_options = [opt for opt in all_options if opt != correct_option]
            selected_options = random.sample(other_options, 3)
            options = selected_options + [correct_option]
            random.shuffle(options)
            
            # Display question
            answered = False
            while not answered:
                screen.fill(BLACK)
                
                # Show progress
                progress_text = self.font.render(f"Progress: {self.correct_answers}/{correct_needed}", True, WHITE)
                screen.blit(progress_text, (50, 30))
                
                # Show description (word wrapped)
                y_pos = 80
                words = description.split()
                line = []
                for word in words:
                    line.append(word)
                    text = ' '.join(line)
                    if self.font.size(text)[0] > WIDTH - 100:  # Leave margin
                        line.pop()
                        text = ' '.join(line)
                        text_surface = self.font.render(text, True, WHITE)
                        screen.blit(text_surface, (50, y_pos))
                        y_pos += 30
                        line = [word]
                if line:
                    text = ' '.join(line)
                    text_surface = self.font.render(text, True, WHITE)
                    screen.blit(text_surface, (50, y_pos))
                
                # Show options
                y_pos = HEIGHT - 200
                for i, option in enumerate(options):
                    text = self.font.render(f"{i+1}. {option}", True, WHITE)
                    screen.blit(text, (50, y_pos + i*40))
                
                pygame.display.flip()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            return
                        if event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                            answered = True
                            is_correct = event.key - pygame.K_1 == options.index(correct_option)
                            if is_correct:
                                self.correct_answers += 1
                            
                            # Show result
                            result = "Correct!" if is_correct else f"Incorrect! It was {correct_option}"
                            text = self.font.render(result, True, WHITE)
                            screen.blit(text, (50, HEIGHT - 50))
                            pygame.display.flip()
                            pygame.time.wait(2000)
    
        
    def run_without_info(self, algorithm_type, algorithm_name):
        """Runs the specified algorithm without showing complexity information"""
        original_sort_draw = self.sorting_vis.draw_array
        original_graph_draw = self.graph_vis.draw_graph
        
        try:
            def sort_draw_wrapper(*args, **kwargs):
                self.sorting_vis.screen.fill(BLACK)
                bar_width = WIDTH // len(self.sorting_vis.array)
                for i, val in enumerate(self.sorting_vis.array):
                    color_to_use = args[1] if args[0] and i in args[0] else WHITE
                    height = (val/MAX_VAL) * (HEIGHT-100)
                    pygame.draw.rect(self.sorting_vis.screen, color_to_use,
                                (i*bar_width, HEIGHT-height-50,
                                    bar_width-1, height))
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                
                # Adjust sleep time based on algorithm
                if algorithm_name in ['selection', 'bubble']:
                    time.sleep(0.017)  # About 3x faster (0.05 / 3)
                else:
                    time.sleep(0.025)  # About 2x faster (0.05 / 2)
            
            def graph_draw_wrapper():
                self.graph_vis.screen.fill(BLACK)
                # Draw edges
                for start, edges in self.graph_vis.graph.edges.items():
                    start_pos = self.graph_vis.graph.nodes[start]
                    for end, weight in edges:
                        end_pos = self.graph_vis.graph.nodes[end]
                        color = self.graph_vis.edge_colors.get((min(start, end), max(start, end)), WHITE)
                        pygame.draw.line(self.graph_vis.screen, color, start_pos, end_pos, 2)
                        mid_x = (start_pos[0] + end_pos[0]) // 2
                        mid_y = (start_pos[1] + end_pos[1]) // 2
                        text = self.graph_vis.font.render(str(weight), True, color)
                        self.graph_vis.screen.blit(text, (mid_x, mid_y))
                
                # Draw nodes using the constant
                node_radius = getattr(self.graph_vis, 'NODE_RADIUS', NODE_RADIUS)
                for node, pos in self.graph_vis.graph.nodes.items():
                    color = self.graph_vis.node_colors.get(node, WHITE)
                    pygame.draw.circle(self.graph_vis.screen, color, pos, node_radius)
                    text = self.graph_vis.font.render(str(node), True, BLACK)
                    text_rect = text.get_rect(center=pos)
                    self.graph_vis.screen.blit(text, text_rect)
                
                pygame.display.flip()
                time.sleep(0.25)  # Speed up graph animations a bit too
            
            # Set temporary draw methods
            self.sorting_vis.draw_array = sort_draw_wrapper
            self.graph_vis.draw_graph = graph_draw_wrapper
            
            # Run the algorithm
            if algorithm_type == 'sort':
                if algorithm_name == 'merge':
                    self.sorting_vis.merge_sort()
                elif algorithm_name == 'quick':
                    self.sorting_vis.quicksort()
                elif algorithm_name == 'selection':
                    self.sorting_vis.selection_sort()
                elif algorithm_name == 'insertion':
                    self.sorting_vis.insertion_sort()
                elif algorithm_name == 'bubble':
                    self.sorting_vis.bubble_sort()
            else:
                if algorithm_name == 'prims':
                    self.graph_vis.prims_algorithm()
                elif algorithm_name == 'dijkstra':
                    self.graph_vis.dijkstra_algorithm()
                elif algorithm_name == 'bellman':
                    self.graph_vis.bellman_ford_algorithm()
                elif algorithm_name == 'kruskal':
                    self.graph_vis.kruskal_algorithm()
                    
        finally:
            # Restore original draw methods
            self.sorting_vis.draw_array = original_sort_draw
            self.graph_vis.draw_graph = original_graph_draw
    
    def generate_question(self):
        self.current_type = random.choice(['sort', 'graph'])
        
        if self.current_type == 'sort':
            algorithms = ['merge', 'quick', 'selection', 'insertion', 'bubble']
            self.current_answer = random.choice(algorithms)
            self.sorting_vis.reset_array()
            # Store backup of initial array for replays
            self.backup_array = self.sorting_vis.array.copy()
            self.run_without_info('sort', self.current_answer)
        else:
            algorithms = ['prims', 'dijkstra', 'bellman', 'kruskal']
            self.current_answer = random.choice(algorithms)
            self.graph_vis.graph = Graph()
            self.run_without_info('graph', self.current_answer)

    def generate_options(self):
        if self.current_type == 'sort':
            all_options = ['Merge Sort', 'Quick Sort', 'Selection Sort', 'Insertion Sort', 'Bubble Sort']
            correct_index = ['merge', 'quick', 'selection', 'insertion', 'bubble'].index(self.current_answer)
            correct_option = all_options[correct_index]
        else:
            all_options = ["Prim's Algorithm", "Dijkstra's Algorithm", "Bellman-Ford Algorithm", "Kruskal's Algorithm"]
            correct_index = ['prims', 'dijkstra', 'bellman', 'kruskal'].index(self.current_answer)
            correct_option = all_options[correct_index]
            
        # Remove correct answer and select 3 random wrong answers
        other_options = [opt for opt in all_options if opt != correct_option]
        selected_options = random.sample(other_options, 3)
        # Add correct answer back and shuffle
        options = selected_options + [correct_option]
        random.shuffle(options)
        return options, options.index(correct_option)

class MainProgram:
    def __init__(self):
        pygame.init()
        pygame.mixer.init(44100, -16, 2, 512)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Algorithm Study Tool")
        self.font = pygame.font.Font(None, 48)
        
        # Initialize our visualizers with the screen reference
        self.sorting_vis = SortingVisualizer()
        self.sorting_vis.screen = self.screen  # Add this line
        self.graph_vis = GraphVisualizer()
        self.quiz = AlgorithmQuiz(self.sorting_vis, self.graph_vis)

    def run_visual_quiz(self):
        """Run the visualization-based quiz"""
        correct_needed = 7
        self.quiz.correct_answers = 0
        
        while self.quiz.correct_answers < correct_needed:
            try:
                self.quiz.generate_question()
                options, correct_index = self.quiz.generate_options()
                replays_remaining = 2  # Allow 2 replays
                
                answered = False
                while not answered:
                    self.screen.fill(BLACK)
                    
                    # Show question and replays remaining
                    text = self.font.render(f"Which algorithm was just shown? ({self.quiz.correct_answers}/{correct_needed})", 
                                        True, WHITE)
                    self.screen.blit(text, (50, 50))
                    
                    replay_text = self.font.render(f"Press R to replay ({replays_remaining} replays remaining)", 
                                                True, WHITE if replays_remaining > 0 else GRAY)
                    self.screen.blit(replay_text, (50, 100))
                    
                    # Show options
                    for i, option in enumerate(options):
                        text = self.font.render(f"{i+1}. {option}", True, WHITE)
                        self.screen.blit(text, (50, 150 + i*50))
                    
                    pygame.display.flip()
                    
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_q:
                                return  # Allow quitting during quiz
                            elif event.key == pygame.K_r and replays_remaining > 0:
                                # Replay the algorithm
                                replays_remaining -= 1
                                if self.quiz.current_type == 'sort':
                                    self.sorting_vis.array = self.quiz.backup_array.copy()
                                    self.quiz.run_without_info('sort', self.quiz.current_answer)
                                else:
                                    self.quiz.run_without_info('graph', self.quiz.current_answer)
                            elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                                answered = True
                                is_correct = event.key - pygame.K_1 == correct_index
                                if is_correct:
                                    self.quiz.correct_answers += 1
                                
                                # Show result
                                result = "Correct!" if is_correct else f"Incorrect! It was {options[correct_index]}"
                                text = self.font.render(result, True, WHITE)
                                self.screen.blit(text, (50, 400))
                                pygame.display.flip()
                                pygame.time.wait(2000)
            
            except Exception as e:
                print(f"Error during quiz: {e}")
                pygame.time.wait(1000)
                continue
        
        # Show completion message
        self.screen.fill(BLACK)
        text = self.font.render("Quiz Complete! Press any key to continue", True, WHITE)
        text_rect = text.get_rect(center=(WIDTH//2, HEIGHT//2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    waiting = False
            
    def show_menu(self):
        running = True
        while running:
            self.screen.fill(BLACK)
            texts = [
                "Algorithm Study Tool",
                "",
                "1. Sorting Algorithms",
                "2. Graph Algorithms",
                "3. Algorithm Quiz",
                "",
                "Q. Quit"
            ]
            
            for i, text in enumerate(texts):
                text_surface = self.font.render(text, True, WHITE)
                text_rect = text_surface.get_rect(center=(WIDTH//2, HEIGHT//4 + i*50))
                self.screen.blit(text_surface, text_rect)
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return False
                    elif event.key == pygame.K_1:
                        self.run_sorting()
                    elif event.key == pygame.K_2:
                        self.run_graphs()
                    elif event.key == pygame.K_3:
                        self.run_quiz()
            
        return True
    
    def run_sorting(self):
        running = True
        while running:
            self.screen.fill(BLACK)
            texts = [
                "Select a sorting algorithm:",
                "1. Merge Sort",
                "2. Quick Sort",
                "3. Selection Sort",
                "4. Insertion Sort",
                "5. Bubble Sort",
                "Press Q to return to main menu"
            ]
            
            for i, text in enumerate(texts):
                text_surface = self.font.render(text, True, WHITE)
                self.screen.blit(text_surface, (WIDTH//4, HEIGHT//4 + i*40))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_1:
                        self.sorting_vis.reset_array()
                        self.sorting_vis.merge_sort()
                        self.sorting_vis.victory_sound()
                        self.sorting_vis.wait_for_user()
                    elif event.key == pygame.K_2:
                        self.sorting_vis.reset_array()
                        self.sorting_vis.quicksort()
                        self.sorting_vis.victory_sound()
                        self.sorting_vis.wait_for_user()
                    elif event.key == pygame.K_3:
                        self.sorting_vis.reset_array()
                        self.sorting_vis.selection_sort()
                        self.sorting_vis.victory_sound()
                        self.sorting_vis.wait_for_user()
                    elif event.key == pygame.K_4:
                        self.sorting_vis.reset_array()
                        self.sorting_vis.insertion_sort()
                        self.sorting_vis.victory_sound()
                        self.sorting_vis.wait_for_user()
                    elif event.key == pygame.K_5:
                        self.sorting_vis.reset_array()
                        self.sorting_vis.bubble_sort()
                        self.sorting_vis.victory_sound()
                        self.sorting_vis.wait_for_user()
    
    def run_graphs(self):
        running = True
        while running:
            self.screen.fill(BLACK)
            texts = [
                "Select a graph algorithm:",
                "1. Prim's Algorithm",
                "2. Dijkstra's Algorithm",
                "3. Bellman-Ford Algorithm",
                "4. Kruskal's Algorithm",
                "Press Q to return to main menu"
            ]
            
            for i, text in enumerate(texts):
                text_surface = self.font.render(text, True, WHITE)
                self.screen.blit(text_surface, (WIDTH//4, HEIGHT//4 + i*40))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_1:
                        # Create a new graph directly
                        self.graph_vis.graph = Graph()  # Use Graph directly, not as a method
                        self.graph_vis.prims_algorithm()
                        self.graph_vis.wait_for_user()
                    elif event.key == pygame.K_2:
                        self.graph_vis.graph = Graph()
                        self.graph_vis.dijkstra_algorithm()
                        self.graph_vis.wait_for_user()
                    elif event.key == pygame.K_3:
                        self.graph_vis.graph = Graph()
                        self.graph_vis.bellman_ford_algorithm()
                        self.graph_vis.wait_for_user()
                    elif event.key == pygame.K_4:
                        self.graph_vis.graph = Graph()
                        self.graph_vis.kruskal_algorithm()
                        self.graph_vis.wait_for_user()

    def run_quiz(self):
        running = True
        while running:
            self.screen.fill(BLACK)
            texts = [
                "Select Quiz Type:",
                "1. Visual Recognition Quiz",
                "2. Theoretical Knowledge Quiz",
                "",
                "Press Q to return to main menu"
            ]
            
            for i, text in enumerate(texts):
                text_surface = self.font.render(text, True, WHITE)
                text_rect = text_surface.get_rect(center=(WIDTH//2, HEIGHT//4 + i*50))
                self.screen.blit(text_surface, text_rect)
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_1:
                        self.run_visual_quiz()  # Previous quiz implementation
                    elif event.key == pygame.K_2:
                        self.quiz.run_theoretical_quiz(self.screen)

    def run(self):
        running = True
        while running:
            running = self.show_menu()
        pygame.quit()

if __name__ == "__main__":
    program = MainProgram()
    program.run()