import pygame
import random
import time
import sys
import numpy as np

# Initialize Pygame and its mixer for sound
pygame.init()
pygame.mixer.init(44100, -16, 2, 512)

# Constants
WIDTH = 800
HEIGHT = 600
ARRAY_SIZE = 50
MAX_VAL = 500

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (147, 0, 211)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)

# Algorithm Information
ALGORITHM_INFO = {
    "merge": {
        "name": "Merge Sort",
        "description": """Merge Sort is a divide-and-conquer algorithm that recursively breaks down the array into smaller subarrays until each has only one element, then merges them back together in sorted order.

Key characteristics:
- Stable sort
- Divide-and-conquer approach
- Predictable performance
- Space complexity O(n)""",
        "upper": "O(n log n)",
        "lower": "Ω(n log n)",
        "tight": "Θ(n log n)"
    },
    "quick": {
        "name": "Quick Sort",
        "description": """Quick Sort works by selecting a 'pivot' element and partitioning the array around it, such that smaller elements are on the left and larger ones on the right.

Key characteristics:
- In-place sorting
- Usually faster in practice
- Unstable sort
- Partition-based strategy""",
        "upper": "O(n²)",
        "lower": "Ω(n log n)",
        "tight": "Θ(n log n)*"
    },
    "selection": {
        "name": "Selection Sort",
        "description": """Selection Sort works by repeatedly finding the minimum element from the unsorted portion and placing it at the beginning.

Key characteristics:
- Simple implementation
- In-place sorting
- Quadratic complexity
- Minimal memory usage""",
        "upper": "O(n²)",
        "lower": "Ω(n²)",
        "tight": "Θ(n²)"
    },
    "insertion": {
        "name": "Insertion Sort",
        "description": """Insertion Sort builds the final sorted array one item at a time, by repeatedly inserting a new element into the sorted portion of the array.

Key characteristics:
- Adaptive algorithm
- Stable sort
- In-place sorting
- Efficient for small data""",
        "upper": "O(n²)",
        "lower": "Ω(n)",
        "tight": "Θ(n²)"
    },
    "bubble": {
        "name": "Bubble Sort",
        "description": """Bubble Sort repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order.

Key characteristics:
- Simple implementation
- Stable sort
- In-place sorting
- Early exit possible""",
        "upper": "O(n²)",
        "lower": "Ω(n)",
        "tight": "Θ(n²)"
    }
}

# Setup display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interactive Sorting Algorithm Visualization")

class SortingVisualizer:
    def __init__(self):
        self.reset_array()
        self.bar_width = WIDTH // ARRAY_SIZE
        self.current_sort = None
        self.font = pygame.font.Font(None, 24)
        self.path_progression = []
        self.execution_time = 0
        self.start_time = 0
        self.comparisons = 0
        self.swaps = 0
        self.screen = None  # Will be set by MainProgram
        
    def reset_array(self):
        self.array = [random.randint(1, MAX_VAL) for _ in range(ARRAY_SIZE)]
        self.backup_array = self.array.copy()

    def restore_array(self):
        self.array = self.backup_array.copy()
        self.comparisons = 0
        self.swaps = 0

    def show_info_screen(self):
        if not self.current_sort:
            return
            
        info = ALGORITHM_INFO[self.current_sort]
        running = True
        
        while running:
            screen.fill(BLACK)
            
            # Draw title
            title_font = pygame.font.Font(None, 48)
            title = title_font.render(info["name"], True, WHITE)
            screen.blit(title, (50, 50))
            
            # Draw description
            desc_font = pygame.font.Font(None, 24)
            lines = info["description"].split('\n')
            y = 120
            for line in lines:
                text = desc_font.render(line, True, WHITE)
                screen.blit(text, (50, y))
                y += 30
            
            # Draw performance metrics
            y += 30
            metrics = [
                f"Array Size: {len(self.array)} elements",
                f"Execution Time: {self.execution_time:.3f} seconds",
                f"Comparisons: {self.comparisons}",
                f"Swaps: {self.swaps}",
                f"Time Complexity: {info['upper']}",
                f"Space Complexity: O(1) auxiliary"
            ]
            
            for metric in metrics:
                text = desc_font.render(metric, True, GREEN)
                screen.blit(text, (50, y))
                y += 25

            # Draw instructions
            instructions = desc_font.render("Press SPACE to return to visualization", True, WHITE)
            screen.blit(instructions, (50, HEIGHT - 50))
            
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

    def draw_complexity_box(self):
        if self.current_sort and self.current_sort in ALGORITHM_INFO:
            info = ALGORITHM_INFO[self.current_sort]
            
            # Draw semi-transparent background
            complexity_surface = pygame.Surface((250, 100))
            complexity_surface.fill(BLACK)
            complexity_surface.set_alpha(200)
            screen.blit(complexity_surface, (10, 10))
            
            # Draw text
            y_offset = 15
            titles = [
                f"{info['name']} Complexity:",
                f"Upper Bound: {info['upper']}",
                f"Lower Bound: {info['lower']}",
                f"Tight Bound: {info['tight']}"
            ]
            
            for title in titles:
                text_surface = self.font.render(title, True, WHITE)
                screen.blit(text_surface, (20, y_offset))
                y_offset += 25
                
            if self.current_sort == "quick":
                note = self.font.render("*Average case", True, GRAY)
                screen.blit(note, (20, y_offset))

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
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                        self.restore_array()
                        if event.key == pygame.K_1:
                            self.merge_sort()
                        elif event.key == pygame.K_2:
                            self.quicksort()
                        elif event.key == pygame.K_3:
                            self.selection_sort()
                        elif event.key == pygame.K_4:
                            self.insertion_sort()
                        elif event.key == pygame.K_5:
                            self.bubble_sort()
            
            # Draw instruction text
            text = self.font.render("Press SPACE to continue, I for algorithm info, or 1-5 to run another algorithm", True, WHITE)
            text_rect = text.get_rect(center=(WIDTH//2, HEIGHT - 30))
            # Clear the bottom area
            pygame.draw.rect(screen, BLACK, (0, HEIGHT - 50, WIDTH, 50))
            screen.blit(text, text_rect)
            pygame.display.flip()

    def play_sound(self, value):
        try:
            # Generate a frequency based on the value
            frequency = 200 + (value / MAX_VAL) * 500
            duration = 50  # milliseconds
            sample_rate = 44100
            samples = int(duration * sample_rate / 1000.0)
            
            # Generate time array
            t = np.linspace(0, duration/1000.0, samples, False)
            
            # Create the sine wave
            wave = np.sin(2.0 * np.pi * frequency * t)
            
            # Apply an envelope to smooth the sound
            attack = int(samples * 0.1)  # 10% attack
            decay = int(samples * 0.2)   # 20% decay
            
            # Create envelope
            envelope = np.ones(samples)
            # Attack (fade in)
            envelope[:attack] = np.linspace(0, 1, attack)
            # Decay (fade out)
            envelope[-decay:] = np.linspace(1, 0, decay)
            
            # Apply envelope to wave
            wave = wave * envelope
            
            # Add a small amount of the previous and next harmonic for richness
            wave += 0.3 * np.sin(2.0 * np.pi * (frequency/2) * t) * envelope  # Sub-harmonic
            wave += 0.2 * np.sin(2.0 * np.pi * (frequency*2) * t) * envelope  # Harmonic
            
            # Normalize
            wave = wave / np.max(np.abs(wave))
            
            # Convert to 16-bit integers
            wave = np.int16(wave * 32767 * 0.5)  # Reduced volume to 50%
            
            # Make it stereo
            stereo = np.column_stack((wave, wave))
            stereo = np.ascontiguousarray(stereo)
            
            # Create and play the sound
            sound = pygame.sndarray.make_sound(stereo)
            sound.play()
            
        except Exception as e:
            pass  # Fail silently if sound generation fails
    
    def victory_sound(self):
        for val in self.array:
            self.draw_array([self.array.index(val)], GREEN)
            self.play_sound(val)
            time.sleep(0.02)

    def draw_array(self, highlight_positions=None, color=RED):
            if self.screen is None:
                return
                
            self.screen.fill(BLACK)
            for i, val in enumerate(self.array):
                color_to_use = color if highlight_positions and i in highlight_positions else WHITE
                height = (val/MAX_VAL) * (HEIGHT-20)
                pygame.draw.rect(self.screen, color_to_use, 
                            (i*self.bar_width, HEIGHT-height, 
                                self.bar_width-1, height))
            
            pygame.display.update()

    def merge_sort(self, left=0, right=None):
        self.current_sort = "merge"
        if right is None:
            right = len(self.array) - 1
            self.start_time = time.time()
            
        if left < right:
            mid = (left + right) // 2
            self.merge_sort(left, mid)
            self.merge_sort(mid + 1, right)
            self.merge(left, mid, right)
            
        if right == len(self.array) - 1 and left == 0:
            self.execution_time = time.time() - self.start_time
    
    def merge(self, left, mid, right):
        left_part = self.array[left:mid + 1]
        right_part = self.array[mid + 1:right + 1]
        
        i = j = 0
        k = left
        
        while i < len(left_part) and j < len(right_part):
            self.draw_array([k], BLUE)
            self.play_sound(self.array[k])
            
            self.comparisons += 1
            if left_part[i] <= right_part[j]:
                self.array[k] = left_part[i]
                self.swaps += 1
                i += 1
            else:
                self.array[k] = right_part[j]
                self.swaps += 1
                j += 1
            k += 1
        
        while i < len(left_part):
            self.draw_array([k], BLUE)
            self.play_sound(self.array[k])
            self.array[k] = left_part[i]
            self.swaps += 1
            i += 1
            k += 1
        
        while j < len(right_part):
            self.draw_array([k], BLUE)
            self.play_sound(self.array[k])
            self.array[k] = right_part[j]
            self.swaps += 1
            j += 1
            k += 1

    def quicksort(self, low=0, high=None):
        self.current_sort = "quick"
        if high is None:
            high = len(self.array) - 1
            self.start_time = time.time()
            
        if low < high:
            pi = self.partition(low, high)
            self.quicksort(low, pi - 1)
            self.quicksort(pi + 1, high)
            
        if high == len(self.array) - 1 and low == 0:
            self.execution_time = time.time() - self.start_time
    
    def partition(self, low, high):
        pivot = self.array[high]
        i = low - 1
        
        for j in range(low, high):
            self.draw_array([j, high], GREEN)
            self.play_sound(self.array[j])
            
            self.comparisons += 1
            if self.array[j] <= pivot:
                i += 1
                self.array[i], self.array[j] = self.array[j], self.array[i]
                self.swaps += 1
        
        self.array[i + 1], self.array[high] = self.array[high], self.array[i + 1]
        self.swaps += 1
        return i + 1
    
    def selection_sort(self):
        self.current_sort = "selection"
        self.start_time = time.time()
        
        for i in range(len(self.array)):
            min_idx = i
            for j in range(i + 1, len(self.array)):
                self.draw_array([min_idx, j], PURPLE)
                self.play_sound(self.array[j])
                self.comparisons += 1
                if self.array[j] < self.array[min_idx]:
                    min_idx = j
            
            if min_idx != i:
                self.array[i], self.array[min_idx] = self.array[min_idx], self.array[i]
                self.swaps += 1
        
        self.execution_time = time.time() - self.start_time
    
    def insertion_sort(self):
        self.current_sort = "insertion"
        self.start_time = time.time()
        
        for i in range(1, len(self.array)):
            key = self.array[i]
            j = i - 1
            while j >= 0:
                self.comparisons += 1
                if self.array[j] > key:
                    self.draw_array([j, j+1], RED)
                    self.play_sound(self.array[j])
                    self.array[j + 1] = self.array[j]
                    self.swaps += 1
                    j -= 1
                else:
                    break
            self.array[j + 1] = key
            self.swaps += 1
        
        self.execution_time = time.time() - self.start_time

    def bubble_sort(self):
        self.current_sort = "bubble"
        self.start_time = time.time()
        
        n = len(self.array)
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                self.draw_array([j, j + 1], BLUE)
                self.play_sound(self.array[j])
                
                self.comparisons += 1
                if self.array[j] > self.array[j + 1]:
                    self.array[j], self.array[j + 1] = self.array[j + 1], self.array[j]
                    self.swaps += 1
                    swapped = True
            
            if not swapped:
                break
        
        self.execution_time = time.time() - self.start_time

def main():
    visualizer = SortingVisualizer()
    running = True
    
    while running:
        screen.fill(BLACK)
        font = pygame.font.Font(None, 36)
        texts = [
            "Select a sorting algorithm:",
            "1. Merge Sort",
            "2. Quick Sort",
            "3. Selection Sort",
            "4. Insertion Sort",
            "5. Bubble Sort",
            "Press Q to quit"
        ]
        
        for i, text in enumerate(texts):
            text_surface = font.render(text, True, WHITE)
            screen.blit(text_surface, (WIDTH//4, HEIGHT//4 + i*40))
        
        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                    visualizer.reset_array()
                    if event.key == pygame.K_1:
                        visualizer.merge_sort()
                    elif event.key == pygame.K_2:
                        visualizer.quicksort()
                    elif event.key == pygame.K_3:
                        visualizer.selection_sort()
                    elif event.key == pygame.K_4:
                        visualizer.insertion_sort()
                    elif event.key == pygame.K_5:
                        visualizer.bubble_sort()
                    visualizer.victory_sound()
                    visualizer.wait_for_user()
    
    pygame.quit()

if __name__ == "__main__":
    main()