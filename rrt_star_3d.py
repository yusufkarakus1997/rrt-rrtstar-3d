import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.widgets import Button, Slider
import time
import sys
from collections import deque
import argparse
import csv
import os
import statistics

# =============================================================================
# PARAMETRELER (İstediğiniz gibi değiştirebilirsiniz)
# =============================================================================

MAP_SIZE = np.array([100.0, 100.0, 100.0])  # 3D Çalışma alanının boyutları (X, Y, Z)

START = np.array([5.0, 5.0, 5.0])           # Başlangıç noktası koordinatları
GOAL = np.array([90.0, 90.0, 90.0])         # Hedef noktası koordinatları

MAX_ITERATIONS = 3000                       # Maksimum iterasyon sayısı (Düğüm sayısı)
STEP_SIZE = 5.0                             # Ağacın her adımda uzama miktarı
GOAL_SAMPLE_RATE = 0.05                     # Hedefe doğru rastgele nokta üretme olasılığı (%5)
GOAL_THRESHOLD = 5.0                        # Hedefe varıldı sayılması için gereken maksimum mesafe

SEARCH_RADIUS_MAX = 20.0                    # RRT* için yakındaki düğümleri arama maksimum yarıçapı
RRT_STAR_GAMMA = 45.0                       # RRT* Adaptive Radius için çarpan (Dinamik yarıçap)
USE_ADAPTIVE_RADIUS = True                  # RRT* için dinamik yarıçap kullanılsın mı?

COLLISION_CHECK_STEP = 0.5                  # Çarpışma testi için çizgi üzerindeki örnekleme aralığı
ROBOT_RADIUS = 1.0                          # Robotun fiziksel yarıçapı
SAFETY_MARGIN = 0.5                         # Engellerle aradaki güvenlik mesafesi

RANDOM_SEED = 42                            # Adil karşılaştırma için rastgelelik tohumu
COST_EPSILON = 1e-9                         # Rewiring noise prevention
NODE_EPSILON = 1e-9                         # Duplicate and zero-length edge prevention

ANIMATION_INTERVAL_MS = 10                  # Animasyon yenileme hızı (ms)
DRAW_EVERY_N_ITERATIONS = 5                 # Performans modu: Kaç iterasyonda bir ekran güncellenecek?

DEMO_SEED_START = 1                         # Tarama başlangıç seed'i
DEMO_SEED_END = 200                         # Tarama bitiş seed'i
DEMO_MIN_IMPROVEMENT_PERCENT = 5.0          # Eğitim amaçlı kabul edilebilir minimum iyileşme (%)
DEMO_MIN_COST_DROPS = 2                     # En az kaç kez maliyet düşüş eventi yaşanmalı

BENCHMARK_SEED_START = 1
BENCHMARK_SEEDS = 200

# Engeller: (x, y, z, genişlik, derinlik, yükseklik)
OBSTACLES = [
    (20, 20, 0, 20, 20, 60),
    (60, 60, 40, 20, 20, 60),
    (20, 60, 20, 60, 10, 20),
    (60, 20, 20, 10, 60, 20),
    (40, 40, 40, 20, 20, 20),
    (10, 80, 0, 15, 15, 80),
    (80, 10, 0, 15, 15, 80),
    (40, 10, 60, 20, 30, 15),
]

# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def validate_start_goal(obstacles):
    margin = ROBOT_RADIUS + SAFETY_MARGIN
    for p, name in [(START, "START"), (GOAL, "GOAL")]:
        if not (0 <= p[0] <= MAP_SIZE[0] and 0 <= p[1] <= MAP_SIZE[1] and 0 <= p[2] <= MAP_SIZE[2]):
            raise ValueError(f"{name} noktası {p} workspace sınırları dışındadır!")
        for (x, y, z, w, d, h) in obstacles:
            if (x - margin <= p[0] <= x + w + margin and 
                y - margin <= p[1] <= y + d + margin and 
                z - margin <= p[2] <= z + h + margin):
                raise ValueError(f"{name} noktası {p} bir engelin veya güvenlik marjının içindedir!")

def is_inside_workspace(p):
    return (0 <= p[0] <= MAP_SIZE[0] and 
            0 <= p[1] <= MAP_SIZE[1] and 
            0 <= p[2] <= MAP_SIZE[2])

def is_collision_free(p1, p2, obstacles, step_size=COLLISION_CHECK_STEP):
    if step_size <= 0:
        step_size = 0.5
    if not is_inside_workspace(p1) or not is_inside_workspace(p2):
        return False

    dist = np.linalg.norm(p2 - p1)
    if dist == 0:
        return True
    
    direction = (p2 - p1) / dist
    steps = int(np.ceil(dist / step_size))
    
    margin = ROBOT_RADIUS + SAFETY_MARGIN
    
    for i in range(steps + 1):
        p = p1 + i * step_size * direction if i < steps else p2
        for (x, y, z, w, d, h) in obstacles:
            if (x - margin <= p[0] <= x + w + margin and 
                y - margin <= p[1] <= y + d + margin and 
                z - margin <= p[2] <= z + h + margin):
                return False
    return True

def generate_random_points(seed=RANDOM_SEED):
    sample_rng = np.random.default_rng(seed)
    points = []
    for _ in range(MAX_ITERATIONS):
        if sample_rng.random() < GOAL_SAMPLE_RATE:
            points.append(GOAL)
        else:
            points.append(np.array([
                sample_rng.uniform(0, MAP_SIZE[0]),
                sample_rng.uniform(0, MAP_SIZE[1]),
                sample_rng.uniform(0, MAP_SIZE[2])
            ]))
    return points

def calculate_path_cost(path):
    if len(path) < 2:
        return 0.0
    cost = 0.0
    for i in range(len(path)-1):
        cost += np.linalg.norm(path[i+1] - path[i])
    return cost

def smooth_path(path_positions, obstacles, max_iter=150):
    if len(path_positions) <= 2:
        return path_positions, calculate_path_cost(path_positions)
    
    smooth_rng = np.random.default_rng(RANDOM_SEED + 1000)
    smoothed = list(path_positions)
    for _ in range(max_iter):
        if len(smoothed) <= 2:
            break
        i = smooth_rng.integers(0, len(smoothed) - 1)
        j = smooth_rng.integers(i + 1, len(smoothed))
        if j - i <= 1:
            continue
        if is_collision_free(smoothed[i], smoothed[j], obstacles):
            smoothed = smoothed[:i+1] + smoothed[j:]
    
    cost = calculate_path_cost(smoothed)
    return smoothed, cost

# =============================================================================
# ALGORİTMA SINIFLARI
# =============================================================================

class Node:
    def __init__(self, position, node_id):
        self.position = position
        self.node_id = node_id
        self.parent = None
        self.children = []
        self.cost = 0.0
        self.rewired = False

class BaseRRT:
    def __init__(self, is_rrt_star=False):
        self.is_rrt_star = is_rrt_star
        self.max_iterations = MAX_ITERATIONS
        self.iterations = 0
        
        self.first_solution_iteration = -1
        self.first_solution_cost = np.nan
        self.best_solution_iteration = -1
        self.best_solution_cost = np.inf
        
        self.best_cost_history = []
        self.solution_improvements = []
        
        self.collision_rejects = 0
        self.duplicate_rejects = 0
        self.rewire_count = 0
        self.parent_changes = 0
        self.runtime = 0.0
        
        self.nodes = [Node(START, 0)]
        self.node_positions = np.zeros((self.max_iterations + 2, 3))
        self.node_positions[0] = START
        self.num_nodes = 1
        
        self.obstacles = OBSTACLES
        
        self.goal_candidates = {}
        self.best_goal_parent = None
        
        self.latest_new_node = None
        self.current_radius = 0.0
        self.is_finished = False

    def get_nearest_node(self, random_point):
        dists = np.linalg.norm(self.node_positions[:self.num_nodes] - random_point, axis=1)
        return self.nodes[np.argmin(dists)]
        
    def steer(self, from_point, to_point):
        dist = np.linalg.norm(to_point - from_point)
        if dist < STEP_SIZE:
            return to_point
        return from_point + (to_point - from_point) * STEP_SIZE / dist
        
    def add_node(self, node):
        self.nodes.append(node)
        self.node_positions[self.num_nodes] = node.position
        self.num_nodes += 1

    def would_create_cycle(self, potential_parent, node):
        curr = potential_parent
        while curr is not None:
            if curr.node_id == node.node_id:
                return True
            curr = curr.parent
        return False

    def set_parent(self, node, new_parent):
        if self.would_create_cycle(new_parent, node):
            return False
            
        old_parent = node.parent
        if old_parent is not None:
            if node in old_parent.children:
                old_parent.children.remove(node)
                
        node.parent = new_parent
        if node not in new_parent.children:
            new_parent.children.append(node)
            
        node.cost = new_parent.cost + np.linalg.norm(node.position - new_parent.position)
        return True

    def update_descendant_costs(self, root):
        queue = deque([root])
        visited = set()
        while queue:
            curr = queue.popleft()
            if curr.node_id in visited:
                raise RuntimeError(f"Cycle detected in update_descendant_costs at node {curr.node_id}")
            visited.add(curr.node_id)
            for child in curr.children:
                child.cost = curr.cost + np.linalg.norm(child.position - curr.position)
                queue.append(child)
        self.refresh_best_goal()
                
    def refresh_best_goal(self):
        best_parent = None
        best_cost = np.inf
        
        for node_id, node in self.goal_candidates.items():
            dist_to_goal = np.linalg.norm(GOAL - node.position)
            if dist_to_goal > GOAL_THRESHOLD:
                continue
            if not is_collision_free(node.position, GOAL, self.obstacles):
                continue
                
            candidate_cost = node.cost + dist_to_goal
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_parent = node
                
        if best_parent is not None:
            if self.first_solution_iteration == -1:
                self.first_solution_iteration = self.iterations
                self.first_solution_cost = best_cost
                self.best_solution_iteration = self.iterations
                self.best_solution_cost = best_cost
                self.best_goal_parent = best_parent
                self.solution_improvements.append({"iteration": self.iterations, "cost": best_cost})
            else:
                if self.is_rrt_star and best_cost + COST_EPSILON < self.best_solution_cost:
                    self.best_solution_iteration = self.iterations
                    self.best_solution_cost = best_cost
                    self.best_goal_parent = best_parent
                    if len(self.solution_improvements) > 0 and self.solution_improvements[-1]["iteration"] == self.iterations:
                        self.solution_improvements[-1]["cost"] = best_cost
                    else:
                        self.solution_improvements.append({"iteration": self.iterations, "cost": best_cost})
                elif not self.is_rrt_star:
                    if self.best_goal_parent is None:
                        self.best_goal_parent = best_parent
                        self.best_solution_cost = best_cost

    def register_goal_candidate(self, node):
        dist_to_goal = np.linalg.norm(GOAL - node.position)
        if dist_to_goal <= GOAL_THRESHOLD:
            if is_collision_free(node.position, GOAL, self.obstacles):
                self.goal_candidates[node.node_id] = node
                self.refresh_best_goal()

    def calculate_search_radius(self):
        if not USE_ADAPTIVE_RADIUS:
            return SEARCH_RADIUS_MAX
        n = self.num_nodes
        if n <= 1:
            return SEARCH_RADIUS_MAX
        r = RRT_STAR_GAMMA * (np.log(n) / n) ** (1.0 / 3.0)
        return min(SEARCH_RADIUS_MAX, r)

    def get_final_path(self):
        if self.best_goal_parent is None:
            return []
            
        path = []
        current = self.best_goal_parent
        visited = set()
        
        while current is not None:
            if current.node_id in visited:
                raise RuntimeError("Cycle detected during path extraction")
            visited.add(current.node_id)
            path.append(current.position.copy())
            current = current.parent
            
        path.reverse()
        if np.linalg.norm(path[-1] - GOAL) > NODE_EPSILON:
            path.append(GOAL.copy())
        return path

    def finish_iteration(self):
        val = self.best_solution_cost if not np.isinf(self.best_solution_cost) else np.nan
        self.best_cost_history.append(val)
        if self.iterations >= self.max_iterations:
            self.is_finished = True

    def step(self, random_point):
        if self.is_finished:
            return False
            
        t0 = time.perf_counter()
        self.iterations += 1
        
        nearest_node = self.get_nearest_node(random_point)
        new_point = self.steer(nearest_node.position, random_point)
        
        dists_all = np.linalg.norm(self.node_positions[:self.num_nodes] - new_point, axis=1)
        if len(dists_all) > 0 and np.min(dists_all) <= NODE_EPSILON:
            self.duplicate_rejects += 1
            self.runtime += (time.perf_counter() - t0)
            self.finish_iteration()
            return True
        
        if not is_collision_free(nearest_node.position, new_point, self.obstacles):
            self.collision_rejects += 1
            self.runtime += (time.perf_counter() - t0)
            self.finish_iteration()
            return True
            
        new_node = Node(new_point, self.num_nodes)
        
        if not self.is_rrt_star:
            self.set_parent(new_node, nearest_node)
            self.add_node(new_node)
            self.latest_new_node = new_node
            self.register_goal_candidate(new_node)
            self.runtime += (time.perf_counter() - t0)
            self.finish_iteration()
            return True
            
        self.current_radius = self.calculate_search_radius()
        dists_to_new = dists_all
        near_indices = np.where(dists_to_new <= self.current_radius)[0]
        near_nodes = [self.nodes[i] for i in near_indices]
        
        best_parent = nearest_node
        min_cost = nearest_node.cost + np.linalg.norm(new_point - nearest_node.position)
        
        for near_node in near_nodes:
            cost = near_node.cost + np.linalg.norm(new_point - near_node.position)
            if cost < min_cost:
                if is_collision_free(near_node.position, new_point, self.obstacles):
                    if not self.would_create_cycle(near_node, new_node):
                        best_parent = near_node
                        min_cost = cost
                    
        self.set_parent(new_node, best_parent)
        self.add_node(new_node)
        self.latest_new_node = new_node
        self.register_goal_candidate(new_node)
        
        if best_parent.node_id != nearest_node.node_id:
            self.parent_changes += 1
        
        for near_node in near_nodes:
            if near_node.node_id == best_parent.node_id:
                continue
                
            candidate_cost = new_node.cost + np.linalg.norm(near_node.position - new_point)
            if candidate_cost + COST_EPSILON < near_node.cost:
                if not self.would_create_cycle(new_node, near_node):
                    if is_collision_free(new_point, near_node.position, self.obstacles):
                        self.rewire_count += 1
                        self.parent_changes += 1
                        
                        self.set_parent(near_node, new_node)
                        near_node.rewired = True
                        self.update_descendant_costs(near_node)
                        
        self.runtime += (time.perf_counter() - t0)
        self.finish_iteration()
        return True


# =============================================================================
# GÖRSELLEŞTİRME VE GUI
# =============================================================================

def draw_obstacles(ax, obstacles, presentation_mode=False):
    alpha_val = 0.15 if presentation_mode else 0.3
    edge_lw = 0.3 if presentation_mode else 0.5
    for (x, y, z, w, d, h) in obstacles:
        vertices = np.array([
            [x, y, z], [x+w, y, z], [x+w, y+d, z], [x, y+d, z],
            [x, y, z+h], [x+w, y, z+h], [x+w, y+d, z+h], [x, y+d, z+h]
        ])
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[3], vertices[0], vertices[4], vertices[7]]
        ]
        poly3d = Poly3DCollection(faces, alpha=alpha_val, facecolors='gray', edgecolors='k', linewidths=edge_lw)
        ax.add_collection3d(poly3d)

def generate_sphere_lines(cx, cy, cz, r, resolution=15):
    t = np.linspace(0, 2*np.pi, resolution)
    xy = np.column_stack((cx + r*np.cos(t), cy + r*np.sin(t), np.full_like(t, cz)))
    xz = np.column_stack((cx + r*np.cos(t), np.full_like(t, cy), cz + r*np.sin(t)))
    yz = np.column_stack((np.full_like(t, cx), cy + r*np.cos(t), cz + r*np.sin(t)))
    segments = []
    for arr in [xy, xz, yz]:
        for i in range(len(arr)-1):
            segments.append([arr[i], arr[i+1]])
    return segments

class Visualizer:
    def __init__(self, algo, ax, ax_stats, title, color_tree, color_path, presentation_mode=False):
        self.algo = algo
        self.ax = ax
        self.ax_stats = ax_stats
        self.color_tree = color_tree
        self.color_path = color_path
        self.presentation_mode = presentation_mode
        
        self.ax.set_title(title, fontsize=16 if presentation_mode else 12, fontweight='bold')
        self.ax.set_xlim([0, MAP_SIZE[0]])
        self.ax.set_ylim([0, MAP_SIZE[1]])
        self.ax.set_zlim([0, MAP_SIZE[2]])
        self.ax.set_xlabel('X', fontsize=10)
        self.ax.set_ylabel('Y', fontsize=10)
        self.ax.set_zlabel('Z', fontsize=10)
        self.ax.view_init(elev=25, azim=45)
        
        if presentation_mode:
            self.ax.set_box_aspect((1, 1, 1))
        
        self.ax.scatter(*START, color='green', s=150 if presentation_mode else 100, label='Start', zorder=5)
        self.ax.scatter(*GOAL, color='red', s=150 if presentation_mode else 100, label='Goal', zorder=5)
        
        draw_obstacles(self.ax, self.algo.obstacles, presentation_mode)
        
        dummy = [[[-1,-1,-1], [-1,-1,-1]]]
        tree_lw = 0.3 if presentation_mode else 0.5
        tree_alpha = 0.25 if presentation_mode else 0.6
        self.tree_lines = Line3DCollection(dummy, colors=self.color_tree, linewidths=tree_lw, alpha=tree_alpha)
        self.ax.add_collection3d(self.tree_lines)
        
        path_lw = 4.0 if presentation_mode else 3.5
        self.path_line = Line3DCollection(dummy, colors=self.color_path, linewidths=path_lw)
        self.ax.add_collection3d(self.path_line)
        
        self.smooth_line = Line3DCollection(dummy, colors='cyan', linewidths=4.5, linestyles='solid')
        self.ax.add_collection3d(self.smooth_line)
        
        self.radius_lines = Line3DCollection(dummy, colors='cyan', linewidths=1.0, alpha=0.3)
        self.ax.add_collection3d(self.radius_lines)
        
        self.smoothed_drawn = False
        self.smoothed_cost = np.nan

    def update(self, active_seed):
        # 1. Tree
        segments = []
        colors = []
        for node in self.algo.nodes[1:]:
            if node.parent:
                segments.append([node.parent.position, node.position])
                colors.append('magenta' if getattr(node, 'rewired', False) else self.color_tree)
        if not segments:
            segments = [[[-1,-1,-1], [-1,-1,-1]]]
            colors = [self.color_tree]
        self.tree_lines.set_segments(segments)
        self.tree_lines.set_color(colors)
        
        # 2. Path
        if self.algo.best_goal_parent:
            path_segments = []
            path_pos = self.algo.get_final_path()
            if len(path_pos) > 1:
                for i in range(len(path_pos)-1):
                    path_segments.append([path_pos[i], path_pos[i+1]])
            if not path_segments:
                path_segments = [[[-1,-1,-1], [-1,-1,-1]]]
            self.path_line.set_segments(path_segments)
        else:
            self.path_line.set_segments([[[-1,-1,-1], [-1,-1,-1]]])
            
        # 3. Radius Sphere
        if not self.presentation_mode:
            if self.algo.is_rrt_star and self.algo.latest_new_node and self.algo.current_radius > 0:
                cx, cy, cz = self.algo.latest_new_node.position
                sphere_segs = generate_sphere_lines(cx, cy, cz, self.algo.current_radius)
                if not sphere_segs:
                    sphere_segs = [[[-1,-1,-1], [-1,-1,-1]]]
                self.radius_lines.set_segments(sphere_segs)
            else:
                self.radius_lines.set_segments([[[-1,-1,-1], [-1,-1,-1]]])
            
        # 4. Smooth Path (When finished)
        if self.algo.is_finished and not self.smoothed_drawn and self.algo.best_goal_parent and not self.presentation_mode:
            path_pos = self.algo.get_final_path()
            smooth_pos, self.smoothed_cost = smooth_path(path_pos, self.algo.obstacles)
            smooth_segments = [[smooth_pos[i], smooth_pos[i+1]] for i in range(len(smooth_pos)-1)]
            if not smooth_segments:
                smooth_segments = [[[-1,-1,-1], [-1,-1,-1]]]
            self.smooth_line.set_segments(smooth_segments)
            self.smoothed_drawn = True
            self.radius_lines.set_segments([[[-1,-1,-1], [-1,-1,-1]]])
            
        # 5. Stats Panel (Normal mode)
        if self.ax_stats and not self.presentation_mode:
            status = "REACHED" if self.algo.best_goal_parent else "SEARCHING..."
            internal_impr = 0.0
            if not np.isnan(self.algo.first_solution_cost) and not np.isinf(self.algo.best_solution_cost):
                internal_impr = (self.algo.first_solution_cost - self.algo.best_solution_cost) / self.algo.first_solution_cost * 100
                
            t_first = f"Iter: {self.algo.first_solution_iteration} | Cost: {self.algo.first_solution_cost:.2f}" if self.algo.first_solution_iteration != -1 else "N/A"
            t_best = f"Iter: {self.algo.best_solution_iteration} | Cost: {self.algo.best_solution_cost:.2f}" if self.algo.best_solution_iteration != -1 else "N/A"
            
            stats = (
                f"{'RRT*' if self.algo.is_rrt_star else 'RRT'} STATISTICS\n"
                f"{'='*30}\n"
                f"Random Seed: {active_seed}\n"
                f"Iteration: {self.algo.iterations} / {self.algo.max_iterations}\n"
                f"Nodes: {self.algo.num_nodes}\n"
                f"Goal: {status}\n\n"
                f"Goal Candidates: {len(self.algo.goal_candidates)}\n"
                f"Duplicate Rejects: {self.algo.duplicate_rejects}\n\n"
                f"First Solution:\n  {t_first}\n\n"
                f"Current Best:\n  {t_best}\n\n"
                f"Internal Improvement:\n  {internal_impr:.2f} %\n\n"
                f"Improvement Events:\n  {max(0, len(self.algo.solution_improvements)-1)}\n\n"
                f"Rewirings: {self.algo.rewire_count}\n"
                f"CPU Time: {self.algo.runtime:.3f} sec"
            )
            
            self.ax_stats.clear()
            self.ax_stats.axis('off')
            self.ax_stats.text(0.05, 0.95, stats, transform=self.ax_stats.transAxes, 
                               fontsize=10, va='top', family='monospace')

# =============================================================================
# VALIDATION TESTLERI
# =============================================================================

def validate_no_cycles(algo):
    for node in algo.nodes:
        visited = set()
        curr = node
        while curr is not None:
            assert curr.node_id not in visited, f"Tree cycle detected at node {curr.node_id}"
            visited.add(curr.node_id)
            curr = curr.parent

def validate_parent_child_consistency(algo):
    for node in algo.nodes:
        if node.parent is not None:
            assert node in node.parent.children, f"Node {node.node_id} has parent {node.parent.node_id} but is not in children list"
        for child in node.children:
            assert child.parent is node, f"Child {child.node_id} is in node {node.node_id} children but has different parent"
            assert node.children.count(child) == 1, f"Duplicate child {child.node_id} in node {node.node_id}"

def validate_cost_consistency(algo):
    start_node = algo.nodes[0]
    assert np.isclose(start_node.cost, 0.0), "Start node cost is not 0"
    assert start_node.parent is None, "Start node has parent"
        
    for node in algo.nodes[1:]:
        expected_cost = node.parent.cost + np.linalg.norm(node.position - node.parent.position)
        assert np.isclose(node.cost, expected_cost, rtol=1e-6, atol=1e-7), f"Cost inconsistency at node {node.node_id}: {node.cost} != {expected_cost}"

def validate_tree_edges_collision_free(algo):
    for node in algo.nodes[1:]:
        assert is_collision_free(node.parent.position, node.position, algo.obstacles), f"Edge from {node.parent.node_id} to {node.node_id} is in collision"

def validate_no_zero_length_edges(algo):
    for node in algo.nodes[1:]:
        dist = np.linalg.norm(node.position - node.parent.position)
        assert dist > NODE_EPSILON, f"Zero-length edge detected between {node.parent.node_id} and {node.node_id}"

def validate_goal_candidates(algo):
    if algo.best_goal_parent is None:
        return
        
    actual_candidates = []
    for node in algo.nodes:
        dist_to_goal = np.linalg.norm(GOAL - node.position)
        if dist_to_goal <= GOAL_THRESHOLD and is_collision_free(node.position, GOAL, algo.obstacles):
            actual_candidates.append(node)
            
    assert len(algo.goal_candidates) == len(actual_candidates), "Goal candidates registry size mismatch"
    for c in actual_candidates:
        assert c.node_id in algo.goal_candidates, f"Valid candidate {c.node_id} missing from registry"

    if algo.is_rrt_star:
        min_cost = np.inf
        for c in actual_candidates:
            cost = c.cost + np.linalg.norm(GOAL - c.position)
            if cost < min_cost:
                min_cost = cost
        assert np.isclose(algo.best_solution_cost, min_cost, rtol=1e-6, atol=1e-6), f"best_solution_cost {algo.best_solution_cost} does not match global minimum candidate cost {min_cost}"

def validate_final_path(algo):
    if algo.best_goal_parent is None:
        return
    path = algo.get_final_path()
    assert len(path) > 1, "Path is empty or has only one node but goal was reached"
    assert np.allclose(path[0], START), "Final path does not start at START"
    assert np.allclose(path[-1], GOAL), "Final path does not end at GOAL"
    
    for i in range(len(path)-1):
        assert np.linalg.norm(path[i+1] - path[i]) > NODE_EPSILON, f"Zero-length segment in final path at index {i}"

def validate_final_path_collision_free(algo):
    if algo.best_goal_parent is None:
        return
    path = algo.get_final_path()
    for i in range(len(path)-1):
        assert is_collision_free(path[i], path[i+1], algo.obstacles), f"Final path edge from {path[i]} to {path[i+1]} is in collision"

def validate_goal_terminal_outside_tree(algo):
    if algo.best_goal_parent is not None:
        assert algo.best_goal_parent in algo.nodes, "best_goal_parent not in tree"

def validate_best_cost_path_cost_consistency(algo):
    if algo.best_goal_parent is not None:
        path_cost = calculate_path_cost(algo.get_final_path())
        assert np.isclose(path_cost, algo.best_solution_cost, rtol=1e-6, atol=1e-6), f"Path cost {path_cost} != Best cost {algo.best_solution_cost}"

def validate_history_consistency(algo):
    if algo.best_goal_parent is None:
        return
    
    assert len(algo.best_cost_history) == algo.iterations, "best_cost_history length mismatch"
    finite_indices = [i for i, cost in enumerate(algo.best_cost_history) if np.isfinite(cost)]
    assert len(finite_indices) > 0, "Goal reached but no finite cost in history"
    first_idx = finite_indices[0]
    assert first_idx + 1 == algo.first_solution_iteration, "First finite cost iteration mismatch"
    assert np.isclose(algo.best_cost_history[first_idx], algo.first_solution_cost, rtol=1e-6, atol=1e-6), "First finite cost mismatch"
    
    for i in range(len(algo.solution_improvements)-1):
        assert algo.solution_improvements[i+1]["cost"] + COST_EPSILON < algo.solution_improvements[i]["cost"], "Improvements are not strictly decreasing"
        assert algo.solution_improvements[i+1]["iteration"] > algo.solution_improvements[i]["iteration"], "Improvement iterations are not strictly increasing"
        
    if not algo.is_rrt_star:
        for i in range(first_idx, len(algo.best_cost_history)):
            assert np.isclose(algo.best_cost_history[i], algo.first_solution_cost, rtol=1e-6, atol=1e-6), "RRT cost should be constant after first solution"

def run_algo_validation(algo, name="Algo"):
    validate_no_cycles(algo)
    validate_parent_child_consistency(algo)
    validate_cost_consistency(algo)
    validate_tree_edges_collision_free(algo)
    validate_no_zero_length_edges(algo)
    validate_goal_candidates(algo)
    validate_final_path(algo)
    validate_final_path_collision_free(algo)
    validate_best_cost_path_cost_consistency(algo)
    validate_history_consistency(algo)

def run_headless_self_test():
    print("Running headless validation tests (MAX_ITERATIONS = 3000)...")
    
    random_points = generate_random_points(RANDOM_SEED)
    
    rrt = BaseRRT(is_rrt_star=False)
    for p in random_points:
        rrt.step(p)
        
    rrtstar = BaseRRT(is_rrt_star=True)
    for p in random_points:
        rrtstar.step(p)
        
    print("\nRRT VALIDATION")
    run_algo_validation(rrt)
    print("PASS")
    
    print("\nRRT* VALIDATION")
    run_algo_validation(rrtstar)
    print("PASS")
    
    print("\nLifecycle")
    assert rrt.iterations == MAX_ITERATIONS, "RRT iterations != MAX_ITERATIONS"
    print("RRT iterations == MAX_ITERATIONS: PASS")
    
    assert rrtstar.iterations == MAX_ITERATIONS, "RRT* iterations != MAX_ITERATIONS"
    print("RRT* iterations == MAX_ITERATIONS: PASS")
    
    assert rrt.is_finished, "RRT is not finished"
    print("RRT is_finished: PASS")
    
    assert rrtstar.is_finished, "RRT* is not finished"
    print("RRT* is_finished: PASS")
    
    print("\nALL TESTS PASSED\n")

def run_multi_seed_test():
    seeds = [1, 2, 3, 10, 21, 42, 77, 100, 256, 999]
    print("\nMULTI-SEED REGRESSION\n")
    
    passed_count = 0
    for s in seeds:
        try:
            random_points = generate_random_points(s)
            rrtstar = BaseRRT(is_rrt_star=True)
            for p in random_points:
                rrtstar.step(p)
                
            run_algo_validation(rrtstar)
            print(f"Seed {s:<4} PASS")
            passed_count += 1
        except Exception as e:
            print(f"Seed {s:<4} FAIL: {e}")
            raise
            
    print(f"\n{passed_count} / {len(seeds)} seeds passed")

def run_demo_seed_scanner(save_csv=False):
    print(f"Scanning seeds from {DEMO_SEED_START} to {DEMO_SEED_END} for optimal educational demonstration...\n")
    
    results = []
    
    for s in range(DEMO_SEED_START, DEMO_SEED_END + 1):
        random_points = generate_random_points(s)
        
        rrt = BaseRRT(is_rrt_star=False)
        for p in random_points:
            rrt.step(p)
            
        rrtstar = BaseRRT(is_rrt_star=True)
        for p in random_points:
            rrtstar.step(p)
            
        try:
            run_algo_validation(rrtstar)
        except Exception as e:
            print(f"Seed {s} failed algorithmic invariant: {e}")
            sys.exit(1)
            
        rrt_reached = rrt.best_goal_parent is not None
        rrtstar_reached = rrtstar.best_goal_parent is not None
        
        if not rrtstar_reached or not rrt_reached:
            continue
            
        rrt_cost = rrt.best_solution_cost
        rrtstar_first = rrtstar.first_solution_cost
        rrtstar_best = rrtstar.best_solution_cost
        
        internal_impr = (rrtstar_first - rrtstar_best) / rrtstar_first * 100
        vs_impr = (rrt_cost - rrtstar_best) / rrt_cost * 100
        drops = max(0, len(rrtstar.solution_improvements) - 1)
        
        if internal_impr >= DEMO_MIN_IMPROVEMENT_PERCENT and drops >= DEMO_MIN_COST_DROPS and rrtstar_first > rrtstar_best:
            score = (internal_impr * 2.0) + (min(drops, 10) * 2.0) + vs_impr
            
            results.append({
                "seed": s,
                "rrt_cost": rrt_cost,
                "rrtstar_first": rrtstar_first,
                "rrtstar_best": rrtstar_best,
                "internal_impr": internal_impr,
                "vs_impr": vs_impr,
                "drops": drops,
                "rewires": rrtstar.rewire_count,
                "best_iter": rrtstar.best_solution_iteration,
                "score": score
            })
            
    results.sort(key=lambda x: x["score"], reverse=True)
    
    print("================================================================================================================")
    print("TOP DEMO SEEDS")
    print("================================================================================================================")
    print(f"{'Rank':<4} {'Seed':<6} {'RRT Cost':<10} {'RRT* First':<12} {'RRT* Best':<10} {'Internal Improve':<18} {'vs RRT Improve':<16} {'Cost Drops':<12} {'Rewires':<9} {'Best Iter':<10}")
    print("-" * 112)
    
    for i, r in enumerate(results[:10]):
        rank = i + 1
        print(f"{rank:<4} {r['seed']:<6} {r['rrt_cost']:<10.2f} {r['rrtstar_first']:<12.2f} {r['rrtstar_best']:<10.2f} {f'{r['internal_impr']:.2f} %':<18} {f'{r['vs_impr']:.2f} %':<16} {r['drops']:<12} {r['rewires']:<9} {r['best_iter']:<10}")
        
    print("================================================================================================================\n")
    
    if results:
        top = results[0]
        print(f"Recommended Demo Seed: {top['seed']}\n")
    else:
        print("No seeds matched the demo criteria.\n")


def run_benchmark(num_seeds):
    print(f"Running benchmark on seeds {BENCHMARK_SEED_START} to {BENCHMARK_SEED_START + num_seeds - 1}...")
    
    results = []
    
    for s in range(BENCHMARK_SEED_START, BENCHMARK_SEED_START + num_seeds):
        random_points = generate_random_points(s)
        
        rrt = BaseRRT(is_rrt_star=False)
        for p in random_points:
            rrt.step(p)
            
        rrtstar = BaseRRT(is_rrt_star=True)
        for p in random_points:
            rrtstar.step(p)
            
        try:
            run_algo_validation(rrt)
            run_algo_validation(rrtstar)
        except Exception as e:
            print(f"Seed {s} failed algorithmic invariant during benchmark: {e}")
            sys.exit(1)
            
        rrt_reached = rrt.best_goal_parent is not None
        rrtstar_reached = rrtstar.best_goal_parent is not None
        
        rrt_cost = rrt.best_solution_cost if rrt_reached else np.nan
        rrtstar_first = rrtstar.first_solution_cost if rrtstar_reached else np.nan
        rrtstar_best = rrtstar.best_solution_cost if rrtstar_reached else np.nan
        
        internal_impr = np.nan
        if rrtstar_reached and rrtstar_first > 0:
            internal_impr = (rrtstar_first - rrtstar_best) / rrtstar_first * 100
            
        vs_impr = np.nan
        if rrt_reached and rrtstar_reached and rrt_cost > 0:
            vs_impr = (rrt_cost - rrtstar_best) / rrt_cost * 100
            
        results.append({
            "seed": s,
            "rrt_goal_reached": rrt_reached,
            "rrt_first_solution_iteration": rrt.first_solution_iteration,
            "rrt_first_cost": rrt.first_solution_cost,
            "rrt_final_cost": rrt_cost,
            "rrt_nodes": rrt.num_nodes,
            "rrt_collision_rejects": rrt.collision_rejects,
            "rrt_duplicate_rejects": rrt.duplicate_rejects,
            "rrt_runtime": rrt.runtime,
            
            "rrtstar_goal_reached": rrtstar_reached,
            "rrtstar_first_solution_iteration": rrtstar.first_solution_iteration,
            "rrtstar_first_cost": rrtstar_first,
            "rrtstar_best_iteration": rrtstar.best_solution_iteration,
            "rrtstar_best_cost": rrtstar_best,
            "rrtstar_nodes": rrtstar.num_nodes,
            "rrtstar_collision_rejects": rrtstar.collision_rejects,
            "rrtstar_duplicate_rejects": rrtstar.duplicate_rejects,
            "rrtstar_rewires": rrtstar.rewire_count,
            "rrtstar_parent_changes": rrtstar.parent_changes,
            "rrtstar_improvement_events": max(0, len(rrtstar.solution_improvements)-1),
            "rrtstar_internal_improvement_percent": internal_impr,
            "rrtstar_runtime": rrtstar.runtime,
            
            "rrt_vs_rrtstar_improvement_percent": vs_impr
        })
    
    # Save CSV
    csv_file = f"rrt_rrtstar_benchmark_{num_seeds}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    # Calculate stats
    trials = len(results)
    rrt_success = sum(1 for r in results if r["rrt_goal_reached"])
    rrtstar_success = sum(1 for r in results if r["rrtstar_goal_reached"])
    
    comparable_trials = [r for r in results if r["rrt_goal_reached"] and r["rrtstar_goal_reached"]]
    comp_count = len(comparable_trials)
    
    if comp_count > 0:
        rrt_costs = [r["rrt_final_cost"] for r in comparable_trials]
        rrtstar_costs = [r["rrtstar_best_cost"] for r in comparable_trials]
        vs_imprs = [r["rrt_vs_rrtstar_improvement_percent"] for r in comparable_trials]
        int_imprs = [r["rrtstar_internal_improvement_percent"] for r in comparable_trials]
        
        rrt_mean = statistics.mean(rrt_costs)
        rrt_med = statistics.median(rrt_costs)
        rrtstar_mean = statistics.mean(rrtstar_costs)
        rrtstar_med = statistics.median(rrtstar_costs)
        
        mean_vs_impr = statistics.mean(vs_imprs)
        med_vs_impr = statistics.median(vs_imprs)
        
        mean_int_impr = statistics.mean(int_imprs)
        med_int_impr = statistics.median(int_imprs)
        
        rrtstar_better = sum(1 for r in comparable_trials if r["rrtstar_best_cost"] + COST_EPSILON < r["rrt_final_cost"])
        rrt_better = sum(1 for r in comparable_trials if r["rrt_final_cost"] + COST_EPSILON < r["rrtstar_best_cost"])
        tied = comp_count - rrtstar_better - rrt_better
        
        rrt_rts = [r["rrt_runtime"] for r in comparable_trials]
        rrtstar_rts = [r["rrtstar_runtime"] for r in comparable_trials]
        rrt_rt_mean = statistics.mean(rrt_rts)
        rrtstar_rt_mean = statistics.mean(rrtstar_rts)
    else:
        rrt_mean = rrt_med = rrtstar_mean = rrtstar_med = 0.0
        mean_vs_impr = med_vs_impr = mean_int_impr = med_int_impr = 0.0
        rrtstar_better = rrt_better = tied = 0
        rrt_rt_mean = rrtstar_rt_mean = 0.0
        rrt_costs = []
        rrtstar_costs = []
        vs_imprs = []
        int_imprs = []
        rrt_rts = []
        rrtstar_rts = []

    summary_text = (
        "================================================================\n"
        "RRT vs RRT* BENCHMARK\n"
        "================================================================\n\n"
        f"Trials:                       {trials}\n\n"
        "Success Rate\n"
        f"RRT:                          {rrt_success / trials * 100:.1f} %\n"
        f"RRT*:                         {rrtstar_success / trials * 100:.1f} %\n\n"
        f"Comparable Successful Runs:  {comp_count} / {trials}\n\n"
        "Path Cost\n"
        f"RRT Mean:                     {rrt_mean:.2f}\n"
        f"RRT Median:                   {rrt_med:.2f}\n"
        f"RRT* Mean:                    {rrtstar_mean:.2f}\n"
        f"RRT* Median:                  {rrtstar_med:.2f}\n\n"
        "RRT* vs RRT\n"
        f"Mean Improvement:             {mean_vs_impr:.2f} %\n"
        f"Median Improvement:           {med_vs_impr:.2f} %\n"
        f"RRT* Better:                  {rrtstar_better} / {comp_count}\n"
        f"RRT Better:                   {rrt_better} / {comp_count}\n"
        f"Tied:                         {tied} / {comp_count}\n\n"
        "RRT* Internal Optimization\n"
        f"Mean:                         {mean_int_impr:.2f} %\n"
        f"Median:                       {med_int_impr:.2f} %\n\n"
        "Runtime\n"
        f"RRT Mean:                     {rrt_rt_mean:.3f} sec\n"
        f"RRT* Mean:                    {rrtstar_rt_mean:.3f} sec\n\n"
        "================================================================\n"
        "MULTI-SEED BENCHMARK RESULT\n\n"
        f"Trials: {trials}\n"
        f"Comparable Trials: {comp_count}\n"
        f"RRT Success: {rrt_success / trials * 100:.1f}%\n"
        f"RRT* Success: {rrtstar_success / trials * 100:.1f}%\n"
        f"RRT Mean Cost: {rrt_mean:.2f}\n"
        f"RRT* Mean Cost: {rrtstar_mean:.2f}\n"
        f"Median Improvement: {med_vs_impr:.2f}%\n"
        f"RRT* Better Count: {rrtstar_better}\n"
        f"Runtime comparison: RRT {rrt_rt_mean:.3f}s vs RRT* {rrtstar_rt_mean:.3f}s\n"
        "================================================================\n"
    )
    print(summary_text)
    
    with open("rrt_rrtstar_benchmark_summary.txt", "w") as f:
        f.write(summary_text)
        
    if comp_count > 0:
        plt.figure(figsize=(10, 6), facecolor='white')
        plt.boxplot([rrt_costs, rrtstar_costs])
        plt.xticks([1, 2], ['RRT', 'RRT*'])
        plt.title('RRT vs RRT* Path Cost Distribution', fontsize=16, fontweight='bold')
        plt.text(0.5, 0.95, f"N={comp_count} comparable deterministic runs", transform=plt.gca().transAxes, ha='center', va='top', fontsize=12)
        plt.text(1, rrt_med, f" Median\n {rrt_med:.2f}", va='center', color='blue', fontsize=10)
        plt.text(2, rrtstar_med, f" Median\n {rrtstar_med:.2f}", va='center', color='green', fontsize=10)
        plt.ylabel('Path Cost', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig('benchmark_path_cost.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6), facecolor='white')
        plt.hist(vs_imprs, bins=20, color='purple', alpha=0.7, edgecolor='black')
        plt.axvline(med_vs_impr, color='red', linestyle='dashed', linewidth=2, label=f'Median: {med_vs_impr:.2f}%')
        plt.title('RRT* vs RRT Path Cost Improvement', fontsize=16, fontweight='bold')
        plt.text(0.5, 0.95, f"N={comp_count} comparable deterministic runs", transform=plt.gca().transAxes, ha='center', va='top', fontsize=12)
        plt.xlabel('Path Cost Reduction vs RRT (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig('benchmark_improvement.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6), facecolor='white')
        plt.boxplot([rrt_rts, rrtstar_rts])
        plt.xticks([1, 2], ['RRT', 'RRT*'])
        plt.title('Planning Runtime Comparison', fontsize=16, fontweight='bold')
        plt.text(0.5, 0.95, f"N={comp_count} trials | 3000 iterations per trial", transform=plt.gca().transAxes, ha='center', va='top', fontsize=12)
        plt.ylabel('Runtime (sec)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig('benchmark_runtime.png', dpi=200, bbox_inches='tight')
        plt.close()
        
    print(f"Benchmark complete. Data saved to {csv_file}, summary to txt, and plots to PNGs.")

# =============================================================================
# MAIN
# =============================================================================

is_playing = False
steps_per_frame = DRAW_EVERY_N_ITERATIONS
summary_printed = False
exported_final = False

def create_kpi_card(ax, title, value, subtext=""):
    ax.clear()
    ax.axis('off')
    # Background box
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='lightgray', linewidth=2, transform=ax.transAxes)
    ax.fill([0, 1, 1, 0], [0, 0, 1, 1], color='#f8f9fa', transform=ax.transAxes)
    
    ax.text(0.5, 0.75, title, ha='center', va='center', fontsize=11, transform=ax.transAxes, color='black')
    ax.text(0.5, 0.40, value, ha='center', va='center', fontsize=24, fontweight='bold', transform=ax.transAxes, color='black')
    if subtext:
        ax.text(0.5, 0.15, subtext, ha='center', va='center', fontsize=10, transform=ax.transAxes, color='dimgray')

def main():
    parser = argparse.ArgumentParser(description="RRT vs RRT* 3D Path Planning")
    parser.add_argument('--test', action='store_true', help="Run headless validation tests")
    parser.add_argument('--test-multi-seed', action='store_true', help="Run multi-seed regression tests")
    parser.add_argument('--find-demo-seeds', action='store_true', help="Scan for best educational seeds")
    parser.add_argument('--benchmark', action='store_true', help="Run headless statistical benchmark")
    parser.add_argument('--benchmark-seeds', type=int, default=BENCHMARK_SEEDS, help="Number of seeds for benchmark")
    parser.add_argument('--save-csv', action='store_true', help="Save demo seed scanner results to CSV (deprecated in favor of benchmark)")
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help="Random seed for GUI run")
    parser.add_argument('--presentation', action='store_true', help="Launch with LinkedIn presentation mode layout")
    parser.add_argument('--auto-start', action='store_true', help="Auto start animation")
    parser.add_argument('--export-final', action='store_true', help="Export high-res PNG at end of simulation")
    parser.add_argument('--record-demo', action='store_true', help="Export MP4 video of simulation (requires FFmpeg)")
    parser.add_argument('--demo-speed', type=int, default=DRAW_EVERY_N_ITERATIONS, help="Steps per animation frame")
    args = parser.parse_args()
    
    if args.test:
        run_headless_self_test()
        sys.exit(0)
    elif args.test_multi_seed:
        run_multi_seed_test()
        sys.exit(0)
    elif args.find_demo_seeds:
        run_demo_seed_scanner(save_csv=args.save_csv)
        sys.exit(0)
    elif args.benchmark:
        run_benchmark(args.benchmark_seeds)
        sys.exit(0)

    active_seed = args.seed
    presentation_mode = args.presentation
    global steps_per_frame
    steps_per_frame = args.demo_speed

    global rrt, rrtstar, vis_rrt, vis_rrtstar, random_points
    global btn_play, btn_reset, slider_speed, line_rrt_cost, line_rrtstar_cost
    global scatter_impr, scatter_first, scatter_best
    global ax_rrt, ax_rrtstar, ax_cost, fig, ax_stats_rrt, ax_stats_rrtstar
    global ax_kpi_1, ax_kpi_2, ax_kpi_3, ax_kpi_4
    global is_playing, summary_printed
    
    if args.auto_start:
        is_playing = True

    try:
        validate_start_goal(OBSTACLES)
    except ValueError as e:
        print(f"\nHATA: {e}\nLütfen parametreleri düzeltin.")
        sys.exit(1)

    random_points = generate_random_points(active_seed)
    
    if presentation_mode:
        fig = plt.figure(figsize=(16, 9), facecolor='white')
        fig.canvas.manager.set_window_title("RRT vs RRT* — 3D Path Planning")
        fig.suptitle("RRT vs RRT* — 3D PATH PLANNING", fontsize=22, fontweight='bold', y=0.96)
        fig.text(0.5, 0.91, f"Same sample sequence • {MAX_ITERATIONS} iterations • Seed: {active_seed}", ha='center', fontsize=14, color='dimgray')
        
        # Adjust spacing for a clean 16:9 layout without overlap
        fig.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.95, hspace=0.35, wspace=0.1)
        
        gs = fig.add_gridspec(20, 24)
        
        ax_rrt = fig.add_subplot(gs[0:11, 0:11], projection='3d')
        ax_rrtstar = fig.add_subplot(gs[0:11, 13:24], projection='3d')
        
        ax_kpi_1 = fig.add_subplot(gs[11:14, 2:6])
        ax_kpi_2 = fig.add_subplot(gs[11:14, 7:11])
        ax_kpi_3 = fig.add_subplot(gs[11:14, 13:17])
        ax_kpi_4 = fig.add_subplot(gs[11:14, 18:22])
        
        for k in [ax_kpi_1, ax_kpi_2, ax_kpi_3, ax_kpi_4]:
            k.axis('off')
            
        ax_cost = fig.add_subplot(gs[15:20, 2:22])
        
        ax_stats_rrt = None
        ax_stats_rrtstar = None
    else:
        fig = plt.figure(figsize=(18, 10))
        fig.canvas.manager.set_window_title("Professional RRT vs RRT* 3D Simulator")
        gs = fig.add_gridspec(24, 24)
        
        ax_rrt = fig.add_subplot(gs[0:15, 0:12], projection='3d')
        ax_rrtstar = fig.add_subplot(gs[0:15, 12:24], projection='3d')
        
        ax_stats_rrt = fig.add_subplot(gs[15:21, 0:6])
        ax_stats_rrt.axis('off')
        ax_stats_rrtstar = fig.add_subplot(gs[15:21, 6:12])
        ax_stats_rrtstar.axis('off')
        
        ax_cost = fig.add_subplot(gs[15:22, 12:24])
    
    rrt = BaseRRT(is_rrt_star=False)
    rrtstar = BaseRRT(is_rrt_star=True)
    
    vis_rrt = Visualizer(rrt, ax_rrt, ax_stats_rrt, "RRT", "blue", "orange", presentation_mode)
    vis_rrtstar = Visualizer(rrtstar, ax_rrtstar, ax_stats_rrtstar, "RRT*", "purple", "yellow", presentation_mode)
    
    if presentation_mode:
        ax_cost.set_title("PATH COST CONVERGENCE", fontweight='bold', fontsize=16)
    else:
        ax_cost.set_title("Path Cost vs Iteration", fontweight='bold')
        
    ax_cost.set_xlabel("Iteration", fontsize=12)
    ax_cost.set_ylabel("Best Path Cost", fontsize=12)
    ax_cost.grid(True, linestyle='--', alpha=0.3 if presentation_mode else 0.7)
    
    line_rrt_cost, = ax_cost.plot([], [], color='orange', label='RRT Cost', linewidth=3 if presentation_mode else 2)
    line_rrtstar_cost, = ax_cost.plot([], [], color='green', label='RRT* Cost', linewidth=3 if presentation_mode else 2)
    
    scatter_impr = ax_cost.scatter([], [], color='red', s=60 if presentation_mode else 40, zorder=5, label='Improvement Event')
    scatter_first = ax_cost.scatter([], [], color='blue', s=120 if presentation_mode else 80, marker='s', zorder=6, label='First Solution')
    scatter_best = ax_cost.scatter([], [], color='purple', s=250 if presentation_mode else 100, marker='*', zorder=7, label='Best Solution')
    
    ax_cost.legend(loc='upper right', fontsize=12 if presentation_mode else 10)
    
    if not presentation_mode:
        ax_play = fig.add_axes([0.60, 0.02, 0.08, 0.04])
        ax_reset = fig.add_axes([0.70, 0.02, 0.08, 0.04])
        ax_speed = fig.add_axes([0.83, 0.02, 0.12, 0.04])
        
        btn_play = Button(ax_play, 'Start / Pause')
        btn_reset = Button(ax_reset, 'Reset')
        slider_speed = Slider(ax_speed, 'Speed', 1, 50, valinit=steps_per_frame, valstep=1)
        
        def toggle_play(event):
            global is_playing
            is_playing = not is_playing
            btn_play.label.set_text("Pause" if is_playing else "Start")

        def reset_sim(event):
            global is_playing, rrt, rrtstar, vis_rrt, vis_rrtstar, summary_printed, exported_final
            is_playing = False
            btn_play.label.set_text("Start")
            summary_printed = False
            exported_final = False
            
            rrt = BaseRRT(is_rrt_star=False)
            rrtstar = BaseRRT(is_rrt_star=True)
            
            ax_rrt.cla()
            ax_rrtstar.cla()
            vis_rrt = Visualizer(rrt, ax_rrt, ax_stats_rrt, "RRT", "blue", "orange", presentation_mode)
            vis_rrtstar = Visualizer(rrtstar, ax_rrtstar, ax_stats_rrtstar, "RRT*", "purple", "yellow", presentation_mode)
            
            line_rrt_cost.set_data([], [])
            line_rrtstar_cost.set_data([], [])
            scatter_impr.set_offsets(np.empty((0, 2)))
            scatter_first.set_offsets(np.empty((0, 2)))
            scatter_best.set_offsets(np.empty((0, 2)))
            [t.remove() for t in ax_cost.texts] # Clear annotations
            fig.canvas.draw()

        def update_speed(val):
            global steps_per_frame
            steps_per_frame = int(val)
            
        btn_play.on_clicked(toggle_play)
        btn_reset.on_clicked(reset_sim)
        slider_speed.on_changed(update_speed)
    
    def print_final_summary():
        global exported_final
        impr_internal = 0.0
        impr_vs = 0.0
        if not np.isnan(rrtstar.first_solution_cost) and not np.isinf(rrtstar.best_solution_cost):
            impr_internal = (rrtstar.first_solution_cost - rrtstar.best_solution_cost) / rrtstar.first_solution_cost * 100
        if rrt.best_goal_parent and rrtstar.best_goal_parent:
            impr_vs = (rrt.best_solution_cost - rrtstar.best_solution_cost) / rrt.best_solution_cost * 100
            
        print("\n=========================================================")
        print("                      RRT vs RRT*")
        print("=========================================================")
        print(f"{'Metric':<30} {'RRT':<14} {'RRT*':<14}")
        print("-" * 60)
        print(f"{'Iterations':<30} {rrt.iterations:<14} {rrtstar.iterations:<14}")
        print(f"{'Nodes':<30} {rrt.num_nodes:<14} {rrtstar.num_nodes:<14}")
        print(f"{'Goal Candidates':<30} {len(rrt.goal_candidates):<14} {len(rrtstar.goal_candidates):<14}")
        print(f"{'Duplicate Rejects':<30} {rrt.duplicate_rejects:<14} {rrtstar.duplicate_rejects:<14}")
        print(f"{'Collision Rejects':<30} {rrt.collision_rejects:<14} {rrtstar.collision_rejects:<14}")
        print(f"{'Rewirings':<30} {rrt.rewire_count:<14} {rrtstar.rewire_count:<14}")
        
        c_first_rrt = f"{rrt.first_solution_cost:.2f}" if not np.isnan(rrt.first_solution_cost) else "-"
        c_first_rrtstar = f"{rrtstar.first_solution_cost:.2f}" if not np.isnan(rrtstar.first_solution_cost) else "-"
        print(f"{'First Solution Cost':<30} {c_first_rrt:<14} {c_first_rrtstar:<14}")
        
        c_best_rrt = f"{rrt.best_solution_cost:.2f}" if not np.isinf(rrt.best_solution_cost) else "-"
        c_best_rrtstar = f"{rrtstar.best_solution_cost:.2f}" if not np.isinf(rrtstar.best_solution_cost) else "-"
        print(f"{'Best Raw Cost':<30} {c_best_rrt:<14} {c_best_rrtstar:<14}")
        
        raw_path_rrt = f"{calculate_path_cost(rrt.get_final_path()):.2f}" if rrt.best_goal_parent else "-"
        raw_path_rrtstar = f"{calculate_path_cost(rrtstar.get_final_path()):.2f}" if rrtstar.best_goal_parent else "-"
        print(f"{'Extracted Raw Cost':<30} {raw_path_rrt:<14} {raw_path_rrtstar:<14}")
        
        print("-" * 60)
        print(f"{'RRT* Internal Improvement:':<30} {f'{impr_internal:.2f} %':<14}")
        print(f"{'RRT vs RRT* Improvement:':<30} {f'{impr_vs:.2f} %':<14}")
            
        print("=========================================================\n")
        print("DEMO RESULT — SEED " + str(active_seed))
        print(f"RRT Final Cost: {c_best_rrt}")
        print(f"RRT* First Cost: {c_first_rrtstar}")
        print(f"RRT* Final Cost: {c_best_rrtstar}")
        print(f"RRT* Internal Improvement: {impr_internal:.2f}%")
        print(f"RRT vs RRT* Improvement: {impr_vs:.2f}%")
        print(f"Rewirings: {rrtstar.rewire_count}")
        print(f"Improvement Events: {max(0, len(rrtstar.solution_improvements)-1)}")
        print("=========================================================\n")

        if presentation_mode:
            create_kpi_card(ax_kpi_1, "RRT PATH COST", c_best_rrt)
            create_kpi_card(ax_kpi_2, "RRT* PATH COST", c_best_rrtstar)
            create_kpi_card(ax_kpi_3, "LOWER PATH COST", f"{impr_vs:.2f}%", "vs RRT")
            create_kpi_card(ax_kpi_4, "RRT* INTERNAL\nOPTIMIZATION", f"{impr_internal:.2f}%")
            
            # Add final chart annotations
            if not np.isnan(rrtstar.first_solution_cost):
                ax_cost.annotate(f"First\n{rrtstar.first_solution_cost:.2f}",
                                 xy=(rrtstar.first_solution_iteration, rrtstar.first_solution_cost),
                                 xytext=(15, 10), textcoords='offset points',
                                 fontsize=10, fontweight='bold', color='blue',
                                 arrowprops=dict(arrowstyle="->", color='blue'))
                                 
            if not np.isinf(rrtstar.best_solution_cost):
                ax_cost.annotate(f"Best\n{rrtstar.best_solution_cost:.2f}",
                                 xy=(rrtstar.best_solution_iteration, rrtstar.best_solution_cost),
                                 xytext=(-15, -25), textcoords='offset points',
                                 ha='center', fontsize=10, fontweight='bold', color='purple',
                                 arrowprops=dict(arrowstyle="->", color='purple'))
            
            # Bottom Footer
            events = max(0, len(rrtstar.solution_improvements)-1)
            footer_text = f"{rrtstar.rewire_count} REWIRINGS  •  {events} IMPROVEMENT EVENTS  •  BEST SOLUTION AT ITERATION {rrtstar.best_solution_iteration}"
            fig.text(0.5, 0.02, footer_text, ha='center', fontsize=11, fontweight='bold', color='dimgray')
            
            fig.canvas.draw()
            
        if args.export_final and not exported_final:
            exported_final = True
            fname = f"rrt_rrtstar_seed{active_seed}_final.png"
            plt.savefig(fname, dpi=200, bbox_inches='tight')
            print(f"Exported final frame to {fname}")

    def animate(frame):
        global summary_printed, is_playing
        
        if not is_playing and not args.record_demo:
            return
            
        active = False
        for _ in range(steps_per_frame):
            if not rrt.is_finished or not rrtstar.is_finished:
                active = True
                idx = min(rrt.iterations, MAX_ITERATIONS - 1)
                rrt.step(random_points[idx])
                
                idx_star = min(rrtstar.iterations, MAX_ITERATIONS - 1)
                rrtstar.step(random_points[idx_star])
            else:
                break
                
        vis_rrt.update(active_seed)
        vis_rrtstar.update(active_seed)
        
        if rrt.iterations > 0:
            x_rrt = range(1, rrt.iterations + 1)
            line_rrt_cost.set_data(x_rrt, rrt.best_cost_history)
            
            x_rrtstar = range(1, rrtstar.iterations + 1)
            line_rrtstar_cost.set_data(x_rrtstar, rrtstar.best_cost_history)
            
            impr_pts = []
            first_pt = []
            best_pt = []
            
            if len(rrtstar.solution_improvements) > 0:
                first_pt = [[rrtstar.solution_improvements[0]["iteration"], rrtstar.solution_improvements[0]["cost"]]]
                if len(rrtstar.solution_improvements) > 1:
                    impr_pts = [[ev["iteration"], ev["cost"]] for ev in rrtstar.solution_improvements[1:]]
                    best_pt = [[rrtstar.solution_improvements[-1]["iteration"], rrtstar.solution_improvements[-1]["cost"]]]
                else:
                    best_pt = first_pt
                    
            if first_pt:
                scatter_first.set_offsets(first_pt)
            if impr_pts:
                scatter_impr.set_offsets(impr_pts)
            if best_pt:
                scatter_best.set_offsets(best_pt)
            
            ax_cost.set_xlim(0, max(MAX_ITERATIONS, rrt.iterations))
            valid_costs = [c for c in rrt.best_cost_history + rrtstar.best_cost_history if not np.isnan(c)]
            if valid_costs:
                ax_cost.set_ylim(max(0, min(valid_costs) - 10), max(valid_costs) + 20)
            
        if rrt.is_finished and rrtstar.is_finished and not summary_printed:
            summary_printed = True
            print_final_summary()
            
            if args.record_demo:
                ani.event_source.stop()
            elif args.export_final:
                plt.close(fig)

    if args.record_demo:
        is_playing = True
        total_frames = int(MAX_ITERATIONS / steps_per_frame) + 5
        ani = FuncAnimation(fig, animate, frames=total_frames, interval=ANIMATION_INTERVAL_MS, blit=False, cache_frame_data=False)
        try:
            mp4_name = f"rrt_vs_rrtstar_seed{active_seed}.mp4"
            print(f"Recording video to {mp4_name}. This may take a few minutes...")
            writer = FFMpegWriter(fps=30, bitrate=3000)
            ani.save(mp4_name, writer=writer, dpi=150)
            print("Recording complete.")
        except Exception as e:
            print(f"Failed to record video (FFmpeg not available?). Error: {e}")
            print("Run without --record-demo or install FFmpeg.")
    else:
        ani = FuncAnimation(fig, animate, interval=ANIMATION_INTERVAL_MS, blit=False, cache_frame_data=False)
        plt.show()

if __name__ == "__main__":
    main()
