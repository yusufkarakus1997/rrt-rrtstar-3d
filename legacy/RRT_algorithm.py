import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import time

# =============================================================================
# PARAMETRELER (İstediğiniz gibi değiştirebilirsiniz)
# =============================================================================

MAP_SIZE = (100, 100, 100)           # 3D Çalışma alanının boyutları (X, Y, Z)
START = np.array([5, 5, 5])          # Başlangıç noktası koordinatları
GOAL = np.array([90, 90, 90])        # Hedef noktası koordinatları

MAX_ITERATIONS = 3000                # Maksimum iterasyon sayısı (Düğüm sayısı)
STEP_SIZE = 5.0                      # Ağacın her adımda uzama miktarı
GOAL_SAMPLE_RATE = 0.05              # Hedefe doğru rastgele nokta üretme olasılığı (%5)
SEARCH_RADIUS = 15.0                 # RRT* için yakındaki düğümleri arama yarıçapı
COLLISION_CHECK_STEP = 1.0           # Çarpışma testi için çizgi üzerindeki örnekleme aralığı
RANDOM_SEED = 42                     # Adil karşılaştırma için rastgelelik tohumu

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

def is_collision_free(p1, p2, obstacles, step_size=COLLISION_CHECK_STEP):
    """İki nokta arasındaki doğrunun herhangi bir engele çarpıp çarpmadığını kontrol eder."""
    dist = np.linalg.norm(p2 - p1)
    if dist == 0:
        return True
    
    direction = (p2 - p1) / dist
    steps = int(np.ceil(dist / step_size))
    
    for i in range(steps + 1):
        # Uç noktayı tam olarak kontrol etmek için son adımda p2'yi al
        p = p1 + i * step_size * direction if i < steps else p2
        
        # Herhangi bir engelin içinde mi?
        for (x, y, z, w, d, h) in obstacles:
            if x <= p[0] <= x + w and y <= p[1] <= y + d and z <= p[2] <= z + h:
                return False
    return True

def generate_random_points():
    """Her iki algoritmanın tam olarak aynı rastgele noktaları denemesi için noktaları önceden üretir."""
    np.random.seed(RANDOM_SEED)
    points = []
    for _ in range(MAX_ITERATIONS):
        if np.random.rand() < GOAL_SAMPLE_RATE:
            points.append(GOAL)
        else:
            points.append(np.array([
                np.random.uniform(0, MAP_SIZE[0]),
                np.random.uniform(0, MAP_SIZE[1]),
                np.random.uniform(0, MAP_SIZE[2])
            ]))
    return points

def get_path_positions(goal_node):
    """Hedef düğümden geriye doğru giderek yolun koordinatlarını çıkarır."""
    if not goal_node:
        return []
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr.position)
        curr = curr.parent
    return path[::-1] # Başlangıçtan hedefe doğru sırala

def smooth_path(path_positions, obstacles, max_iter=150):
    """Bulunan yolu rastgele kısaltmalar (shortcutting) yaparak pürüzsüzleştirir."""
    if len(path_positions) <= 2:
        return path_positions
    
    smoothed = list(path_positions)
    for _ in range(max_iter):
        if len(smoothed) <= 2:
            break
        i = np.random.randint(0, len(smoothed) - 1)
        j = np.random.randint(i + 1, len(smoothed))
        
        if j - i <= 1:
            continue
            
        if is_collision_free(smoothed[i], smoothed[j], obstacles):
            # Aradaki düğümleri atla (kestirme yap)
            smoothed = smoothed[:i+1] + smoothed[j:]
            
    return smoothed


# =============================================================================
# ALGORİTMA SINIFLARI
# =============================================================================

class Node:
    def __init__(self, position):
        self.position = position
        self.parent = None
        self.cost = 0.0
        self.children = []
        self.rewired = False # Görselleştirme için (Pembe/Cyan renk)

class BaseRRT:
    """RRT ve RRT* için ortak fonksiyonları barındıran temel sınıf."""
    def __init__(self, max_iterations=MAX_ITERATIONS):
        self.max_iterations = max_iterations
        self.iterations = 0
        self.rewirings = 0
        
        self.nodes = [Node(START)]
        # Hız optimizasyonu için pozisyonları bir numpy dizisinde tutuyoruz
        self.node_positions = np.zeros((self.max_iterations + 1, 3))
        self.node_positions[0] = START
        self.num_nodes = 1
        
        self.obstacles = OBSTACLES
        self.best_goal_node = None
        self.search_radius = SEARCH_RADIUS
        
        self.runtime = 0.0
        self.is_finished = False

    def get_nearest_node(self, random_point):
        """Ağaçtaki rastgele noktaya en yakın düğümü bulur."""
        dists = np.linalg.norm(self.node_positions[:self.num_nodes] - random_point, axis=1)
        return self.nodes[np.argmin(dists)]
        
    def steer(self, from_point, to_point):
        """Mevcut noktadan hedefe doğru STEP_SIZE kadar ilerler."""
        dist = np.linalg.norm(to_point - from_point)
        if dist < STEP_SIZE:
            return to_point
        return from_point + (to_point - from_point) * STEP_SIZE / dist
        
    def add_node(self, node):
        """Yeni düğümü ağaca ekler."""
        self.nodes.append(node)
        self.node_positions[self.num_nodes] = node.position
        self.num_nodes += 1

    def check_goal(self, node):
        """Düğümün hedefe ulaşıp ulaşmadığını kontrol eder, ulaştıysa en iyi yolu günceller."""
        dist_to_goal = np.linalg.norm(node.position - GOAL)
        if dist_to_goal <= STEP_SIZE:
            if is_collision_free(node.position, GOAL, self.obstacles):
                cost_to_goal = node.cost + dist_to_goal
                if self.best_goal_node is None or cost_to_goal < self.best_goal_node.cost:
                    if self.best_goal_node is None:
                        self.best_goal_node = Node(GOAL)
                    if self.best_goal_node.parent:
                        self.best_goal_node.parent.children.remove(self.best_goal_node)
                    self.best_goal_node.parent = node
                    self.best_goal_node.cost = cost_to_goal
                    node.children.append(self.best_goal_node)


class RRT(BaseRRT):
    """Klasik RRT Algoritması (Optimizasyon yapmaz)"""
    def step(self, random_point):
        if self.iterations >= self.max_iterations:
            self.is_finished = True
            return False
            
        t0 = time.time()
        self.iterations += 1
        
        nearest_node = self.get_nearest_node(random_point)
        new_point = self.steer(nearest_node.position, random_point)
        
        # Sadece çarpışma yoksa ekle (Klasik RRT)
        if is_collision_free(nearest_node.position, new_point, self.obstacles):
            new_node = Node(new_point)
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + np.linalg.norm(new_point - nearest_node.position)
            nearest_node.children.append(new_node)
            self.add_node(new_node)
            self.check_goal(new_node)
            
        self.runtime += (time.time() - t0)
        return True


class RRTStar(BaseRRT):
    """RRT* Algoritması (Asimptotik olarak optimal)"""
    def get_near_nodes(self, new_node):
        """Belirtilen yarıçap (SEARCH_RADIUS) içindeki tüm düğümleri getirir."""
        dists = np.linalg.norm(self.node_positions[:self.num_nodes] - new_node.position, axis=1)
        near_indices = np.where(dists <= self.search_radius)[0]
        return [self.nodes[i] for i in near_indices]

    def propagate_cost_to_leaves(self, parent_node):
        """Bir düğümün maliyeti değiştiğinde (Rewiring), bu değişimi alt düğümlerine (çocuklarına) yansıtır."""
        for child in parent_node.children:
            child.cost = parent_node.cost + np.linalg.norm(child.position - parent_node.position)
            self.propagate_cost_to_leaves(child)

    def step(self, random_point):
        if self.iterations >= self.max_iterations:
            self.is_finished = True
            return False
            
        t0 = time.time()
        self.iterations += 1
        
        nearest_node = self.get_nearest_node(random_point)
        new_point = self.steer(nearest_node.position, random_point)
        
        if not is_collision_free(nearest_node.position, new_point, self.obstacles):
            self.runtime += (time.time() - t0)
            return True
            
        new_node = Node(new_point)
        
        near_nodes = self.get_near_nodes(new_node)
        best_parent = nearest_node
        min_cost = nearest_node.cost + np.linalg.norm(new_point - nearest_node.position)
        
        # 1. CHOOSE PARENT (En İyi Ebeveyni Seç)
        for near_node in near_nodes:
            if is_collision_free(near_node.position, new_point, self.obstacles):
                cost = near_node.cost + np.linalg.norm(new_point - near_node.position)
                if cost < min_cost:
                    best_parent = near_node
                    min_cost = cost
                    
        new_node.parent = best_parent
        new_node.cost = min_cost
        best_parent.children.append(new_node)
        self.add_node(new_node)
        
        # 2. REWIRE (Ağacı Yeniden Düzenle)
        for near_node in near_nodes:
            if near_node == best_parent:
                continue
            
            # Yeni düğüm üzerinden gitmek daha ucuz mu?
            new_cost = new_node.cost + np.linalg.norm(near_node.position - new_node.position)
            if new_cost < near_node.cost:
                if is_collision_free(new_node.position, near_node.position, self.obstacles):
                    self.rewirings += 1
                    # Eski bağı kopar
                    near_node.parent.children.remove(near_node)
                    # Yeni bağ kur
                    near_node.parent = new_node
                    new_node.children.append(near_node)
                    
                    near_node.cost = new_cost
                    near_node.rewired = True # Görselde pembe yapmak için
                    self.propagate_cost_to_leaves(near_node) # Alt düğümlerin maliyetini güncelle
                    
        self.check_goal(new_node)
        self.runtime += (time.time() - t0)
        return True


# =============================================================================
# GÖRSELLEŞTİRME VE GUI SİSTEMİ
# =============================================================================

def setup_axis(ax, title):
    """3D Plot için eksen ve kamera ayarlarını yapar."""
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim([0, MAP_SIZE[0]])
    ax.set_ylim([0, MAP_SIZE[1]])
    ax.set_zlim([0, MAP_SIZE[2]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=25, azim=45) # 3D Kamera başlangıç açısı
    
    # Start ve Goal çizimi
    ax.scatter(*START, color='green', s=100, label='Start', zorder=5)
    ax.scatter(*GOAL, color='red', s=100, label='Goal', zorder=5)

def draw_obstacles(ax, obstacles):
    """Yarı saydam 3D engelleri çizer."""
    for (x, y, z, w, d, h) in obstacles:
        vertices = np.array([
            [x, y, z], [x+w, y, z], [x+w, y+d, z], [x, y+d, z],
            [x, y, z+h], [x+w, y, z+h], [x+w, y+d, z+h], [x, y+d, z+h]
        ])
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]], # alt
            [vertices[4], vertices[5], vertices[6], vertices[7]], # üst
            [vertices[0], vertices[1], vertices[5], vertices[4]], # ön
            [vertices[2], vertices[3], vertices[7], vertices[6]], # arka
            [vertices[1], vertices[2], vertices[6], vertices[5]], # sağ
            [vertices[3], vertices[0], vertices[4], vertices[7]]  # sol
        ]
        poly3d = Poly3DCollection(faces, alpha=0.3, facecolors='gray', edgecolors='k', linewidths=0.5)
        ax.add_collection3d(poly3d)

class Visualizer:
    """Bir algoritmanın 3D plotunu ve istatistik panelini yönetir."""
    def __init__(self, algo, ax, title, color_tree, color_path):
        self.algo = algo
        self.ax = ax
        self.color_tree = color_tree
        self.color_path = color_path
        
        setup_axis(self.ax, title)
        draw_obstacles(self.ax, self.algo.obstacles)
        
        self.tree_lines = Line3DCollection([], colors=self.color_tree, linewidths=0.5, alpha=0.5)
        self.ax.add_collection3d(self.tree_lines)
        
        self.path_line = Line3DCollection([], colors=self.color_path, linewidths=3.0)
        self.ax.add_collection3d(self.path_line)
        
        # Yumuşatılmış yol için (Sadece RRT* bitince çizilir)
        self.smooth_line = Line3DCollection([], colors='lime', linewidths=4.0, linestyles='solid')
        self.ax.add_collection3d(self.smooth_line)
        
        self.text_stats = self.ax.text2D(0.02, 0.95, "", transform=self.ax.transAxes, 
                                         va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        self.smoothed_drawn = False

    def update(self):
        # Ağaç dallarını güncelle
        segments = []
        colors = []
        for node in self.algo.nodes[1:]:
            segments.append([node.parent.position, node.position])
            # Rewire olan dallar pembe/magenta görünür
            if getattr(node, 'rewired', False):
                colors.append('magenta')
            else:
                colors.append(self.color_tree)
                
        self.tree_lines.set_segments(segments)
        self.tree_lines.set_color(colors)
        
        # Bulunan en iyi yolu güncelle
        if self.algo.best_goal_node:
            path_segments = []
            curr = self.algo.best_goal_node
            while curr.parent is not None:
                path_segments.append([curr.parent.position, curr.position])
                curr = curr.parent
            self.path_line.set_segments(path_segments)
            
        # Algoritma bitmişse yolu pürüzsüzleştir ve çiz (Sadece bir kez)
        if self.algo.is_finished and not self.smoothed_drawn and self.algo.best_goal_node:
            path_pos = get_path_positions(self.algo.best_goal_node)
            smooth_pos = smooth_path(path_pos, self.algo.obstacles)
            smooth_segments = [[smooth_pos[i], smooth_pos[i+1]] for i in range(len(smooth_pos)-1)]
            self.smooth_line.set_segments(smooth_segments)
            self.smoothed_drawn = True
            
        # İstatistik metnini güncelle
        status = "REACHED" if self.algo.best_goal_node else "SEARCHING..."
        cost = f"{self.algo.best_goal_node.cost:.2f}" if self.algo.best_goal_node else "N/A"
        
        stats = (f"Nodes: {self.algo.num_nodes}\n"
                 f"Iter: {self.algo.iterations} / {self.algo.max_iterations}\n"
                 f"Goal: {status}\n"
                 f"Cost: {cost}\n"
                 f"Runtime: {self.algo.runtime:.2f} s\n"
                 f"Rewirings: {self.algo.rewirings}")
        self.text_stats.set_text(stats)

# =============================================================================
# ANA UYGULAMA (MAIN)
# =============================================================================

# Global Durum Değişkenleri
is_playing = False
steps_per_frame = 10
summary_printed = False

cost_history_rrt = []
cost_history_rrtstar = []
iters = []

def main():
    global rrt, rrtstar, vis_rrt, vis_rrtstar, random_points
    global btn_play, btn_reset, slider_speed, line_rrt_cost, line_rrtstar_cost
    global ax_rrt, ax_rrtstar, ax_cost, fig

    # Rastgele noktaları önceden üret (Adil karşılaştırma için)
    random_points = generate_random_points()
    
    # Figure Düzeni (GridSpec)
    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title("RRT vs RRT* 3D Path Planning Simulator")
    gs = fig.add_gridspec(20, 20)
    
    # Alt Grafikler
    ax_rrt = fig.add_subplot(gs[0:13, 0:10], projection='3d')
    ax_rrtstar = fig.add_subplot(gs[0:13, 10:20], projection='3d')
    ax_cost = fig.add_subplot(gs[14:18, 5:15])
    
    # Algoritmalar ve Görselleştiriciler
    rrt = RRT()
    rrtstar = RRTStar()
    
    vis_rrt = Visualizer(rrt, ax_rrt, "RRT (Klasik)", "blue", "orange")
    vis_rrtstar = Visualizer(rrtstar, ax_rrtstar, "RRT* (Asimptotik Optimal)", "purple", "yellow")
    
    # Maliyet Grafiği Ayarları
    ax_cost.set_title("Path Cost vs Iteration", fontweight='bold')
    ax_cost.set_xlabel("Iteration")
    ax_cost.set_ylabel("Path Cost")
    ax_cost.grid(True, linestyle='--', alpha=0.7)
    line_rrt_cost, = ax_cost.plot([], [], color='orange', label='RRT', linewidth=2)
    line_rrtstar_cost, = ax_cost.plot([], [], color='green', label='RRT*', linewidth=2)
    ax_cost.legend()
    
    # Eğitim Paneli
    ax_edu = fig.add_axes([0.02, 0.02, 0.40, 0.15])
    ax_edu.axis('off')
    edu_text = (
        "EĞİTİM MODU:\n"
        "• RRT (Mavi): Rastgele uzay keşfi yapar. İlk bulduğu yol kullanılır, maliyet optimizasyonu yoktur.\n"
        "• RRT* (Mor): 'Choose Parent' (En ucuz ebeveyni seç) işlemi yapar.\n"
        "• Rewiring (Pembe): Ağaç büyüdükçe eski yolları daha ucuz alternatiflerle değiştirir (Yeniden bağlar).\n"
        "• Path Cost Grafiği: RRT*'ın zamanla maliyeti nasıl düşürdüğünü ve optimale yakınsadığını gösterir.\n"
        "• Yumuşatılmış Yol (Açık Yeşil): Algoritma bitince kısaltmalar (shortcutting) yapılarak elde edilir."
    )
    ax_edu.text(0, 0, edu_text, fontsize=9, va='bottom', 
                bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
    
    # GUI Kontrolleri (Butonlar ve Slider)
    ax_play = fig.add_axes([0.48, 0.05, 0.1, 0.05])
    ax_reset = fig.add_axes([0.60, 0.05, 0.1, 0.05])
    ax_speed = fig.add_axes([0.75, 0.05, 0.2, 0.03])
    
    btn_play = Button(ax_play, 'Start / Pause')
    btn_reset = Button(ax_reset, 'Reset')
    slider_speed = Slider(ax_speed, 'Speed', 1, 100, valinit=steps_per_frame, valstep=1)
    
    # Buton Callbacks
    def toggle_play(event):
        global is_playing
        is_playing = not is_playing
        btn_play.label.set_text("Pause" if is_playing else "Start")

    def reset_sim(event):
        global is_playing, rrt, rrtstar, vis_rrt, vis_rrtstar, summary_printed
        global cost_history_rrt, cost_history_rrtstar, iters
        
        is_playing = False
        btn_play.label.set_text("Start")
        summary_printed = False
        
        rrt = RRT()
        rrtstar = RRTStar()
        
        ax_rrt.cla()
        ax_rrtstar.cla()
        vis_rrt = Visualizer(rrt, ax_rrt, "RRT (Klasik)", "blue", "orange")
        vis_rrtstar = Visualizer(rrtstar, ax_rrtstar, "RRT* (Asimptotik Optimal)", "purple", "yellow")
        
        cost_history_rrt.clear()
        cost_history_rrtstar.clear()
        iters.clear()
        line_rrt_cost.set_data([], [])
        line_rrtstar_cost.set_data([], [])
        fig.canvas.draw()

    def update_speed(val):
        global steps_per_frame
        steps_per_frame = int(val)
        
    btn_play.on_clicked(toggle_play)
    btn_reset.on_clicked(reset_sim)
    slider_speed.on_changed(update_speed)
    
    # Sonuçları Konsola Yazdıran Fonksiyon
    def print_summary():
        print("\n" + "="*45)
        print(" "*12 + "RRT vs RRT* SONUÇLARI")
        print("="*45)
        print(f"{'Metrik':<16} | {'RRT':<12} | {'RRT*':<12}")
        print("-" * 45)
        print(f"{'İterasyon':<16} | {rrt.iterations:<12} | {rrtstar.iterations:<12}")
        print(f"{'Düğüm Sayısı':<16} | {rrt.num_nodes:<12} | {rrtstar.num_nodes:<12}")
        
        cost_rrt = f"{rrt.best_goal_node.cost:.2f}" if rrt.best_goal_node else "Bulunamadı"
        cost_rrtstar = f"{rrtstar.best_goal_node.cost:.2f}" if rrtstar.best_goal_node else "Bulunamadı"
        print(f"{'Path Cost':<16} | {cost_rrt:<12} | {cost_rrtstar:<12}")
        
        print(f"{'Çalışma Süresi':<16} | {rrt.runtime:.2f} s     | {rrtstar.runtime:.2f} s")
        print(f"{'Rewiring Sayısı':<16} | {rrt.rewirings:<12} | {rrtstar.rewirings:<12}")
        print("="*45 + "\n")

    # Animasyon Döngüsü
    def animate(frame):
        global summary_printed
        if not is_playing:
            return
            
        active = False
        # Speed slider'a göre her karede birden fazla iterasyon atla
        for _ in range(steps_per_frame):
            if rrt.iterations < MAX_ITERATIONS or rrtstar.iterations < MAX_ITERATIONS:
                active = True
                pt = random_points[min(rrt.iterations, MAX_ITERATIONS - 1)]
                if rrt.iterations < MAX_ITERATIONS:
                    rrt.step(pt)
                    
                pt_star = random_points[min(rrtstar.iterations, MAX_ITERATIONS - 1)]
                if rrtstar.iterations < MAX_ITERATIONS:
                    rrtstar.step(pt_star)
            else:
                break
                
        # Grafikleri Güncelle
        vis_rrt.update()
        vis_rrtstar.update()
        
        # Maliyet Grafiğini Güncelle
        iters.append(rrt.iterations)
        c_rrt = rrt.best_goal_node.cost if rrt.best_goal_node else np.nan
        c_rrtstar = rrtstar.best_goal_node.cost if rrtstar.best_goal_node else np.nan
        
        cost_history_rrt.append(c_rrt)
        cost_history_rrtstar.append(c_rrtstar)
        
        line_rrt_cost.set_data(iters, cost_history_rrt)
        line_rrtstar_cost.set_data(iters, cost_history_rrtstar)
        
        if len(iters) > 0:
            ax_cost.set_xlim(0, max(MAX_ITERATIONS, iters[-1]))
        valid_costs = [c for c in cost_history_rrt + cost_history_rrtstar if not np.isnan(c)]
        if valid_costs:
            ax_cost.set_ylim(max(0, min(valid_costs) - 10), max(valid_costs) + 20)
            
        # Simülasyon Bittiğinde Sonuçları Yazdır
        if not active and not summary_printed:
            summary_printed = True
            print_summary()

    ani = FuncAnimation(fig, animate, interval=30, blit=False, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    main()
