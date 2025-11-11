import seaborn as sns
import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML


def plot_policy(probs_or_qvals, frame, action_meanings=None):
    """
    Visualiza una política o Q-values como mapa de calor con flechas.

    Crea una figura con dos paneles:
    - Izquierda: mapa de calor con la mejor acción por estado.
    - Derecha: imagen del entorno renderizado.

    Parámetros:
    - probs_or_qvals (np.ndarray): matriz (nS, nA) con probabilidades o Q, o vector (nS,)
      con acciones ya argmax. nS = rows*cols.
    - frame (np.ndarray): imagen del entorno (resultado de env.render()).
    - action_meanings (dict[int, str], opcional): mapeo índice->símbolo para anotar flechas.
      Por defecto {0: '←', 1: '↓', 2: '→', 3: '↑'}.

    Detalles:
    - (rows, cols) se infieren factorizando nS y, si hay `frame`, aprovechando su relación
      de aspecto. Soporta rejillas rectangulares (p. ej., 4x4, 4x12).

    Devuelve:
    - None. Muestra la figura con matplotlib.
    """
    # Clear any existing figures to prevent multiple plots
    plt.clf()
    plt.close('all')

    # Default action symbols if not provided
    if action_meanings is None:
        action_meanings = {0: '←', 1: '↓', 2: '→', 3: '↑'}

    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Find the best action for each state (highest probability/Q-value)
    # Works for either (nS, nA) or (nS,) inputs
    max_prob_actions = probs_or_qvals.argmax(axis=-1) if probs_or_qvals.ndim > 1 else probs_or_qvals

    # Infer grid shape (rows, cols) from nS and frame aspect ratio
    nS = int(max_prob_actions.shape[0])

    def _infer_grid_shape(n_states, frame_img):
        # Factorize n_states into (r, c) pairs (r <= c)
        candidates = []
        for r in range(1, int(np.sqrt(n_states)) + 1):
            if n_states % r == 0:
                c = n_states // r
                if r <= c:
                    candidates.append((r, c))
                else:
                    candidates.append((c, r))
        # Use frame aspect ratio if available to choose best candidate
        if frame_img is not None and hasattr(frame_img, 'shape') and len(frame_img.shape) >= 2:
            h, w = frame_img.shape[:2]
            ratio = (w / h) if h > 0 else 1.0
            best = min(candidates, key=lambda rc: abs((rc[1] / rc[0]) - ratio))
        else:
            # Fall back to the most square-ish (minimize c - r)
            best = min(candidates, key=lambda rc: (rc[1] - rc[0], -rc[0]))
        return best

    rows, cols = _infer_grid_shape(nS, frame)

    # Reshape to grid (rows, cols)
    try:
        max_prob_actions_grid = max_prob_actions.reshape(rows, cols)
    except Exception as e:
        raise ValueError(f"Could not reshape actions of length {nS} into grid {rows}x{cols}: {e}")

    # Convert action indices to readable symbols for annotation
    probs_copy = max_prob_actions_grid.copy().astype(object)
    for key in action_meanings:
        probs_copy[probs_copy == key] = action_meanings[key]

    # Create heatmap showing best actions
    sns.heatmap(
        max_prob_actions_grid,
        annot=probs_copy,
        fmt='',
        cbar=False,
        cmap='coolwarm',
        annot_kws={'weight': 'bold', 'size': 12},
        linewidths=2,
        ax=axes[0]
    )

    # Display the environment frame
    axes[1].imshow(frame)

    # Clean up axes and add title
    axes[0].axis('off')
    axes[1].axis('off')
    plt.suptitle("Policy", size=18)
    plt.tight_layout()


def plot_stochastic_policy(policy_probs, frame, action_meanings=None, show_probs=True):
    """
    Visualiza una política estocástica con flechas cuyo tamaño es proporcional a la probabilidad.

    Paneles:
    - Izquierda: por cada celda se dibujan flechas (↑, →, ↓, ←) escaladas por probabilidad.
    - Derecha: imagen del entorno renderizado.

    Parámetros:
    - policy_probs (np.ndarray): matriz (nS, nA), típicamente nA = 4 (entornos con 4 acciones).
    - frame (np.ndarray): imagen del entorno.
    - action_meanings (dict[int, str], opcional): símbolos de acciones para anotaciones.
      Por defecto {0: '↑', 1: '→', 2: '↓', 3: '←'}.
    - show_probs (bool): si True, muestra los valores numéricos.

    Notas:
    - (rows, cols) se infieren de nS; si hay `frame`, se usa su relación de aspecto para
      elegir la factorización más adecuada.

    Devuelve:
    - None.
    """
    # Clear any existing figures to prevent multiple plots
    plt.clf()
    plt.close('all')

    # Default action symbols if not provided
    if action_meanings is None:
        action_meanings = {0: '←', 1: '↓', 2: '→', 3: '↑'}

    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Infer grid shape (rows, cols) from nS and frame aspect ratio
    if policy_probs.ndim != 2:
        raise ValueError(f"policy_probs must have shape (nS, nA), got {policy_probs.shape}")
    nS, nA = policy_probs.shape

    def _infer_grid_shape(n_states, frame_img):
        candidates = []
        for r in range(1, int(np.sqrt(n_states)) + 1):
            if n_states % r == 0:
                c = n_states // r
                if r <= c:
                    candidates.append((r, c))
                else:
                    candidates.append((c, r))
        if frame_img is not None and hasattr(frame_img, 'shape') and len(frame_img.shape) >= 2:
            h, w = frame_img.shape[:2]
            ratio = (w / h) if h > 0 else 1.0
            best = min(candidates, key=lambda rc: abs((rc[1] / rc[0]) - ratio))
        else:
            best = min(candidates, key=lambda rc: (rc[1] - rc[0], -rc[0]))
        return best

    rows, cols = _infer_grid_shape(nS, frame)

    # Reshape from flat (nS, nA) to grid (rows, cols, nA)
    try:
        policy_probs_grid = policy_probs.reshape(rows, cols, nA)
    except Exception as e:
        raise ValueError(f"Could not reshape policy_probs of shape {policy_probs.shape} to ({rows},{cols},{nA}): {e}")

    # Get grid dimensions
    rows, cols = policy_probs_grid.shape[:2]

    # Define arrow directions and text positions for each action
    # Action 0: UP, Action 1: RIGHT, Action 2: DOWN, Action 3: LEFT
    arrow_config = {
        0: {'direction': (-0.25, 0), 'text_offset': (-0.35, 0), 'color': 'blue'},     # UP
        1: {'direction': (0, 0.25), 'text_offset': (0, 0.35), 'color': 'green'},      # RIGHT
        2: {'direction': (0.25, 0), 'text_offset': (0.35, 0), 'color': 'red'},        # DOWN
        3: {'direction': (0, -0.25), 'text_offset': (0, -0.35), 'color': 'orange'}    # LEFT
    }

    # Create the policy visualization
    for row in range(rows):
        for col in range(cols):
            # Get probabilities for all actions at this state
            state_probs = policy_probs_grid[row, col]

            # Check if this is a uniform policy (all actions roughly equal)
            #is_uniform = np.allclose(state_probs, 0.25, atol=0.05)

            # Draw arrows for each action with size proportional to probability
            for action in range(state_probs.shape[0]):
                if action not in arrow_config:
                    # Skip actions without a defined arrow direction (non-4-action envs)
                    continue
                prob = state_probs[action]

                # Skip very small probabilities to avoid clutter
                if prob < 0.02:
                    continue

                # Get arrow configuration
                config = arrow_config[action]
                dy, dx = config['direction']
                color = config['color']

                # Scale arrow size by probability (with better scaling)
                arrow_scale = prob * 1.2  # More pronounced scaling

                # Draw arrow with matplotlib's arrow function
                axes[0].arrow(
                    col, row,           # Starting position (x, y)
                    dx * arrow_scale,   # Arrow direction and size (dx)
                    dy * arrow_scale,   # Arrow direction and size (dy)
                    head_width=0.08 * arrow_scale + 0.03,
                    head_length=0.06 * arrow_scale + 0.02,
                    fc=color,
                    ec='black',
                    alpha=min(prob * 1.5 + 0.4, 1.0),  # Better transparency
                    linewidth=1.2
                )

                # Add probability text with better positioning
                if show_probs and prob > 0.02:  # Show for any significant probability
                    text_dy, text_dx = config['text_offset']
                    text_x = col + text_dx
                    text_y = row + text_dy

                    # Only show text if probability is significantly different from uniform
                    # or if user wants to see all probabilities
                    axes[0].text(text_x, text_y, f'{prob:.2f}', 
                                   fontsize=5, ha='center', va='center',
                                   alpha=0.9, weight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', 
                                           facecolor='white', alpha=0.8, edgecolor='none'))

    # Set up the grid
    axes[0].set_xlim(-0.6, cols - 0.4)
    axes[0].set_ylim(rows - 0.4, -0.6)  # Invert y-axis to match grid convention
    axes[0].set_aspect('equal')

    # Add grid lines for better visualization
    axes[0].set_xticks(range(cols))
    axes[0].set_yticks(range(rows))
    axes[0].grid(True, alpha=0.5, linewidth=0.5, color='gray')
    axes[0].tick_params(labelsize=10)

    # Display the environment frame
    axes[1].imshow(frame)
    axes[1].axis('off')

    # Add titles and clean up
    axes[0].set_title('Policy', fontsize=12, weight='bold')
    axes[1].set_title('Environment', fontsize=12, weight='bold')

    plt.tight_layout()


def plot_values(state_values, frame, decimal_places=2):
    """
    Visualiza los valores de estado V(s) junto a la imagen del entorno.

    Parámetros:
    - state_values (np.ndarray): vector plano (nS,) con los valores V(s), nS = rows*cols.
    - frame (np.ndarray): imagen del entorno (env.render()).
    - decimal_places (int, opcional): número de decimales en las anotaciones (por defecto 2).

    Notas:
    - (rows, cols) se infieren factorizando nS y, si hay `frame`, usando su relación de aspecto.

    Devuelve:
    - None.
    """
    # Create side-by-side subplots
    f, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Infer grid shape from nS and frame aspect ratio
    nS = int(state_values.shape[0])

    def _infer_grid_shape(n_states, frame_img):
        candidates = []
        for r in range(1, int(np.sqrt(n_states)) + 1):
            if n_states % r == 0:
                c = n_states // r
                if r <= c:
                    candidates.append((r, c))
                else:
                    candidates.append((c, r))
        if frame_img is not None and hasattr(frame_img, 'shape') and len(frame_img.shape) >= 2:
            h, w = frame_img.shape[:2]
            ratio = (w / h) if h > 0 else 1.0
            best = min(candidates, key=lambda rc: abs((rc[1] / rc[0]) - ratio))
        else:
            best = min(candidates, key=lambda rc: (rc[1] - rc[0], -rc[0]))
        return best

    rows, cols = _infer_grid_shape(nS, frame)

    # Reshape from flat (nS,) to grid (rows, cols)
    try:
        state_values_grid = state_values.reshape(rows, cols)
    except Exception as e:
        raise ValueError(f"Could not reshape state_values of length {nS} into grid {rows}x{cols}: {e}")

    # Create heatmap of state values with numerical annotations
    fmt_str = f".{int(decimal_places)}f"
    sns.heatmap(state_values_grid, annot=True, fmt=fmt_str, cmap='coolwarm',
                annot_kws={'weight': 'bold', 'size': 12}, linewidths=2, ax=axes[0])

    # Display the environment frame
    axes[1].imshow(frame)

    # Clean up axes
    axes[0].axis('off')
    axes[1].axis('off')
    plt.tight_layout()


def plot_values_rgb(state_values, frame):
    """
    Genera la misma visualización de V(s) que `plot_values`, pero como arreglo RGB.

    Esto permite usar la salida como frames para crear vídeos.

    Parámetros:
    - state_values (np.ndarray): matriz 2D (rows, cols) con V(s).
    - frame (np.ndarray): imagen del entorno.

    Devuelve:
    - np.ndarray: imagen RGB con forma (alto, ancho, 3).
    """
    # Save current matplotlib backend
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Use non-interactive backend for image generation

    try:
        # Create side-by-side subplots
        f, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Create heatmap of state values with numerical annotations
        sns.heatmap(state_values, annot=True, fmt=".2f", cmap='coolwarm',
                    annot_kws={'weight': 'bold', 'size': 12}, linewidths=2, ax=axes[0])

        # Display the environment frame
        axes[1].imshow(frame)

        # Clean up axes
        axes[0].axis('off')
        axes[1].axis('off')
        plt.tight_layout()

        # Convert figure to RGB array
        f.canvas.draw()
        buf = np.frombuffer(f.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(f.canvas.get_width_height()[::-1] + (3,))

        # Close figure to prevent memory leaks
        plt.close(f)

        return buf

    finally:
        # Restore original matplotlib backend
        matplotlib.use(orig_backend)


def display_video(frames):
    """
    Crea un video HTML5 a partir de una secuencia de frames (np.ndarray).

    Útil para visualizar el comportamiento del agente en cuadernos Jupyter.

    Parámetros:
    - frames (list[np.ndarray]): lista de imágenes a animar.

    Retorna:
    - IPython.display.HTML: elemento de video HTML5 para notebooks.

    Nota:
    - Usa un backend no interactivo de matplotlib para la generación del video.
    - Si ffmpeg no está disponible, muestra frames individuales como alternativa.
    """
    # Copied from: https://colab.research.google.com/github/deepmind/dm_control/blob/master/tutorial.ipynb

    # Import matplotlib.pyplot and animation
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import platform
    import shutil
    
    # Configure ffmpeg path based on operating system
    if platform.system() == 'Darwin':  # macOS
        # Try to find ffmpeg in common macOS locations
        ffmpeg_path = shutil.which('ffmpeg') or '/opt/homebrew/bin/ffmpeg' or '/usr/local/bin/ffmpeg'
        if ffmpeg_path:
            plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
    # For Windows and Linux, let matplotlib use its default detection
    
    # Save and temporarily change matplotlib backend for animation
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Use non-interactive backend for video creation

    # Create figure and axis for animation
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    matplotlib.use(orig_backend)  # Restore original backend

    # Configure axis for clean video display
    ax.set_axis_off()  # Remove axis labels and ticks
    ax.set_aspect('equal')  # Maintain aspect ratio
    ax.set_position([0, 0, 1, 1])  # Fill entire figure

    # Initialize with first frame
    im = ax.imshow(frames[0])

    def update(frame):
        """Update function for animation - sets new frame data."""
        im.set_data(frame)
        return [im]

    # Create animation using matplotlib's FuncAnimation
    anim = animation.FuncAnimation(
        fig=fig, func=update, frames=frames,
        interval=50,     # 50ms between frames (20 FPS)
        blit=True,       # Optimize by only redrawing changed parts
        repeat=False     # Don't loop the animation
    )

    # Convert animation to HTML5 video for notebook display
    try:
        return HTML(anim.to_html5_video())
    except RuntimeError as e:
        if "ffmpeg" in str(e).lower():
            # Try using pillow writer as fallback
            plt.rcParams['animation.writer'] = 'pillow'
            try:
                return HTML(anim.to_html5_video())
            except:
                # If all else fails, display static frames
                plt.close(fig)
                
                # Create a grid of frames
                num_frames = len(frames)
                cols = min(5, num_frames)
                rows = (num_frames + cols - 1) // cols
                
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
                if rows == 1 and cols == 1:
                    axes = [axes]
                elif rows == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()
                
                for i, frame in enumerate(frames):
                    if i < len(axes):
                        axes[i].imshow(frame)
                        axes[i].set_title(f'Frame {i+1}')
                        axes[i].axis('off')
                
                for i in range(len(frames), len(axes)):
                    axes[i].axis('off')
                
                plt.tight_layout()
                plt.show()
                
                return HTML("<p>Video display not available. Static frames shown above.</p>")
        else:
            raise e


def test_agent(environment, policy, max_episodes=10, until_goal=True):
    """
    Ejecuta varios episodios con una política dada y devuelve un video HTML5 con el comportamiento.

    Parámetros:
    - environment: entorno Gym/Gymnasium donde evaluar (crear con render_mode="rgb_array").
    - policy: función que recibe un estado y devuelve:
        - un int (acción) para políticas deterministas, o
        - un np.ndarray de probabilidades para políticas estocásticas.
    - max_episodes (int, opcional): máximo de episodios a ejecutar (por defecto 10).
    - until_goal (bool, opcional): si True, se detiene al alcanzar la recompensa objetivo (p. ej., 1.0).

    Retorna:
    - IPython.display.HTML: video del comportamiento del agente a lo largo de los episodios.

    Ejemplos:
        # Política determinista
        video = test_agent(env, lambda s: optimal_policy[s])

        # Política estocástica
        video = test_agent(env, lambda s: policy_probs[s], max_episodes=5)
    """
    frames = []  # Store rendered frames for video creation
    env = environment
    goal_achieved = False
    nA = env.action_space.n

    # Run multiple episodes to show agent behavior
    for episode in range(max_episodes):
        if goal_achieved:
            break
        # Start new episode
        state, _ = env.reset()
        done = False
        total_return = 0.0

        # Record initial state
        frames.append(env.render())

        # Run episode until completion
        while not done:
            # Get action from policy (deterministic or stochastic)
            p = policy(state)

            if isinstance(p, np.ndarray):
                # Stochastic policy: sample action according to probabilities
                action = np.random.choice(nA, p=p)
            else:
                # Deterministic policy: use action directly
                action = int(p)

            # Execute action in environment (Gymnasium API)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_return += reward

            if until_goal and total_return >= 1.0:
                done = True
                goal_achieved = True

            # Record frame after action
            img = env.render()
            frames.append(img)

            # Move to next state
            state = next_state

    # Create and return video from collected frames
    return display_video(frames)

def sample_action_from_policy(policy_probs, state, mode='greedy'):
    """
    Selecciona una acción a partir de π(a|s) de dos formas:
    - mode='greedy': elige la(s) acción(es) con mayor probabilidad (tie-break aleatorio).
    - mode='sample': muestrea de la distribución π(a|s).

    Parámetros:
    - policy_probs (np.ndarray): matriz (nS, nA) con π(a|s)
    - state (int): estado actual
    - mode (str): 'greedy' (por defecto) o 'sample'
    """
    probs = policy_probs[state]
    if mode == 'greedy':
        best_actions = np.flatnonzero(probs == probs.max())
        return int(np.random.choice(best_actions))
    elif mode == 'sample':
        return int(np.random.choice(len(probs), p=probs))
    else:
        raise ValueError("mode debe ser 'greedy' o 'sample'")
    
def test_agent2(environment, policy_probs, max_episodes=10, until_goal=True, mode="greedy"):
    """
    Igual que test_agent, pero recibe directamente la matriz de probabilidades π(a|s)
    y selecciona acciones con `sample_action_from_policy`.

    Parámetros:
    - environment: entorno Gymnasium (crear con render_mode="rgb_array").
    - policy_probs (np.ndarray): matriz (nS, nA) con probabilidades π(a|s).
    - max_episodes (int): episodios a ejecutar.
    - until_goal (bool): si True, corta cuando el retorno acumulado >= 1.0.
    - mode (str): 'greedy' (por defecto) o 'sample'.
    """
    frames = []
    env = environment
    goal_achieved = False

    for episode in range(max_episodes):
        if goal_achieved:
            break
        state, _ = env.reset()
        done = False
        total_return = 0.0

        frames.append(env.render())

        while not done:
            action = sample_action_from_policy(policy_probs, state, mode=mode)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_return += reward

            if until_goal and total_return >= 1.0:
                done = True
                goal_achieved = True

            frames.append(env.render())
            state = next_state

    return display_video(frames)


def quatromatrix(action_values, ax=None, triplotkw=None, tripcolorkw=None):
    """
    Dibuja un mosaico triangular por celda para 4 acciones.

    Parámetros:
    - action_values (np.ndarray): arreglo (rows, cols, 4) en orden URDL = [UP, RIGHT, DOWN, LEFT].
    - ax (matplotlib.axes.Axes, opcional): eje sobre el que dibujar. Si None, se crea uno.
    - triplotkw (dict, opcional): kwargs para `ax.triplot` (rejilla).
    - tripcolorkw (dict, opcional): kwargs para `ax.tripcolor` (coloreo; admite cmap, vmin, vmax, ...).

    Retorna:
    - matplotlib.collections.PolyCollection: objeto retornado por `tripcolor` (útil para colorbar).

    Notas:
    - Se invierte verticalmente con `np.flipud` para que la fila 0 quede en la parte inferior.
    - El orden de caras por celda es [left, bottom, right, top].
    """
    av = np.asarray(action_values)
    if av.ndim != 3 or av.shape[2] != 4:
        raise ValueError("action_values debe ser (rows, cols, 4).")

    av = np.flipud(av)  # para que fila 0 quede abajo en el dibujo
    n, m, _ = av.shape

    a = np.array([[0, 0], [0, 1], [0.5, 0.5], [1, 0], [1, 1]], dtype=float)
    tr = np.array([[0, 1, 2], [0, 2, 3], [2, 3, 4], [1, 2, 4]], dtype=int)

    A = np.zeros((n * m * 5, 2), dtype=float)
    Tr = np.zeros((n * m * 4, 3), dtype=int)
    for i in range(n):
        for j in range(m):
            k = i * m + j
            A[k * 5:(k + 1) * 5, :] = a + np.array([j, i])
            Tr[k * 4:(k + 1) * 4, :] = tr + k * 5

    # Caras en orden: left, bottom, right, top. Mapear desde URDL -> [L,B,R,T] = [3,2,1,0]
    C = np.c_[av[:, :, 3].ravel(),  # LEFT
              av[:, :, 2].ravel(),  # DOWN
              av[:, :, 1].ravel(),  # RIGHT
              av[:, :, 0].ravel()]  # UP
    C = C.ravel()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    triplotkw = {"color": "k", "lw": 1} if triplotkw is None else triplotkw
    tripcolorkw = {"cmap": "coolwarm"} if tripcolorkw is None else tripcolorkw

    ax.triplot(A[:, 0], A[:, 1], Tr, **triplotkw)
    tripcolor = ax.tripcolor(A[:, 0], A[:, 1], Tr, facecolors=C, **tripcolorkw)
    return tripcolor


def plot_action_values(action_values, rows=None, cols=None, frame=None,
                       input_order='LDRU',  # FrozenLake: LEFT, DOWN, RIGHT, UP
                       cmap='coolwarm', decimal_places=2,
                       text_color='w', fontsize=8,
                       show_colorbar=True,
                       triplotkw=None, tripcolorkw=None,
                       figsize=None, frame_title='Entorno', frame_cmap=None,
                       ax=None):
    """
    Visualiza Q(s,a) como mosaico triangular y, opcionalmente, muestra `frame` a la derecha.

    Entradas admitidas:
      - (nS, 4)  -> se reacomoda a (rows, cols, 4)
      - (rows, cols, 4)

    Parámetros:
      - rows, cols (int, opcional): si no se especifican y la entrada es (nS,4), se infieren
        factorizando nS.
      - frame (np.ndarray, opcional): imagen del entorno; si se pasa, se genera una figura 1x2.
      - input_order (str): 'LDRU' (por defecto, FrozenLake) o 'URDL'. Internamente se transforma a URDL.
      - cmap (str): mapa de color para los valores.
      - decimal_places (int): decimales a mostrar en las anotaciones.
      - text_color (str): color del texto de las anotaciones.
      - fontsize (int): tamaño de fuente de las anotaciones.
      - show_colorbar (bool): si mostrar barra de color.
      - triplotkw, tripcolorkw (dict): kwargs para malla/coloreo (p. ej., lw, cmap, vmin, vmax).
      - figsize (tuple, opcional): tamaño de la figura cuando `frame` no es None.
      - frame_title (str): título del panel derecho.
      - frame_cmap (str o None): cmap para `frame` si es escala de grises.
      - ax (Axes, opcional): eje existente para el panel izquierdo cuando `frame` es None.

    Devuelve:
      - Si `frame` es None: (fig, ax_left)
      - Si `frame` no es None: (fig, (ax_left, ax_right))
    """
    plt.clf()
    plt.close('all')
    
    data = np.asarray(action_values)

    # Normaliza a (rows, cols, 4)
    if data.ndim == 2:
        nS, nA = data.shape
        if nA != 4:
            raise ValueError(f"Se esperaban 4 acciones, recibido {nA}.")
        if rows is None or cols is None:
            r = int(np.sqrt(nS))
            while r > 1 and (nS % r != 0):
                r -= 1
            rows = r
            cols = nS // r
        if rows * cols != nS:
            raise ValueError(f"rows*cols = {rows*cols} no coincide con nS={nS}.")
        av = data.reshape(rows, cols, 4)
    elif data.ndim == 3:
        rows, cols, nA = data.shape
        if nA != 4:
            raise ValueError(f"Se esperaban 4 acciones, recibido {nA}.")
        av = data
    else:
        raise ValueError("action_values debe ser (nS,4) o (rows,cols,4).")

    # Reordenar al orden URDL (para alinear con quatromatrix)
    if input_order.upper() == 'LDRU':        # LEFT, DOWN, RIGHT, UP
        av_canon = av[:, :, [3, 2, 1, 0]]    # -> [UP, RIGHT, DOWN, LEFT]
    elif input_order.upper() == 'URDL':
        av_canon = av
    else:
        raise ValueError("input_order debe ser 'LDRU' o 'URDL'.")

    # Rango simétrico de color
    m = float(np.nanmax(np.abs(av_canon))) if np.isfinite(av_canon).any() else 1.0
    base_tripcolorkw = {"cmap": cmap, "vmin": -m, "vmax": m}
    if tripcolorkw:
        base_tripcolorkw.update(tripcolorkw)

    # Crear figura/axes
    if frame is None:
        created_ax = False
        if ax is None:
            fig, ax_left = plt.subplots(figsize=(max(6, cols * 1.2), max(6, rows * 1.2)))
            created_ax = True
        else:
            fig = ax.figure
            ax_left = ax
        tripcolor = quatromatrix(av_canon, ax=ax_left, triplotkw=triplotkw, tripcolorkw=base_tripcolorkw)

        ax_left.set_aspect("equal")
        ax_left.set_xlim(0, cols)
        ax_left.set_ylim(0, rows)
        ax_left.margins(0)

        if show_colorbar and (created_ax or ax is None):
            fig.colorbar(tripcolor, ax=ax_left)

        # Texto (usar misma orientación que el dibujo)
        av_disp = np.flipud(av_canon)
        offsets = [(0.50, 0.82), (0.82, 0.50), (0.50, 0.18), (0.18, 0.50)]  # URDL
        for i in range(rows):
            for j in range(cols):
                for a in range(4):
                    v = float(av_disp[i, j, a])
                    ox, oy = offsets[a]
                    x = j + ox
                    y = i + oy
                    ax_left.text(x, y, f"{v:.{int(decimal_places)}f}",
                                 color=text_color, fontsize=fontsize, weight="bold",
                                 ha="center", va="center")

        ax_left.set_title("Action values Q(s,a)", fontsize=16)
        plt.tight_layout()
        plt.show()
        return fig, ax_left

    else:
        # Con frame: crear 1x2 subplots
        if figsize is None:
            figsize = (max(7.5, cols * 1.5), max(6, rows * 1.2))
        fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [3, 2]})
        ax_left, ax_right = axes

        tripcolor = quatromatrix(av_canon, ax=ax_left, triplotkw=triplotkw, tripcolorkw=base_tripcolorkw)
        ax_left.set_aspect("equal")
        ax_left.set_xlim(0, cols)
        ax_left.set_ylim(0, rows)
        ax_left.margins(0)
        ax_left.set_title("Action values Q(s,a)", fontsize=16)

        if show_colorbar:
            fig.colorbar(tripcolor, ax=ax_left, fraction=0.046, pad=0.04)

        # Texto sobre la malla
        av_disp = np.flipud(av_canon)
        offsets = [(0.50, 0.82), (0.82, 0.50), (0.50, 0.18), (0.18, 0.50)]  # URDL
        for i in range(rows):
            for j in range(cols):
                for a in range(4):
                    v = float(av_disp[i, j, a])
                    ox, oy = offsets[a]
                    x = j + ox
                    y = i + oy
                    ax_left.text(x, y, f"{v:.{int(decimal_places)}f}",
                                 color=text_color, fontsize=fontsize, weight="bold",
                                 ha="center", va="center")

        # Panel derecho: imagen del entorno
        ax_right.imshow(frame, cmap=frame_cmap, interpolation='nearest')
        ax_right.set_title(frame_title, fontsize=14)
        ax_right.axis('off')

        plt.tight_layout()
        plt.show()
        return fig, axes