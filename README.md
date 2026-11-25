# Fundamentos de Reinforcement Learning y Proyecto 1 (`CartPole-v1`)

Repositorio correspondiente a la primera etapa de un itinerario de aprendizaje de **Reinforcement Learning**, centrado en los fundamentos del aprendizaje por refuerzo clásico y profundo, y culminado con el entrenamiento y ajuste de un agente **PPO** en `CartPole-v1` utilizando **Gymnasium** y **Stable-Baselines3**.

## Resumen

A lo largo de esta fase he trabajado desde métodos basados en tablas Q hasta métodos basados en redes neuronales profundas y optimización de políticas:

- **`Q-Learning`:** comprensión del bucle agente-entorno, actualización por diferencia temporal (*TD*), equilibrio exploración-explotación ($\epsilon$-greedy) y limitaciones de las tablas Q cuando el espacio de estados crece o es continuo.
- **Deep Q-Network (`DQN`):** aproximación de la función de valor acción-estado con redes neuronales y visualización del comportamiento aprendido.
- **Proyecto 1 (`PPO` en `CartPole-v1`):** entrenamiento de un agente `PPO` (`MlpPolicy`) y comparación de configuraciones de hiperparámetros (`learning_rate` y `n_steps`).

## Estructura del repositorio

```text
semana-01-fundamentos-rl/
├── .gitignore
├── README.md
├── Q-Learning/
│   ├── FrozenLake/
│   │   ├── Flake.py
│   │   └── FLakeTutorial.ipynb
│   ├── CartPole/
│   │   ├── GymnasiumQ-learning.py
│   │   ├── Modify-env.py
│   │   ├── Act-ob-spaces.py
│   │   └── Q-learning.py
│   └── Blackjack/
│       ├── BlackJack.py
│       └── Notebook_BlackJack.ipynb
├── DQN/
│   └── CartPole-v1/
│       ├── cartpole_dqn.gif
│       ├── DQN_CartPole.py
│       ├── Notebook_DQNCartPole.ipynb
│       └── Notebook_DQNCartPole-Visual.ipynb
└── Project/
    └── CartPole_PPO.ipynb
```

## Detalle por módulos

### 1. `Q-Learning/`
Prácticas orientadas a asentar las bases del RL y la interacción con entornos de Gymnasium:
- **`FrozenLake/`:** implementación tabular en un entorno de cuadrícula resbaladiza/determinista (`Flake.py` y `FLakeTutorial.ipynb`).
- **`CartPole/`:** exploración de espacios de acción y observación (`Act-ob-spaces.py`), modificación de entorno (`Modify-env.py`) y pruebas de Q-Learning (`Q-learning.py`, `GymnasiumQ-learning.py`).
- **`Blackjack/`:** agente tabular aplicado a un entorno episódico con reglas de juego (`BlackJack.py` y `Notebook_BlackJack.ipynb`).

### 2. `DQN/CartPole-v1/`
Transición de tablas $Q$ a redes neuronales para estimar valores $Q(s, a)$ en `CartPole-v1`:
- Scripts y notebooks de entrenamiento y análisis (`DQN_CartPole.py`, `Notebook_DQNCartPole.ipynb`, `Notebook_DQNCartPole-Visual.ipynb`).
- Captura visual del agente entrenado en `cartpole_dqn.gif`.

### 3. `Project/CartPole_PPO.ipynb`
Proyecto práctico de la semana con **Stable-Baselines3** (`PPO` con `MlpPolicy`) en `CartPole-v1`, evaluando distintas combinaciones de tasa de aprendizaje y horizonte de rollout:
- `lr = 1e-3`, `n_steps = 512`
- `lr = 3e-4`, `n_steps = 2048`
- `lr = 1e-4`, `n_steps = 4096`

Los archivos de modelos guardados (`.zip`) y los registros de TensorBoard (`events.out.tfevents.*`) se generan en local durante los experimentos y quedan excluidos del control de versiones mediante `.gitignore`.

## Entorno y ejecución

1. Activa el entorno de trabajo:
   ```bash
   conda activate RLenv
   ```
2. Ejecuta cualquiera de los scripts o abre los notebooks:
   ```bash
   jupyter notebook
   ```
   o bien:
   ```bash
   python Q-Learning/FrozenLake/Flake.py
   ```

## Notas sobre archivos excluidos
Por higiene del repositorio y tamaño de archivos, `.gitignore` excluye carpetas temporales y artefactos pesados (`__pycache__/`, `.ipynb_checkpoints/`, copias `*-Copy*.ipynb`, `Project/models/`, `Project/tb_logs/`, `*.zip`, `*.pt`, `*.pth`, `events.out.tfevents.*`, etc.).

## Referencias y recursos consultados

Material de referencia de la Semana 1 utilizado como guía conceptual y práctica:

- **OpenAI Spinning Up in Deep RL:** [spinningup.openai.com](https://spinningup.openai.com)
- **Gymnasium Documentation & GeeksforGeeks (Q-Learning):** fundamentos del bucle agente-entorno y control tabular
- **Mnih et al. (2015):** *Human-level control through deep reinforcement learning* (*Nature*, abstract y sección 3 sobre DQN, *experience replay* y *target network*)
- **PyTorch DQN Tutorial:** [pytorch.org/tutorials](https://pytorch.org/tutorials)
- **Lilian Weng (2018):** *Policy Gradient Algorithms* — [lilianweng.github.io/posts/2018-04-08-policy-gradient](https://lilianweng.github.io/posts/2018-04-08-policy-gradient)
- **Stable-Baselines3 Documentation:** [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io)
