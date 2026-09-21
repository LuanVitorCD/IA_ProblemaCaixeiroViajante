import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import io

# ---------------------------------------------------------
# CONFIGURAÇÕES DA SESSÃO
# ---------------------------------------------------------
st.set_page_config(page_title="PCV - Algoritmo Genético", layout="wide", initial_sidebar_state="expanded")

SIZE_OPTIONS = {"Pequeno": 10, "Médio": 25, "Grande": 50}
POP_OPTIONS = {"Pequena (50)": 50, "Média (100)": 100, "Grande (250)": 250}
GEN_OPTIONS = {"Curto (100)": 100, "Médio (250)": 250, "Longo (500)": 500}
CROSS_OPTIONS = {"Baixa (70%)": 0.70, "Média (85%)": 0.85, "Alta (95%)": 0.95}
MUT_OPTIONS = {"Baixa (5%)": 0.05, "Média (15%)": 0.15, "Alta (30%)": 0.30}
K_OPTIONS = {"Baixa (2)": 2, "Média (3)": 3, "Alta (5)": 5}

DEFAULT_SIZE_LABEL = "Médio"
FRAME_DELAY = 0.05

# ---------------------------------------------------------
# FUNÇÕES DO ALGORITMO GENÉTICO
# ---------------------------------------------------------
def generate_cities(n):
    """Gera coordenadas aleatórias para N cidades."""
    return [(random.uniform(5, 95), random.uniform(5, 95)) for _ in range(n)]

def calc_distance(city1, city2):
    return math.dist(city1, city2)

def route_distance(route, cities):
    """Calcula a distância total de uma rota (incluindo retorno à origem)."""
    dist = 0
    for i in range(len(route)):
        dist += calc_distance(cities[route[i]], cities[route[(i + 1) % len(route)]])
    return dist

def init_population(pop_size, num_cities):
    """Gera a população inicial com rotas aleatórias."""
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

# SELEÇÃO
def selection_tournament(pop, fitnesses, k=3):
    """Seleciona o melhor entre K indivíduos aleatórios."""
    selected = random.sample(list(zip(pop, fitnesses)), k)
    return min(selected, key=lambda x: x[1])[0]

def selection_roulette(pop, fitnesses):
    """Roleta viciada: chances proporcionais à aptidão (1/distância)."""
    inv_fit = [1.0 / f for f in fitnesses]
    total_fit = sum(inv_fit)
    pick = random.uniform(0, total_fit)
    current = 0
    for ind, fit in zip(pop, inv_fit):
        current += fit
        if current > pick:
            return ind
    return pop[-1]

# CRUZAMENTO (CROSSOVER)
def crossover_ox(parent1, parent2):
    """Order Crossover (OX) - Preserva ordem relativa."""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]
    p2_idx, c_idx = end, end
    while -1 in child:
        if parent2[p2_idx % size] not in child:
            child[c_idx % size] = parent2[p2_idx % size]
            c_idx += 1
        p2_idx += 1
    return child

def crossover_pmx(parent1, parent2):
    """Partially Mapped Crossover (PMX)."""
    size = len(parent1)
    child = [-1] * size
    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent1[start:end]
    for i in range(start, end):
        if parent2[i] not in child:
            spot = i
            while start <= spot < end:
                val = parent1[spot]
                spot = parent2.index(val)
            child[spot] = parent2[i]
    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]
    return child

def crossover_uniforme(parent1, parent2):
    """Cruzamento Uniforme (UOX) adaptado para permutação (PCV)."""
    size = len(parent1)
    mask = [random.choice([0, 1]) for _ in range(size)]
    child = [-1] * size
    
    for i in range(size):
        if mask[i] == 1:
            child[i] = parent1[i]
            
    p2_idx = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx]
    return child

# MUTAÇÃO
def mutate_swap(route):
    """Troca duas cidades de lugar."""
    idx1, idx2 = random.sample(range(len(route)), 2)
    route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

def mutate_inversion(route):
    """Inverte um trecho inteiro da rota (muito eficaz para PCV)."""
    start, end = sorted(random.sample(range(len(route)), 2))
    route[start:end] = reversed(route[start:end])
    return route

# ---------------------------------------------------------
# ESTADOS DA SESSÃO
# ---------------------------------------------------------
if "num_cities" not in st.session_state:
    st.session_state.num_cities = SIZE_OPTIONS[DEFAULT_SIZE_LABEL]
if "cities" not in st.session_state or not st.session_state.cities:
    st.session_state.cities = generate_cities(st.session_state.num_cities)
if "best_route" not in st.session_state or not st.session_state.best_route:
    st.session_state.best_route = list(range(st.session_state.num_cities))
if "best_distance" not in st.session_state:
    st.session_state.best_distance = float('inf')
if "mutations_count" not in st.session_state:
    st.session_state.mutations_count = 0
if "generations_count" not in st.session_state:
    st.session_state.generations_count = 0
if "running" not in st.session_state:
    st.session_state.running = False

# ---------------------------------------------------------
# RENDERIZAÇÃO GRÁFICA
# ---------------------------------------------------------
def plot_route(cities, route, distance, gen):
    """Gera o gráfico estilizado em tempo real."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    if len(cities) > 0 and len(route) > 0:
        x = [cities[i][0] for i in route] + [cities[route[0]][0]]
        y = [cities[i][1] for i in route] + [cities[route[0]][1]]
        
        ax.plot(x, y, color='#5ea1ff', linewidth=2, linestyle='-', zorder=1)
        ax.scatter(x, y, color='#ff4d4d', s=80, edgecolors='white', zorder=2)
        ax.scatter(x[0], y[0], color='#00ff00', s=150, edgecolors='white', zorder=3, label="Início")

    ax.axis('off')
    plt.tight_layout()
    return fig

# ---------------------------------------------------------
# INTERFACE PRINCIPAL
# ---------------------------------------------------------
def main():
    primary_color = st.get_option("theme.primaryColor")
    
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        .block-container { padding-top: 2rem !important; padding-bottom: 0rem !important; }
        div { text-align: justify; }
        [data-testid="stImage"] { display: flex; justify-content: center; align-items: center; width: 100% !important; }
        [data-testid="stImage"] img {
            max-height: 75vh !important;
            width: auto !important;
            object-fit: contain !important;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    title_html = f"""
                <div style='background-color: #1e1e1e; padding: 5px; border-radius: 5px; border-left: 4px solid {primary_color};'>
                    <h1 style='font-size: 32px; margin-left: 10px;'><b>Problema do Caixeiro Viajante</b><br>
                        <i style='font-size: 20px; margin-left: 10px;'><b style='color: {primary_color};'>Grupo: </b> Ana, Luan e Wesley</i>
                    </h1>
                </div>
                """
    
    st.sidebar.markdown(title_html, unsafe_allow_html=True)
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    info_html = f"""
            <div style='background-color: #1e1e1e; padding: 15px; border-radius: 5px; border-left: 2px solid #5ea1ff; font-size: 16px; margin-bottom: 15px;'>
                <b style='color: #5ea1ff;'>Meta-Heurística: Algoritmo Genético</b><br>
                O PCV exato tem complexidade <b>O(N!)</b> (NP-Difícil). <br><br>
                Nesta aplicação, a IA explora apenas <b>O(População × Gerações)</b> estados, limitando drasticamente o espaço de busca e trocando a garantia da solução perfeita por uma convergência rápida e inteligente.<br><br>
            </div>
            """
    with st.sidebar.expander("Explicação e Complexidade", expanded=False, icon="ℹ️"):
        st.markdown(info_html, unsafe_allow_html=True)

    st.sidebar.divider()

    st.sidebar.subheader("Técnicas Genéticas (Principais)")
    selection_type = st.sidebar.selectbox(
        "Método de Seleção", 
        ["Torneio", "Roleta"],
        help="Define como os pais são escolhidos para gerar a próxima geração."
    )
    crossover_type = st.sidebar.selectbox(
        "Método de Cruzamento", 
        ["Uniforme", "OX (Order Crossover)", "PMX (Partially Mapped)"],
        help="Define como o material genético de dois pais é combinado."
    )
    mutation_type = st.sidebar.selectbox(
        "Método de Mutação", 
        ["Inversão (Recomendado)", "Swap (Troca Simples)"],
        help="Garante diversidade alterando pequenas características das rotas geradas."
    )
    
    dyn_selection = "<b>Torneio:</b> Escolhe o melhor entre <i>K</i> indivíduos, acelerando a convergência." if selection_type == "Torneio" else "<b>Roleta:</b> Chance proporcional à aptidão, mantendo maior diversidade genética."
    dyn_crossover = "<b>Uniforme:</b> Usa máscara binária, focado na manutenção de variedade." if crossover_type == "Uniforme" else ("<b>OX:</b> Foca em manter trechos em ordem relativa, ideal para PCV." if crossover_type == "OX (Order Crossover)" else "<b>PMX:</b> Mantém posição absoluta das cidades no array, reduzindo colisões de rota.")
    dyn_mutation = "<b>Inversão:</b> Vira uma seção da rota ao contrário. Essencial no PCV para remover loops sem quebrar a rota inteira." if mutation_type == "Inversão (Recomendado)" else "<b>Swap:</b> Troca apenas 2 cidades aleatórias. Mais lento para otimizar caminhos longos."

    dynamic_info_html = f"""
            <div style='background-color: #1e1e1e; padding: 10px; border-radius: 5px; border-left: 2px solid #5ea1ff; font-size: 14px; margin-bottom: 15px;'>
                {dyn_selection}<br><br>
                {dyn_crossover}<br><br>
                {dyn_mutation}
            </div>
            """
    
    with st.sidebar.expander("Detalhes das Técnicas", expanded=False, icon="📖"):
        st.markdown(dynamic_info_html, unsafe_allow_html=True)

    st.sidebar.divider()
    
    animar = st.sidebar.checkbox("▶️ Ativar Animação", value=True, help="Visualiza o progresso do algoritmo passo a passo.")
    start_btn = st.sidebar.button("Iniciar Evolução", type="primary", use_container_width=True)

    st.sidebar.divider()

    with st.sidebar.popover("⚙️ Configurações Avançadas", use_container_width=True):
        st.subheader("Mapa de Cidades")
        
        cfg_size = st.radio("Tamanho do Mapa", list(SIZE_OPTIONS.keys()), index=1, horizontal=True)
        
        if st.button("🎲 Gerar Novo Mapa Aleatório", use_container_width=True):
            st.session_state.num_cities = SIZE_OPTIONS[cfg_size]
            st.session_state.cities = generate_cities(st.session_state.num_cities)
            st.session_state.best_route = list(range(st.session_state.num_cities))
            st.session_state.best_distance = route_distance(st.session_state.best_route, st.session_state.cities)
            st.session_state.generations_count = 0
            st.session_state.mutations_count = 0
            st.rerun()

        st.divider()
        st.subheader("Variáveis do Algoritmo Genético")
        
        cfg_pop = st.radio("Tamanho da População", list(POP_OPTIONS.keys()), index=1, horizontal=True)
        pop_size = POP_OPTIONS[cfg_pop]

        cfg_gen = st.radio("Intervalo de Geração (Máx)", list(GEN_OPTIONS.keys()), index=1, horizontal=True)
        max_gen = GEN_OPTIONS[cfg_gen]

        cfg_cross = st.radio("Taxa de Cruzamento", list(CROSS_OPTIONS.keys()), index=1, horizontal=True)
        crossover_rate = CROSS_OPTIONS[cfg_cross]

        cfg_mut = st.radio("Taxa de Mutação", list(MUT_OPTIONS.keys()), index=1, horizontal=True)
        mutation_rate = MUT_OPTIONS[cfg_mut]
        
        k_tournament = 3
        if selection_type == "Torneio":
            cfg_k = st.radio("Tamanho do Torneio (k)", list(K_OPTIONS.keys()), index=1, horizontal=True)
            k_tournament = K_OPTIONS[cfg_k]

    # Obter a contagem final atual de cidades para o loop
    num_cities = st.session_state.num_cities
    
    col_metrics, col_chart = st.columns([1, 3])
    
    with col_metrics:
        st.subheader("Estatísticas")
        metric_dist = st.empty()
        metric_gen = st.empty()
        metric_mut = st.empty()
        
        metric_dist.metric("Melhor Distância", f"{st.session_state.best_distance:.2f}")
        metric_gen.metric("Geração Atual", st.session_state.generations_count)
        metric_mut.metric("Mutações Ocorridas", st.session_state.mutations_count)

    with col_chart:
        chart_placeholder = st.empty()
        
        fig = plot_route(st.session_state.cities, st.session_state.best_route, st.session_state.best_distance, st.session_state.generations_count)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.1)
        buf.seek(0)
        chart_placeholder.image(buf)
        plt.close(fig)

    if start_btn:
        population = init_population(pop_size, num_cities)
        best_overall_route = []
        best_overall_dist = float('inf')
        total_mutations = 0
        
        with st.spinner("Evoluindo população..."):
            for gen in range(1, max_gen + 1):
                distances = [route_distance(ind, st.session_state.cities) for ind in population]
                
                min_dist_idx = distances.index(min(distances))
                if distances[min_dist_idx] < best_overall_dist:
                    best_overall_dist = distances[min_dist_idx]
                    best_overall_route = population[min_dist_idx][:]
                    
                new_population = [best_overall_route]
                
                while len(new_population) < pop_size:
                    if selection_type == "Torneio":
                        parent1 = selection_tournament(population, distances, k_tournament)
                        parent2 = selection_tournament(population, distances, k_tournament)
                    else:
                        parent1 = selection_roulette(population, distances)
                        parent2 = selection_roulette(population, distances)
                        
                    if random.random() < crossover_rate:
                        if crossover_type == "Uniforme":
                            child = crossover_uniforme(parent1, parent2)
                        elif crossover_type == "OX (Order Crossover)":
                            child = crossover_ox(parent1, parent2)
                        else:
                            child = crossover_pmx(parent1, parent2)
                    else:
                        child = parent1[:]
                        
                    if random.random() < mutation_rate:
                        total_mutations += 1
                        if mutation_type == "Inversão (Recomendado)":
                            child = mutate_inversion(child)
                        else:
                            child = mutate_swap(child)
                            
                    new_population.append(child)
                    
                population = new_population
                
                st.session_state.best_distance = best_overall_dist
                st.session_state.best_route = best_overall_route
                st.session_state.generations_count = gen
                st.session_state.mutations_count = total_mutations

                if animar:
                    if gen % 5 == 0 or gen == max_gen:
                        metric_dist.metric("Melhor Distância", f"{best_overall_dist:.2f}", delta=f"Geração {gen}")
                        metric_gen.metric("Geração Atual", f"{gen} / {max_gen}")
                        metric_mut.metric("Mutações Ocorridas", total_mutations)
                        
                        fig = plot_route(st.session_state.cities, best_overall_route, best_overall_dist, gen)
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.1)
                        buf.seek(0)
                        chart_placeholder.image(buf)
                        plt.close(fig)
                        
                        time.sleep(FRAME_DELAY)
                            
            if not animar:
                metric_dist.metric("Melhor Distância", f"{best_overall_dist:.2f}", delta="Concluído")
                metric_gen.metric("Geração Atual", f"{max_gen} / {max_gen}")
                metric_mut.metric("Mutações Ocorridas", total_mutations)
                
                fig = plot_route(st.session_state.cities, best_overall_route, best_overall_dist, max_gen)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.1)
                buf.seek(0)
                chart_placeholder.image(buf)
                plt.close(fig)
                
            st.success(f"Evolução concluída! Melhor distância alcançada: {best_overall_dist:.2f}")

if __name__ == "__main__":
    main()
