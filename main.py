import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

# ---------------------------------------------------------
# CONFIGURAÇÕES E ESTADOS DA SESSÃO
# ---------------------------------------------------------
st.set_page_config(page_title="TSP - Algoritmo Genético", layout="wide", initial_sidebar_state="expanded")

if "cities" not in st.session_state:
    st.session_state.cities = []
if "best_route" not in st.session_state:
    st.session_state.best_route = []
if "best_distance" not in st.session_state:
    st.session_state.best_distance = float('inf')
if "mutations_count" not in st.session_state:
    st.session_state.mutations_count = 0
if "generations_count" not in st.session_state:
    st.session_state.generations_count = 0
if "running" not in st.session_state:
    st.session_state.running = False

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
    """Cruzamento Uniforme (UOX) adaptado para permutação (TSP)."""
    size = len(parent1)
    mask = [random.choice([0, 1]) for _ in range(size)]
    child = [-1] * size
    
    # Passo 1: Herda do pai 1 onde a máscara é 1
    for i in range(size):
        if mask[i] == 1:
            child[i] = parent1[i]
            
    # Passo 2: Preenche os espaços vazios com os genes do pai 2 (preservando a ordem)
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
    """Inverte um trecho inteiro da rota (muito eficaz para TSP)."""
    start, end = sorted(random.sample(range(len(route)), 2))
    route[start:end] = reversed(route[start:end])
    return route

# ---------------------------------------------------------
# RENDERIZAÇÃO GRÁFICA
# ---------------------------------------------------------
def plot_route(cities, route, distance, gen):
    """Gera o gráfico estilizado em tempo real."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0e1117') # Cor de fundo do Streamlit
    ax.set_facecolor('#0e1117')
    
    if len(cities) > 0 and len(route) > 0:
        x = [cities[i][0] for i in route] + [cities[route[0]][0]]
        y = [cities[i][1] for i in route] + [cities[route[0]][1]]
        
        # Desenhar linha neon azulada
        ax.plot(x, y, color='#5ea1ff', linewidth=2, linestyle='-', zorder=1)
        
        # Desenhar cidades
        ax.scatter(x, y, color='#ff4d4d', s=80, edgecolors='white', zorder=2)
        
        # Destacar ponto de origem (Verde)
        ax.scatter(x[0], y[0], color='#00ff00', s=150, edgecolors='white', zorder=3, label="Início")

    ax.set_title(f"Geração: {gen} | Menor Distância: {distance:.2f}", color='white', fontsize=16, pad=15)
    ax.axis('off') # Esconde os eixos para ficar bonito
    plt.tight_layout()
    return fig

# ---------------------------------------------------------
# INTERFACE PRINCIPAL
# ---------------------------------------------------------
def main():
    primary_color = st.get_option("theme.primaryColor")
    
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display:none;}
            .block-container {
                padding-top: 2rem !important;
                padding-bottom: 0rem !important;
            }
            div {
            text-align: justify;
        }
        </style>
        """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    title_html = f"""
                <div style='background-color: #1e1e1e; padding: 5px; border-radius: 5px; border-left: 4px solid {primary_color};'>
                    <h1 style='font-size: 32px; margin-left: 10px;'><b>Caixeiro Viajante</b><br>
                        <i style='font-size: 20px; margin-left: 10px;'><b style='color: {primary_color};'>Grupo: </b> Ana, Luan e Wesley</i>
                    </h1>
                </div>
                """
    
    st.sidebar.markdown(title_html, unsafe_allow_html=True)

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    info_html = f"""
            <div style='background-color: #1e1e1e; padding: 15px; border-radius: 5px; border-left: 2px solid #5ea1ff; font-size: 16px; margin-bottom: 15px;'>
                <b style='color: #5ea1ff;'>Meta-Heurística: Algoritmo Genético</b><br>
                O TSP exato tem complexidade <b>O(N!)</b> (NP-Difícil). <br><br>
                Nesta aplicação, a IA explora apenas <b>O(População × Gerações)</b> estados, limitando drasticamente o espaço de busca e trocando a garantia da solução perfeita por uma convergência rápida e inteligente.<br><br>
            </div>
            """
    with st.sidebar.expander("Explicação & Complexidade", expanded=False, icon="ℹ️"):
        st.markdown(info_html, unsafe_allow_html=True)

    st.sidebar.divider()

    st.sidebar.subheader("Técnicas Genéticas (Principais)")
    selection_type = st.sidebar.selectbox("Método de Seleção", ["Torneio", "Roleta"])
    crossover_type = st.sidebar.selectbox("Método de Cruzamento", ["Uniforme", "OX (Order Crossover)", "PMX (Partially Mapped)"])
    mutation_type = st.sidebar.selectbox("Método de Mutação", ["Inversão (Recomendado)", "Swap (Troca Simples)"])

    st.sidebar.divider()

    # ---------------------------------------------------------
    # CORAÇÃO DO SCRIPT - EXECUÇÃO
    # ---------------------------------------------------------
    start_btn = st.sidebar.button("Iniciar Evolução", type="primary", use_container_width=True)

    st.sidebar.divider()

    with st.sidebar.popover("⚙️ Configurações Avançadas", use_container_width=True):
        st.subheader("Mapa de Cidades")
        
        num_cities = st.slider("Quantidade de Cidades", min_value=5, max_value=100, value=25)
        
        # Controle do Mapa
        if len(st.session_state.cities) != num_cities or st.button("🎲 Gerar Novo Mapa Aleatório", use_container_width=True):
            st.session_state.cities = generate_cities(num_cities)
            st.session_state.best_route = list(range(num_cities))
            st.session_state.best_distance = route_distance(st.session_state.best_route, st.session_state.cities)
            st.session_state.generations_count = 0
            st.session_state.mutations_count = 0

        st.divider()
        st.subheader("Variáveis do Algoritmo Genético")
        
        pop_size = st.number_input("Tamanho da População", min_value=10, max_value=1000, value=100, step=10)
        max_gen = st.number_input("Intervalo de Geração (Máx)", min_value=10, max_value=2000, value=200, step=10)
        crossover_rate = st.slider("Taxa de Cruzamento (%)", 0, 100, 90) / 100.0
        mutation_rate = st.slider("Taxa de Mutação (%)", 0, 100, 15) / 100.0
        
        st.markdown("**Aptidão (Fitness):** Calculada como `1 / Distância Total`. Rotas mais curtas ganham maior probabilidade.")
        
        k_tournament = 3
        if selection_type == "Torneio":
            k_tournament = st.slider("Tamanho do Torneio (k)", min_value=2, max_value=10, value=3, help="Define a pressão seletiva. K maior = maior chance de apenas os melhores serem escolhidos.")
        else:
            st.info("A seleção por **Roleta** usa probabilidade estrita baseada na aptidão. Não há parâmetros extras para ajustar.")
            
        if crossover_type == "Uniforme":
            st.info("O **Cruzamento Uniforme** para TSP (UOX) utiliza uma máscara binária aleatória para mesclar as cidades. Evita cidades duplicadas preenchendo os espaços vazios seguindo a ordem do segundo pai.")
            
        if mutation_type == "Swap (Troca Simples)":
            st.warning("A Mutação **Swap** é geralmente mais fraca para o TSP do que a Inversão, pois destrói conexões úteis na rota repetidas vezes de forma ineficiente.")

        st.divider()
        st.subheader("Opções de Performance")
        animar = st.toggle("Visualizar Animação (Evolução em Tempo Real)", value=True)
        fps = 0.0
        if animar:
            fps = st.slider("Velocidade da Animação (Atraso seg)", 0.0, 0.5, 0.05, step=0.01)
    
    # Área de visualização à direita
    col_metrics, col_chart = st.columns([1, 3])
    
    with col_chart:
        chart_placeholder = st.empty()
        # Mostra o mapa inicial
        chart_placeholder.pyplot(plot_route(st.session_state.cities, st.session_state.best_route, st.session_state.best_distance, st.session_state.generations_count))
        
    with col_metrics:
        st.subheader("Estatísticas")
        metric_dist = st.empty()
        metric_gen = st.empty()
        metric_mut = st.empty()
        
        metric_dist.metric("Melhor Distância", f"{st.session_state.best_distance:.2f}")
        metric_gen.metric("Geração Atual", st.session_state.generations_count)
        metric_mut.metric("Mutações Ocorridas", st.session_state.mutations_count)

    # LOOP PRINCIPAL DA IA
    if start_btn:
        population = init_population(pop_size, num_cities)
        best_overall_route = []
        best_overall_dist = float('inf')
        total_mutations = 0
        
        with st.spinner("Evoluindo população..."):
            for gen in range(1, max_gen + 1):
                # 1. Aptidão
                distances = [route_distance(ind, st.session_state.cities) for ind in population]
                
                # Elitismo: Salvar o melhor da geração
                min_dist_idx = distances.index(min(distances))
                if distances[min_dist_idx] < best_overall_dist:
                    best_overall_dist = distances[min_dist_idx]
                    best_overall_route = population[min_dist_idx][:]
                    
                new_population = [best_overall_route] # Mantém o melhor absoluto na nova geração
                
                # 2. Reprodução (Seleção + Crossover + Mutação)
                while len(new_population) < pop_size:
                    # Seleção
                    if selection_type == "Torneio":
                        parent1 = selection_tournament(population, distances, k_tournament)
                        parent2 = selection_tournament(population, distances, k_tournament)
                    else:
                        parent1 = selection_roulette(population, distances)
                        parent2 = selection_roulette(population, distances)
                        
                    # Crossover
                    if random.random() < crossover_rate:
                        if crossover_type == "Uniforme":
                            child = crossover_uniforme(parent1, parent2)
                        elif crossover_type == "OX (Order Crossover)":
                            child = crossover_ox(parent1, parent2)
                        else:
                            child = crossover_pmx(parent1, parent2)
                    else:
                        child = parent1[:]
                        
                    # Mutação
                    if random.random() < mutation_rate:
                        total_mutations += 1
                        if mutation_type == "Inversão (Recomendado)":
                            child = mutate_inversion(child)
                        else:
                            child = mutate_swap(child)
                            
                    new_population.append(child)
                    
                population = new_population
                
                # Atualizar Interface e Session State
                st.session_state.best_distance = best_overall_dist
                st.session_state.best_route = best_overall_route
                st.session_state.generations_count = gen
                st.session_state.mutations_count = total_mutations

                # Controle de Performance: Renderiza a cada 5 gerações na animação, ou apenas no final
                if animar:
                    if gen % 5 == 0 or gen == max_gen:
                        metric_dist.metric("Melhor Distância", f"{best_overall_dist:.2f}", delta=f"Geração {gen}")
                        metric_gen.metric("Geração Atual", f"{gen} / {max_gen}")
                        metric_mut.metric("Mutações Ocorridas", total_mutations)
                        
                        fig = plot_route(st.session_state.cities, best_overall_route, best_overall_dist, gen)
                        chart_placeholder.pyplot(fig)
                        plt.close(fig) # Liberar memória
                        if fps > 0:
                            time.sleep(fps)
                            
            # Renderização Final Obrigatória
            if not animar:
                metric_dist.metric("Melhor Distância", f"{best_overall_dist:.2f}", delta="Concluído")
                metric_gen.metric("Geração Atual", f"{max_gen} / {max_gen}")
                metric_mut.metric("Mutações Ocorridas", total_mutations)
                
                fig = plot_route(st.session_state.cities, best_overall_route, best_overall_dist, max_gen)
                chart_placeholder.pyplot(fig)
                plt.close(fig)
                
            st.success(f"Evolução concluída! Melhor distância alcançada: {best_overall_dist:.2f}")

if __name__ == "__main__":
    main()
    