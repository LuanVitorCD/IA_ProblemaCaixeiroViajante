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
st.set_page_config(page_title="PCV - Algoritmo Genético", layout="wide", page_icon="🧬", initial_sidebar_state="expanded")

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

def calculate_fitness(distance):
    """
    Transforma a distância (minimização) em Aptidão/Fitness (maximização).
    Soluções com menor distância terão maior aptidão.
    """
    if distance == 0: 
        return float('inf')
    return 10000.0 / distance

def init_population(pop_size, num_cities):
    """Gera a população inicial com rotas aleatórias (Cromossomos)."""
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

# SELEÇÃO
def selection_tournament(pop, fitnesses, k=3):
    """Torneio: Seleciona aleatoriamente k indivíduos e escolhe o de maior Aptidão."""
    selected = random.sample(list(zip(pop, fitnesses)), k)
    # Como a aptidão (fitness) maior é melhor, pegamos o máximo
    return max(selected, key=lambda x: x[1])[0]

def selection_roulette(pop, fitnesses):
    """Roleta Tradicional: Chance proporcional à Aptidão (Fitness) do indivíduo."""
    total_fit = sum(fitnesses)
    pick = random.uniform(0, total_fit)
    current = 0
    for ind, fit in zip(pop, fitnesses):
        current += fit
        if current > pick:
            return ind
    return pop[-1]

def selection_linear_ranking(pop, fitnesses):
    """
    Ranking Linear: Ordena pela aptidão e dá probabilidade baseada na posição.
    Minimiza problemas de 'Roleta Viciada' quando um indivíduo é absurdamente melhor que o resto.
    """
    pop_fit = list(zip(pop, fitnesses))
    # Ordena do pior (menor fitness) pro melhor (maior fitness)
    pop_fit.sort(key=lambda x: x[1])
    
    n = len(pop)
    # Atribui pesos lineares: o pior ganha peso 1, o segundo 2... o melhor ganha peso N
    weights = list(range(1, n + 1))
    total_weight = sum(weights)
    
    pick = random.uniform(0, total_weight)
    current = 0
    for i, weight in enumerate(weights):
        current += weight
        if current > pick:
            return pop_fit[i][0]
    return pop_fit[-1][0]

def selection_truncated(pop, fitnesses, fraction=0.5):
    """
    Truncada: Apenas uma fração (ex: 50%) dos melhores participa da seleção aleatória.
    """
    pop_fit = list(zip(pop, fitnesses))
    # Ordena do melhor pro pior (maior fitness pro menor)
    pop_fit.sort(key=lambda x: x[1], reverse=True)
    
    # Pega apenas os melhores
    cut_idx = max(1, int(len(pop) * fraction))
    elite_pool = [ind for ind, fit in pop_fit[:cut_idx]]
    
    # Escolhe um aleatório dessa elite
    return random.choice(elite_pool)

# CRUZAMENTO (CROSSOVER)
def crossover_um_ponto_correcao(parent1, parent2):
    """
    Cruzamento de 1 Ponto com Correção.
    Corta no meio, pega a primeira metade do Pai 1. 
    Para a segunda metade, usa a ordem do Pai 2 para preencher as cidades que faltam.
    """
    size = len(parent1)
    point = size // 2 
    
    child = [-1] * size
    
    # Primeira metade do Pai 1
    child[:point] = parent1[:point]
    
    # Cidades que ainda não estão no filho
    missing = [c for c in parent2 if c not in child]
    
    # Preenche o resto com as cidades faltantes mantendo a ordem que aparecem no Pai 2
    m_idx = 0
    for i in range(point, size):
        child[i] = missing[m_idx]
        m_idx += 1
            
    return child

def crossover_uniforme(parent1, parent2):
    """Cruzamento Uniforme (UOX) - Usa máscara binária e corrige duplicatas."""
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
    """Mutação Swap (Troca): Altera genes trocando duas cidades de lugar."""
    idx1, idx2 = random.sample(range(len(route)), 2)
    route[idx1], route[idx2] = route[idx2], route[idx1]
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

    if st.session_state.get("reset_requested", False):
        st.session_state.map_size = "Médio"
        st.session_state.num_cities = SIZE_OPTIONS["Médio"]
        st.session_state.cities = generate_cities(st.session_state.num_cities)
        st.session_state.best_route = list(range(st.session_state.num_cities))
        st.session_state.best_distance = route_distance(st.session_state.best_route, st.session_state.cities)
        st.session_state.generations_count = 0
        st.session_state.mutations_count = 0
        st.session_state.reset_requested = False

    primary_color = st.get_option("theme.primaryColor")
    
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        .block-container { padding-top: 2rem !important; padding-bottom: 0rem !important; }
        div { text-align: left; }
        [data-testid="stImage"] { display: flex; justify-content: center; align-items: center; width: 100% !important; }
        [data-testid="stImage"] img {
            max-height: 80vh !important;
            width: 80vw !important;
            object-fit: contain !important;
            border-radius: 8px;
        }
        section[data-testid="stSidebar"] {
            width: 500px !important;
        }
        section[data-testid="stSidebar"] > div:not([data-testid="stSidebarContent"]) {
            display: none !important;
            width: 0px !important;
            pointer-events: none !important;
        }
        div[data-testid="stSidebarCollapseButton"] {
            display: inline !important;
            visibility: inline !important;
        }
        button[data-testid="stPopoverButton"]{
            border: 1px solid rgba(57, 119, 255, 0.5) !important;
            border-radius: 10px !important;
            justify-content: space-between;
        }
        button[data-testid="stPopoverButton"] > div[data-testid="stMarkdownContainer"] {
            width: 100%;
        }
        span[data-testid="stIconMaterial"]{
            color: rgb(57, 119, 255) !important;
        }
        [data-testid="stMetric"]{
            background-color: #191c20;
            padding: 15px;
            border-radius: 5px;
            font-size: 16px;
            margin-bottom: 15px;
        }
        [data-testid="stMetricLabel"]{
            font-size: 20px;
        }
        div[data-testid="stMetricValue"]{
            font-size: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    title_html = f"""
                <div style='background-color: #191c20; padding: 5px; border-radius: 5px; border-left: 4px solid {primary_color};'>
                    <h1 style='font-size: 32px; margin-left: 10px;'><b>Problema do Caixeiro Viajante</b><br>
                        <i style='font-size: 20px; margin-left: 10px;'><b style='color: {primary_color};'>Grupo: </b> Ana, Luan e Wesley</i>
                    </h1>
                </div>
                """
    
    st.sidebar.markdown(title_html, unsafe_allow_html=True)
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # Texto de Explicação + Informação sobre a Mutação Fixa
    info_html = f"""
            <div style='background-color: #191c20; padding: 15px; border-radius: 5px; border-left: 2px solid #5ea1ff; font-size: 16px; margin-bottom: 15px;'>
                <b style='color: #5ea1ff;'>Meta-Heurística: Algoritmo Genético</b><br>
                O PCV exato tem complexidade <b>O(N!)</b> (NP-Difícil). <br><br>
                Nesta aplicação, a IA explora apenas <b>O(População × Gerações)</b> estados, limitando drasticamente o espaço de busca e trocando a garantia da solução perfeita por uma convergência rápida e inteligente.<br><br>
                <b>🔄 Sobre a Mutação:</b><br>
                Para manter alinhamento aos conceitos base, esta aplicação fixa o método de mutação no formato <b>Swap (Troca Simples)</b>. A cada geração, há uma chance de ocorrer uma troca de posição entre duas cidades aleatórias da rota, alterando seus genes e garantindo a diversidade genética da população.
            </div>
            """
    with st.sidebar.popover("Explicação e Complexidade", icon=":material/chat_info:", use_container_width=True):
        st.markdown(info_html, unsafe_allow_html=True)

    # Divisor customizado com margens reduzidas (10px em cima e em baixo)
    custom_divider = """
        <hr style="margin-top: 10px; margin-bottom: 25px; border: none; border-top: 1px solid rgba(255, 255, 255, 0.2);">
    """

    st.sidebar.markdown(custom_divider, unsafe_allow_html=True)

    if "map_size" not in st.session_state:
        st.session_state.map_size = "Médio"

    with st.sidebar.form("map_config_form"):
        st.subheader("Mapa de Cidades")
        cfg_size = st.radio("Tamanho do Mapa", list(SIZE_OPTIONS.keys()), key="map_size", horizontal=True)

        col_apply, col_reset = st.columns(2)    
        aplicar = col_apply.form_submit_button(":material/check: Aplicar e Gerar", type="primary", use_container_width=True)
        resetar = col_reset.form_submit_button("↺ Resetar", use_container_width=True)

    if aplicar:
        st.session_state.num_cities = SIZE_OPTIONS[cfg_size]
        st.session_state.cities = generate_cities(st.session_state.num_cities)
        st.session_state.best_route = list(range(st.session_state.num_cities))
        st.session_state.best_distance = route_distance(st.session_state.best_route, st.session_state.cities)
        st.session_state.generations_count = 0
        st.session_state.mutations_count = 0
        st.rerun()
    elif resetar:
        st.session_state.reset_requested = True
        st.rerun()

    selection_type = st.sidebar.selectbox(
        "Método de Seleção", 
        ["Torneio", "Roleta", "Ranking Linear", "Truncada"],
        help="Define como os pais são escolhidos para gerar a próxima geração. Usa a Aptidão (Fitness)."
    )
    crossover_type = st.sidebar.selectbox(
        "Método de Cruzamento (Crossover)", 
        ["1 Ponto (Com Correção)", "Uniforme"],
        help="Define como o material genético (genes) de dois pais é combinado para gerar o filho."
    )

    ativar_elitismo = st.sidebar.toggle( ":primary[:material/crown:] Ativar Elitismo (Manter melhor indivíduo)", value=True, help="Garante que a melhor solução da geração anterior passe direto para a próxima geração sem sofrer mutação ou cruzamento.")

    animar = st.sidebar.toggle(":primary[:material/slideshow:] Ativar Animação", value=True, help="Visualiza o progresso do algoritmo passo a passo.")

    with st.sidebar.popover(":primary[:material/settings:] Configurações Avançadas", use_container_width=True):
        st.subheader("Variáveis do Algoritmo Genético")
        
        cfg_pop = st.radio("Tamanho da População", list(POP_OPTIONS.keys()), index=1, horizontal=True)
        pop_size = POP_OPTIONS[cfg_pop]

        cfg_gen = st.radio("Limite de Gerações (Critério de Parada)", list(GEN_OPTIONS.keys()), index=1, horizontal=True)
        max_gen = GEN_OPTIONS[cfg_gen]

        cfg_cross = st.radio("Taxa de Cruzamento", list(CROSS_OPTIONS.keys()), index=1, horizontal=True)
        crossover_rate = CROSS_OPTIONS[cfg_cross]

        cfg_mut = st.radio("Taxa de Mutação", list(MUT_OPTIONS.keys()), index=1, horizontal=True)
        mutation_rate = MUT_OPTIONS[cfg_mut]
        
        k_tournament = 3
        if selection_type == "Torneio":
            cfg_k = st.radio("Tamanho do Torneio (K)", list(K_OPTIONS.keys()), index=1, horizontal=True)
            k_tournament = K_OPTIONS[cfg_k]

        st.markdown(custom_divider, unsafe_allow_html=True)

        st.subheader("Critérios de Parada Extras")
        distancia_alvo = st.number_input("Distância Alvo (Solução Satisfatória)", value=0.0, step=10.0, help="O algoritmo encerra se encontrar uma 'Solução Satisfatória' (menor ou igual a este valor). Deixe 0.0 para ignorar e rodar até o Limite de Gerações.")

    st.sidebar.markdown(custom_divider, unsafe_allow_html=True)
    
    start_btn = st.sidebar.button("Iniciar Evolução", type="primary", use_container_width=True)

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
        # INÍCIO: Gerar População Inicial
        population = init_population(pop_size, num_cities)
        best_overall_route = []
        best_overall_dist = float('inf')
        total_mutations = 0
        
        with st.spinner("Evoluindo população..."):
            for gen in range(1, max_gen + 1):
                # AVALIAÇÃO: Calcular Custo (Distância) e Aptidão (Fitness)
                distances = [route_distance(ind, st.session_state.cities) for ind in population]
                fitnesses = [calculate_fitness(d) for d in distances]
                
                min_dist_idx = distances.index(min(distances))
                if distances[min_dist_idx] < best_overall_dist:
                    best_overall_dist = distances[min_dist_idx]
                    best_overall_route = population[min_dist_idx][:]
                    
                # CRITÉRIO DE PARADA: Solução satisfatória atingida
                if distancia_alvo > 0 and best_overall_dist <= distancia_alvo:
                    st.toast(f"Critério de parada atingido! Distância ≤ {distancia_alvo}", icon="🛑")
                    break
                    
                # SUBSTITUIR POPULAÇÃO ANTERIOR
                new_population = []
                
                # ELITISMO
                if ativar_elitismo:
                    new_population.append(best_overall_route)
                
                while len(new_population) < pop_size:
                    # SELEÇÃO DE PAIS
                    if selection_type == "Torneio":
                        parent1 = selection_tournament(population, fitnesses, k_tournament)
                        parent2 = selection_tournament(population, fitnesses, k_tournament)
                    elif selection_type == "Roleta":
                        parent1 = selection_roulette(population, fitnesses)
                        parent2 = selection_roulette(population, fitnesses)
                    elif selection_type == "Ranking Linear":
                        parent1 = selection_linear_ranking(population, fitnesses)
                        parent2 = selection_linear_ranking(population, fitnesses)
                    else: # Truncada
                        parent1 = selection_truncated(population, fitnesses)
                        parent2 = selection_truncated(population, fitnesses)
                        
                    # CRUZAMENTO (CROSSOVER)
                    if random.random() < crossover_rate:
                        if crossover_type == "1 Ponto (Com Correção)":
                            child = crossover_um_ponto_correcao(parent1, parent2)
                        else:
                            child = crossover_uniforme(parent1, parent2)
                    else:
                        child = parent1[:]
                        
                    # MUTAÇÃO
                    if random.random() < mutation_rate:
                        total_mutations += 1
                        # Usando estritamente a mutação Swap (apresentada como "alterar genes")
                        child = mutate_swap(child)
                            
                    new_population.append(child)
                    
                # Nova Geração Assume
                population = new_population
                
                # Atualizar estados e UI
                st.session_state.best_distance = best_overall_dist
                st.session_state.best_route = best_overall_route
                st.session_state.generations_count = gen
                st.session_state.mutations_count = total_mutations

                if animar:
                    # Renderizar gráficos intercalados para não pesar muito
                    if gen % 5 == 0 or gen == max_gen or (distancia_alvo > 0 and best_overall_dist <= distancia_alvo):
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
                            
            # FIM: Quando terminar todas as gerações (ou parar antes)
            if not animar:
                metric_dist.metric("Melhor Distância", f"{best_overall_dist:.2f}", delta="Concluído")
                metric_gen.metric("Geração Atual", f"{gen} / {max_gen}")
                metric_mut.metric("Mutações Ocorridas", total_mutations)
                
                fig = plot_route(st.session_state.cities, best_overall_route, best_overall_dist, gen)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.1)
                buf.seek(0)
                chart_placeholder.image(buf)
                plt.close(fig)
                
            st.success(f"Evolução concluída! Melhor distância alcançada: {best_overall_dist:.2f}")

if __name__ == "__main__":
    main()
