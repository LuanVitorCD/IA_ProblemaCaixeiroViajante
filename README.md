# Problema do Caixeiro Viajante com Algoritmos Genéticos

## 📌 Sobre o Projeto

O **Problema do Caixeiro Viajante (PCV)** é um desafio clássico de otimização combinatória. A premissa é simples: dado um conjunto de cidades e a distância entre cada par de cidades, qual é a menor rota possível que visita cada cidade exatamente uma vez e retorna à cidade de origem? Por ser um problema *NP-Difícil*, encontrar a solução exata testando todas as combinações (força bruta) tem complexidade $O(N!)$, tornando-se inviável computacionalmente para um grande número de cidades.

Para contornar esse problema, utilizamos **Algoritmos Genéticos**, uma técnica que não garante a rota perfeita, mas encontra soluções altamente otimizadas em um tempo muito menor, explorando apenas $O(\text{População} \times \text{Gerações})$ estados possíveis.

## 🧬 Definição de Algoritmos Genéticos

Algoritmos Genéticos (AG), propostos por John Henry Holland (1975), são métodos de otimização e busca heurística inspirados nos mecanismos de evolução natural de Charles Darwin. Eles operam sobre uma população inicial de possíveis soluções e as evoluem através de gerações.

A analogia com sistemas naturais funciona da seguinte forma:
*   **Indivíduo / Cromossomo:** Representa uma solução viável para o problema (no nosso caso, uma rota completa passando por todas as cidades).
*   **Gene:** Parte da solução (uma cidade específica na rota).
*   **População:** O conjunto de soluções (várias rotas diferentes).
*   **Aptidão (Fitness):** Uma função matemática que avalia quão boa é a solução (quanto menor a distância da rota, maior o fitness).
*   **Geração:** O ciclo de evolução pelo qual a população passa.

### O Ciclo de Evolução (Como Funciona)

1.  **Geração da População Inicial:** Cria-se um conjunto aleatório de indivíduos (rotas).
2.  **Avaliação:** Calcula-se a aptidão (fitness) de cada indivíduo.
3.  **Critério de Parada:** Verifica se o número limite de gerações foi atingido ou se uma solução satisfatória foi encontrada. Se sim, o algoritmo termina. Se não, continua.
4.  **Seleção:** Indivíduos mais aptos têm maior probabilidade de serem escolhidos para a reprodução, passando seus genes adiante.
5.  **Cruzamento (Crossover):** Combinação do material genético de dois indivíduos selecionados (pais) para gerar um novo indivíduo (filho).
6.  **Mutação:** Alteração aleatória em um ou mais genes do filho gerado para manter a diversidade genética da população e evitar que o algoritmo fique preso em mínimos locais.
7.  **Substituição:** A nova geração de filhos substitui a geração anterior e o ciclo se repete a partir da Avaliação (passo 2).

## ⚙️ Implementação dos Algoritmos Genéticos

A aplicação implementa os conceitos teóricos de Algoritmos Genéticos em Python. Abaixo detalhamos como cada etapa ocorre no código:

### 1. Representação do Cromossomo e População Inicial
*   **Código:** Função `init_population(pop_size, num_cities)`.
*   **Explicação:** Cada rota é um "Cromossomo" representado por uma lista de inteiros, onde cada número é o índice de uma cidade (ex: `[0, 3, 1, 2]`). A população inicial é uma lista com `pop_size` rotas geradas aleatoriamente usando `random.sample`.

### 2. Avaliação (Função Fitness)
*   **Código:** Funções `route_distance(route, cities)` e `calculate_fitness(distance)`.
*   **Explicação:** Como queremos a *menor* distância, o problema é de minimização. O AG geralmente maximiza o *fitness*. A função de fitness inverte isso: `10000.0 / distance`. Assim, rotas curtas geram valores altos de fitness, tornando esses indivíduos mais "aptos".

### 3. Métodos de Seleção
A aplicação permite escolher entre quatro métodos para selecionar os pais na interface, implementados nas seguintes funções:
*   `selection_tournament`: **Torneio.** Seleciona `K` indivíduos aleatoriamente e escolhe o que tem a melhor aptidão.
*   `selection_roulette`: **Roleta Viciada (Tradicional).** A chance de um indivíduo ser selecionado é proporcional à sua aptidão. Indivíduos muito bons podem dominar rapidamente (perda de diversidade).
*   `selection_linear_ranking`: **Ranking Linear.** Ordena os indivíduos pela aptidão e distribui as chances linearmente com base na posição (do pior 1 até o melhor N). Evita o problema da Roleta Viciada.
*   `selection_truncated`: **Truncada.** Separa uma porcentagem (elite) dos melhores indivíduos e seleciona aleatoriamente apenas dentro desse grupo.

### 4. Cruzamento (Crossover)
Como não podemos ter cidades repetidas em uma rota do PCV, cruzamentos simples falhariam. Implementamos dois métodos de cruzamento que corrigem essas repetições:
*   `crossover_um_ponto_correcao`: **1 Ponto com Correção.** Corta o cromossomo ao meio. A primeira metade vem do Pai 1. A segunda metade é preenchida com as cidades que faltam, na exata ordem em que aparecem no Pai 2.
*   `crossover_uniforme`: **Uniforme.** Uma "máscara" binária aleatória define quais genes vêm do Pai 1. Os espaços vazios restantes são preenchidos na ordem pelas cidades do Pai 2, pulando as já inseridas.

### 5. Mutação
*   **Código:** Função `mutate_swap(route)`.
*   **Explicação:** Define-se uma taxa de mutação (ex: 5%). Se sorteada, a mutação ocorre alterando os genes do filho. Utilizamos a técnica **Swap**, onde os genes sofrem uma troca de posição. Duas cidades na rota são selecionadas aleatoriamente e trocam de lugar (ex: rota `[1, 2, 3, 4]` pode virar `[1, 4, 3, 2]`).

### 6. Elitismo (Opcional)
*   Se ativado na interface, o melhor indivíduo da geração anterior (a menor rota encontrada) é copiado diretamente para a nova geração sem passar por cruzamento ou mutação, garantindo que o algoritmo não "esqueça" a melhor solução já encontrada.

## 🛠️ Tecnologias Utilizadas

*   **Python:** Linguagem de programação principal do algoritmo e da lógica.
*   **Streamlit:** Framework utilizado para construir a interface web interativa de forma rápida.
*   **Matplotlib:** Biblioteca utilizada para renderizar o gráfico em tempo real com o mapa das cidades e as conexões da rota.
*   **NumPy / Math:** Para cálculos matemáticos e de distância Euclidiana.

## 🚀 Funcionalidades da Aplicação

Através da interface interativa construída com Streamlit, o usuário pode:
*   **Visualização em Tempo Real:** Acompanhar graficamente o mapa e como a rota (conexões entre cidades) vai se ajustando e encurtando a cada geração.
*   **Configuração Dinâmica:** Escolher em tempo de execução os métodos de seleção (Torneio, Roleta, Ranking, etc) e cruzamento.
*   **Ajuste de Parâmetros:** Controlar o tamanho do mapa (número de cidades), tamanho da população, limite de gerações, taxa de cruzamento e taxa de mutação.
*   **Critério de Parada Customizado:** Definir uma "Distância Alvo"; o algoritmo para automaticamente se atingir uma distância menor ou igual ao alvo antes das gerações acabarem.
*   **Ativação/Desativação de Elitismo e Animação.**

## 📂 Organização do Projeto

A estrutura de arquivos do projeto está organizada da seguinte maneira:

```text
IA_ProblemaCaixeiroViajante/
├── .devcontainer/       # Configurações de ambiente de desenvolvimento (Docker/VSCode)
├── .streamlit/          
│   └── config.toml      # Configurações de tema e exibição do Streamlit
├── .gitignore           # Arquivos ignorados pelo controle de versão do Git
├── main.py              # Código principal da aplicação contendo as lógicas do AG e interface
├── README.md            # Este arquivo de documentação
└── requirements.txt     # Lista de dependências (bibliotecas) necessárias para rodar o projeto
```

## 💻 Execução Local

Siga os passos abaixo para executar a aplicação no seu computador:

1.  **Clone este repositório:**
    ```bash
    git clone https://github.com/LuanVitorCD/IA_ProblemaCaixeiroViajante.git
    cd IA_ProblemaCaixeiroViajante
    ```

2.  **Crie e ative um ambiente virtual (recomendado):**
    *   No Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```
    *   No Linux/Mac:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Instale as dependências:**
    
    Certifique-se de estar com o ambiente virtual ativado e execute:
    ```bash
    pip install -r requirements.txt
    ```
    *(Nota: se o arquivo não existir, instale manualmente usando: `pip install streamlit numpy matplotlib`)*

4.  **Execute a aplicação:**
    ```bash
    streamlit run main.py
    ```

5.  **Acesse a interface:** O Streamlit abrirá uma nova aba no seu navegador automaticamente (geralmente em `http://localhost:8501`). Se não abrir, basta copiar o link gerado no terminal e colar no seu navegador.
