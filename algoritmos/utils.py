
import random

def calcular_fitness(individuo):
    """
    PLACEHOLDER para calcular o fitness de um indivíduo.

    Simula o treinamento de um rede neural.
    """

    rmse_simulado = 1.0
    if 'taxa_aprendizado' in individuo:
        rmse_simulado -= individuo['taxa_aprendizado'] * 5
    if 'numero_camadas' in individuo:
        rmse_simulado += individuo['numero_camadas'] * 0.05
        
    return rmse_simulado + random.uniform(-0.05, 0.05)


# --- Funções do Algoritmo Genético ---

def selecao_torneio(populacao, tamanho_torneio=3):
    """
    Seleciona um pai da população usando seleção por torneio.
    """
    torneio = random.sample(populacao, tamanho_torneio)
    vencedor = min(torneio, key=lambda individuo: individuo['fitness'])
    return vencedor

def cruzamento(pai1, pai2):
    """
    Para cada gene (hiperparâmetro), o filho herda o valor de um dos pais aleatoriamente.
    """
    filho = {}
    # Itera sobre todos os hiperparâmetros (genes)
    for gene in pai1:
        if gene != 'fitness': # Não cruza o valor de fitness
            if random.random() < 0.5:
                filho[gene] = pai1[gene]
            else:
                filho[gene] = pai2[gene]
    return filho

def mutacao(individuo, pm, definicoes_hiperparametros):
    """
    Para cada gene, há uma probabilidade 'pm' de que ele sofra mutação.
    """
    individuo_mutado = individuo.copy()
    for gene in individuo_mutado:
        if random.random() < pm:
            # Gera um novo valor aleatório para este gene
            definicao_gene = definicoes_hiperparametros[gene]
            tipo = definicao_gene.get("type")
            
            if tipo == "int":
                individuo_mutado[gene] = random.randint(definicao_gene["min"], definicao_gene["max"])
            elif tipo == "float":
                individuo_mutado[gene] = random.uniform(definicao_gene["min"], definicao_gene["max"])
            elif tipo == "choice":
                individuo_mutado[gene] = random.choice(definicao_gene["options"])

    return individuo_mutado

def seleciona_melhores(populacao_combinada, n):
    populacao_ordenada = sorted(populacao_combinada, key=lambda individuo: individuo['fitness'])
    return populacao_ordenada[0:n]
