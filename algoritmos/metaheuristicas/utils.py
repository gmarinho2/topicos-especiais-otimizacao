import math
import random

import numpy as np

############ Funções da Têmpera Simulada ##############
def aceita_probabilistico(score_atual, score_novo, temperatura):
    """
    Decide se aceita uma nova solução (pior) com base na probabilidade.
    Assume score_novo >= score_atual.
    A métrica de score deve ser tal que menor é melhor (ex: perda, erro).
    """
    if temperatura <= 1e-9: # para evitar divisão por zero
        return False

    probabilidade = math.exp((score_atual - score_novo) / temperatura)
    return random.random() < probabilidade


def avalia_rede_func(hiperparametros, X_train, y_train):
    """
    PLACEHOLDER para AvaliaRede(). Não implementada para resumo     
    """
    print(f"\nAvaliando: {hiperparametros}")
    # Simulação de score:
    # Exemplo: 'lr' ideal = 0.01, 'camadas' ideal = 2, 'neuronios' ideal = 64
    score = 0.0
    if 'lr' in hiperparametros:
        score += (hiperparametros['lr'] - 0.01)**2 * 100
    if 'camadas' in hiperparametros:
        score += (hiperparametros['camadas'] - 2)**2 * 0.1
    if 'neuronios' in hiperparametros:
        score += ((hiperparametros['neuronios'] - 64)/32)**2 * 0.05
    if 'ativacao' in hiperparametros:
        if hiperparametros['ativacao'] == 'relu': score -= 0.01 # pequena bonificação
        elif hiperparametros['ativacao'] == 'tanh': score += 0.005
    
    score += random.uniform(-0.001, 0.001)
    # Garante que o score não seja negativo se a função objetivo assim o exigir
    # print(f"    [avalia_rede_placeholder] Score gerado: {max(0, score):.5f}")
    return max(0, score) # Retorna o score (menor é melhor)


def perturba(H_atual, config_hiperparametros, temperatura_atual, temperatura_inicial):
    H_novo = H_atual.copy()
    param_para_perturbar = random.choice(list(config_hiperparametros.keys()))
    config_param = config_hiperparametros[param_para_perturbar]
    valor_atual = H_novo[param_para_perturbar]

    fator_temperatura = temperatura_atual / (temperatura_inicial + 1e-9)
    tipo = config_param['type']
    
    # Passamos o NOME do parâmetro para as funções auxiliares
    if tipo == 'float':
        novo_valor = _perturbar_float(param_para_perturbar, valor_atual, config_param, fator_temperatura)
    elif tipo == 'int':
        novo_valor = _perturbar_int(param_para_perturbar, valor_atual, config_param, fator_temperatura)
    elif tipo == 'choice':
        novo_valor = _perturbar_choice(valor_atual, config_param)
    else:
        raise ValueError(f"Tipo de hiperparâmetro desconhecido: {tipo}")

    H_novo[param_para_perturbar] = novo_valor
    return H_novo

def _perturbar_float(param_name, valor_atual, config_param, fator_temperatura):
    min_val, max_val = config_param['min'], config_param['max']

    if param_name == 'lr':
        sigma_base = 0.01
    else:
        sigma_base = (max_val - min_val) * 0.1

    sigma_atual = sigma_base * fator_temperatura + 1e-9
    perturbacao = random.gauss(0, sigma_atual)
    novo_valor = valor_atual + perturbacao
    
    return np.clip(novo_valor, min_val, max_val)

def _perturbar_int(param_name, valor_atual, config_param, fator_temperatura):
    min_val, max_val = config_param['min'], config_param['max']
    
    if param_name == 'camadas':
        passo_maximo = 2
    elif param_name == 'neuronios':
        passo_maximo = 64
    else:
        passo_maximo = max(1, (max_val - min_val) // 10)

    range_atual = max(1, int(round(passo_maximo * fator_temperatura)))
    mudanca = random.randint(-range_atual, range_atual)
    
    if mudanca == 0 and min_val < max_val:
        mudanca = random.choice([-1, 1])

    novo_valor = valor_atual + mudanca
    return int(np.clip(novo_valor, min_val, max_val))

def _perturbar_choice(valor_atual, config_param):
    opcoes = config_param['options']
    if len(opcoes) <= 1:
        return valor_atual
        
    opcoes_validas = [opt for opt in opcoes if opt != valor_atual]
    if not opcoes_validas:
        return valor_atual
        
    return random.choice(opcoes_validas)


############ Funções do Algoritmo Genético ##############
def calcular_fitness(individuo, parametros, target):
    """
    Simula o treinamento de um rede neural.
    """
    return avalia_rede_func(individuo, parametros, target)


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
