from solucao_inicial import solucao_inicial_genetico
from utils import calcular_fitness, cruzamento, mutacao, selecao_torneio, seleciona_melhores


def algoritmo_genetico(populacao_inicial, tamanho_populacao, num_geracoes, pm, definicoes_hp):
    """
    Executa o algoritmo genético conforme o pseudocódigo.
    
    Args:
        populacao_inicial (list): Lista de dicionários, onde cada dicionário é um indivíduo.
        tamanho_populacao (int): Tamanho da população.
        num_geracoes (int): Número de gerações.
        pm (float): Probabilidade de mutação.
        definicoes_hp (dict): Definições dos hiperparâmetros para a mutação.

    Returns:
        dict: O melhor indivíduo encontrado após todas as gerações.
    """
    # avaliar população inicial
    populacao = []
    for individuo_sem_fitness in populacao_inicial:
        fitness = calcular_fitness(individuo_sem_fitness, parametros="placeholder", target="placeholder")
        individuo_com_fitness = individuo_sem_fitness.copy()
        individuo_com_fitness['fitness'] = fitness
        populacao.append(individuo_com_fitness)

    for geracao in range(num_geracoes):
        #2.Pnew ← ∅
        populacao_nova = []
        
        # 3.while |Pnew| < N do
        while len(populacao_nova) < tamanho_populacao:
            # 4. pai1, pai2 ← SelecaoTorneio(P)
            pai1 = selecao_torneio(populacao)
            pai2 = selecao_torneio(populacao)
            
            filho = cruzamento(pai1, pai2)
            filho = mutacao(filho, pm, definicoes_hp)
            
            # avalia o novo filho
            fitness_filho = calcular_fitness(filho, parametros="placeholder", target="placeholder")
            filho['fitness'] = fitness_filho
            
            # 7. adicione filho a Pnew
            populacao_nova.append(filho)
        # 8. end
        
        # 9. P ← SelecionaMelhores(P ∪ Pnew, N)
        P_combinada = populacao + populacao_nova
        populacao = seleciona_melhores(P_combinada, tamanho_populacao)
        
        # log do progresso
        melhor_da_geracao = populacao[0]
        print(f"Geração {geracao+1}/{num_geracoes} | Melhor RMSE: {melhor_da_geracao['fitness']:.4f}")
    
    melhor_final = populacao[0]
    return melhor_final




if __name__ == '__main__':
    definicoes_hiperparametros = {
        "taxa_aprendizado": {"type": "float", "min": 0.0001, "max": 0.1},
        "numero_camadas": {"type": "int", "min": 1, "max": 8},
        "taxa_dropout": {"type": "float", "min": 0.1, "max": 0.6},
        "ativacao": {"type": "choice", "options": ["relu", "tanh", "sigmoid"]}
    }

    N_POPULACAO = 20  
    N_GERACOES = 50   
    PROB_MUTACAO = 0.1
    
    print("Iniciando Algoritmo Genético...")
    print(f"Tamanho da População: {N_POPULACAO}, Gerações: {N_GERACOES}, Prob. Mutação: {PROB_MUTACAO}")
    print("-" * 30)

    #população inicial
    pop_inicial = solucao_inicial_genetico(definicoes_hiperparametros=definicoes_hiperparametros, 
                                            tamanho_populacao=N_POPULACAO)
    
    melhor_individuo_encontrado = algoritmo_genetico(
        pop_inicial,
        N_POPULACAO,
        N_GERACOES,
        PROB_MUTACAO,
        definicoes_hiperparametros,
    )
    
    print("-" * 30)
    print("Otimização Concluída!")

    print("\nMelhor indivíduo encontrado:")
    for nome, valor_gerado in melhor_individuo_encontrado.items():
        print(f" | {nome}: {valor_gerado}")