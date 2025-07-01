import json
from redeneural import calcular_fitness_rmse
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

        fitness = calcular_fitness_rmse(individuo_sem_fitness)
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
            fitness_filho = calcular_fitness_rmse(filho)
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


    '''
    PREVENDO A HEMOGLOBINA GLICADA (VARIA DE 25 a 346) COM BASE EM:
    
    age: Idade em anos
    sex: Sexo do paciente
    bmi: Índice de Massa Corporal (IMC)
    bp: Pressão arterial média
    s1: Colesterol total (TC - uma medida de soro sanguíneo)
    s2: Lipoproteínas de baixa densidade (LDL, o "colesterol ruim")
    s3: Lipoproteínas de alta densidade (HDL, o "colesterol bom")
    s4: Colesterol total / HDL (TCH)
    s5: Possivelmente o logaritmo do nível de triglicerídeos séricos (LTG)
    s6: Nível de açúcar no sangue (Glicose - GLU)'''


if __name__ == '__main__':
    definicoes_hiperparametros = {
        "taxa_aprendizado": {"type": "float", "min": 0.0001, "max": 0.1},
        "numero_camadas": {"type": "int", "min": 1, "max": 15},
        "ativacao": {"type": "choice", "options": ["relu", "tanh", "identity", "logistic"]},
    }
    for i in range(1, 16):
        definicoes_hiperparametros[f"neuronios_camada_{i}"] = {"type": "int", "min": 30, "max": 100}
    
    nome_arquivo_genetico = "/home/gmarinho/code/topicos-especiais-otimizacao/algoritmos/metaheuristicas/populacao_genetico.json"
    pop_inicial = []
    
    try:
        print(f"Lendo população inicial do arquivo '{nome_arquivo_genetico}'...")
        with open(nome_arquivo_genetico, 'r', encoding='utf-8') as f:
            pop_inicial = json.load(f)
        print("População carregada com sucesso.")
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{nome_arquivo_genetico}' não encontrado.")
        print("Por favor, execute o script de geração de soluções primeiro.")
        exit() # Encerra o script se o arquivo não existir
    except json.JSONDecodeError:
        print(f"ERRO: O arquivo '{nome_arquivo_genetico}' está mal formatado ou vazio.")
        exit()

    # Define os parâmetros do AG. N_POPULACAO é definido pelo tamanho da população no arquivo.
    N_POPULACAO = len(pop_inicial)
    N_GERACOES = 50   
    PROB_MUTACAO = 0.4
    
    print("\nIniciando Algoritmo Genético...")
    print(f"Tamanho da População: {N_POPULACAO} (do arquivo), Gerações: {N_GERACOES}, Prob. Mutação: {PROB_MUTACAO}")
    print("-" * 30)

    # O resto do código permanece o mesmo, agora usando a 'pop_inicial' lida do arquivo
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