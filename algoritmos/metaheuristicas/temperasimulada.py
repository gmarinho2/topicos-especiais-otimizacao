import json
from solucao_inicial import solucao_inicial_tempera
from utils import aceita_probabilistico, perturba
from redeneural import calcular_fitness_rmse


def tempera_simulada(
    hiperparametros_inicias,                             
    temperatura_inicial,
    taxa_resfriamento,
    temperatura_minima,                          
    config_hiperparametros,         # dicionario descrevendo o espaço/tipo de cada hiperparâmetro
    max_iter_sem_melhora_global=30, # Parada se não houver melhora global
    max_iter_por_temperatura=50     # Número de perturbações por nível de temperatura
):

    # avalia o candidato inicial
    Hbest_global = hiperparametros_inicias.copy()
    Hbest_iter = hiperparametros_inicias.copy()
    score_best_iter = calcular_fitness_rmse(Hbest_iter)
    score_best_global = score_best_iter

    print(f"Avaliando configuração inicial hiperparametros_inicias: {hiperparametros_inicias}")
    print(f"Score inicial (hiperparametros_inicias): {score_best_iter:.5f}")
    
    # temperatura é inicializada
    temperatura = temperatura_inicial
    
    iter_sem_melhora_global_count = 0
    ciclos_resfriamento = 0

    # enquanto temperatura não é a minima E ainda não teve 10 iterações sem melhora
    while temperatura > temperatura_minima and iter_sem_melhora_global_count < max_iter_sem_melhora_global:
        ciclos_resfriamento += 1
        print(f"\nCiclo de Resfriamento {ciclos_resfriamento} - Temperatura: {temperatura:.5f}")
        
        num_aceitos_nesta_temp = 0
        for i_pert in range(max_iter_por_temperatura):
            
            # cria e avalia nova solução usando perturbação 
            Hnew = perturba(Hbest_iter, config_hiperparametros, temperatura, temperatura_inicial)
            score_new = calcular_fitness_rmse(Hnew)

            # a solução é aceita se for melhor, ou se for pior mas passar no teste probabilístico.
            is_melhor = score_new < score_best_iter
            is_aceita_probabilistica = aceita_probabilistico(score_best_iter, score_new, temperatura)

            if is_melhor or is_aceita_probabilistica:
                if is_melhor:
                    print(f"Perturbação. {i_pert+1}: Solução MELHOR ({score_new:.5f} < {score_best_iter:.5f}) ---> ACEITA")
                else:
                    num_aceitos_nesta_temp += 1
                    print(f"Perturbação. {i_pert+1}: Solução PIOR ({score_new:.5f} >= {score_best_iter:.5f}) ----> ACEITA (probabilisticamente)")

                # atualiza a melhor solução da iteração atual
                Hbest_iter = Hnew.copy()
                score_best_iter = score_new

                # após aceitar uma solução, verificamos se ela é um novo melhor global
                if score_best_iter < score_best_global:
                    Hbest_global = Hbest_iter.copy()
                    score_best_global = score_best_iter
                    iter_sem_melhora_global_count = 0 
                    print(f"NOVO MELHOR GLOBAL: {score_best_global:.5f}")

            else:
                print(f"Perturbação. {i_pert+1}: Solução PIOR ({score_new:.5f} >= {score_best_iter:.5f}) ---> REJEITADA")
        
        print(f"\nResumo da Temperatura {temperatura:.5f}: {num_aceitos_nesta_temp}/{max_iter_por_temperatura} soluções foram aceitas.")
        
        # aplica resfriamento
        temperatura *= taxa_resfriamento
        iter_sem_melhora_global_count +=1

    if temperatura <= temperatura_minima:
        print(f"\nParada: Temperatura temperatura ({temperatura:.5f}) atingiu ou passou temperatura_minima ({temperatura_minima}).")
    if iter_sem_melhora_global_count >= max_iter_sem_melhora_global:
        print(f"\nParada: Sem melhora global por {max_iter_sem_melhora_global} ciclos de resfriamento.")

    print(f"\n=====================================================================================================")
    print(f"\n--- Otimização Concluída ---")
    print(f"Melhor conjunto de hiperparâmetros global (Hbest_global): {Hbest_global}")
    print(f"Melhor score global: {score_best_global:.5f}")

    return Hbest_global, score_best_global

if __name__ == '__main__':
    definicoes_hiperparametros = {
    "taxa_aprendizado": {"type": "float", "min": 0.0001, "max": 0.1},
    "numero_camadas": {"type": "int", "min": 1, "max": 15},
    "ativacao": {"type": "choice", "options": ["relu", "tanh", "identity", "logistic"]},
    }

    for i in range(1, 16):
        definicoes_hiperparametros[f"neuronios_camada_{i}"] = {"type": "int", "min": 30, "max": 100}

    hiperparametros_inicias_exemplo = solucao_inicial_tempera(definicoes_hiperparametros)

    temperatura_inicial_exemplo = 1.0      
    taxa_resfriamento_EXEMPLO = 0.2 
    temperatura_minima = 0.00001 
    max_iter_sem_melhora_ex = 10
    max_iter_por_temp_ex = 10


    #LEITURA DO ARQUIVO
    nome_arquivo_tempera = "algoritmos/metaheuristicas/solucao_tempera.json"
    try:
        print(f"Lendo população inicial do arquivo '{nome_arquivo_tempera}'...")
        with open(nome_arquivo_tempera, 'r', encoding='utf-8') as f:
            pop_inicial = json.load(f)
        print("População carregada com sucesso.")
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{nome_arquivo_tempera}' não encontrado.")
        print("Por favor, execute o script de geração de soluções primeiro.")
        exit() # Encerra o script se o arquivo não existir
    except json.JSONDecodeError:
        print(f"ERRO: O arquivo '{nome_arquivo_tempera}' está mal formatado ou vazio.")
        exit()



    
    print("############################ INICIO TEMPERA SIMULADA ##############################")
    
    melhores_hiperparametros, melhor_score_final = tempera_simulada(
        hiperparametros_inicias_exemplo,
        temperatura_inicial_exemplo,
        taxa_resfriamento_EXEMPLO,
        temperatura_minima,
        definicoes_hiperparametros,
        max_iter_sem_melhora_ex,
        max_iter_por_temp_ex
    )

    for nome, valor_gerado in melhores_hiperparametros.items():
        print(f" | {nome}: {valor_gerado}")