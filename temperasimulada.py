import math
import random
import numpy as np

# Função de aceitação (Metropolis criterion)
def _aceita_probabilistico(score_atual, score_novo, temperatura):
    """
    Decide se aceita uma nova solução (pior) com base na probabilidade.
    Assume score_novo >= score_atual.
    A métrica de score deve ser tal que menor é melhor (ex: perda, erro).
    """
    # Se a temperatura for muito baixa ou zero, evita divisão por zero ou overflow
    if temperatura <= 1e-9: # Praticamente T = 0
        return False
    # Calcula a probabilidade de aceitação de uma solução pior
    # (score_atual - score_novo) será <= 0. exp de não positivo é <= 1.
    probabilidade = math.exp((score_atual - score_novo) / temperatura)
    return random.random() < probabilidade

# Algoritmo Principal
def tempera_simulada_otimizacao_hiperparametros(
    # Entradas do algoritmo
    dados_treino_X,
    dados_treino_y,
    H0,                             # Hiperparâmetros iniciais (dicionário)
    T0,                             # Temperatura inicial (float)
    alpha,                          # Taxa de resfriamento (float, ex: 0.95)
    T_min,                          # Temperatura mínima para parada (float)
    # Funções customizáveis
    avalia_rede_func,               # Função: (H, X, y) -> score (menor é melhor)
    perturba_func,                  # Função: (H_atual, config_hiperparams) -> H_novo
    # Configuração e controle adicionais
    config_hiperparametros,         # Dicionário descrevendo o espaço/tipo de cada hiperparâmetro
    max_iter_sem_melhora_global=30, # Parada se não houver melhora global
    max_iter_por_temperatura=50     # Número de perturbações por nível de temperatura
):
    """
    Implementa o Algoritmo de Tempera Simulada para Otimização de Hiperparâmetros.

    Args:
        dados_treino_X: Dados de características para treino.
        dados_treino_y: Rótulos dos dados de treino.
        H0 (dict): Configuração inicial de hiperparâmetros.
        T0 (float): Temperatura inicial.
        alpha (float): Fator de resfriamento (e.g., 0.8 a 0.99).
        T_min (float): Temperatura mínima para critério de parada.
        avalia_rede_func (callable): Função que recebe (hiperparâmetros, X, y) e
                                     retorna um score (onde menor é melhor, ex: perda).
        perturba_func (callable): Função que recebe (hiperparâmetros_atuais, config_hiperparametros)
                                  e retorna uma nova configuração de hiperparâmetros.
        config_hiperparametros (dict): Dicionário que define o espaço e tipo de cada
                                       hiperparâmetro, usado pela `perturba_func`.
                                       Ex: {'lr': {'type': 'float', 'min': 0.0001, 'max': 0.1}, ...}
        max_iter_sem_melhora_global (int): Número máximo de iterações de resfriamento
                                           sem melhora no `Hbest_global`.
        max_iter_por_temperatura (int): Número de vizinhos a serem explorados em cada
                                        nível de temperatura.

    Returns:
        tuple: (Hbest_global, score_best_global) - Melhor conjunto de hiperparâmetros
               encontrado e seu respectivo score.
    """
    # 1. Hbest_global, Hbest ← H0, H0
    Hbest_global = H0.copy()
    Hbest_iter = H0.copy() # Hbest no pseudocódigo (candidato atual)

    # 2. Avalie Hbest com AvaliaRede()
    print(f"Avaliando configuração inicial H0: {H0}")
    score_best_iter = avalia_rede_func(Hbest_iter, dados_treino_X, dados_treino_y)
    score_best_global = score_best_iter
    print(f"Score inicial (H0): {score_best_iter:.5f}")

    # 3. T ← T0
    T = T0
    
    iter_sem_melhora_global_count = 0
    ciclos_resfriamento = 0

    # 4. while T > Tmin do (e outros critérios de parada)
    while T > T_min and iter_sem_melhora_global_count < max_iter_sem_melhora_global:
        ciclos_resfriamento += 1
        print(f"\nCiclo de Resfriamento {ciclos_resfriamento} - Temperatura: {T:.5f}")
        
        num_aceitos_nesta_temp = 0
        for i_pert in range(max_iter_por_temperatura):
            # 5. Hnew ← Perturba(Hbest_iter)
            Hnew = perturba_func(Hbest_iter, config_hiperparametros)
            
            # Avalia Hnew
            score_new = avalia_rede_func(Hnew, dados_treino_X, dados_treino_y)
            
            # 6. if AvaliaRede(Hnew) < AvaliaRede(Hbest_iter) or Aceita(Hnew, T) then
            #    (Aceita(Hnew,T) é a parte probabilística para soluções piores)
            aceitar_nova_solucao = False
            if score_new < score_best_iter:
                aceitar_nova_solucao = True
                print(f"Perturbação. {i_pert+1}: Solução MELHOR ({score_new:.5f} < {score_best_iter:.5f}). \nAceitando Hnew (Perturbação. {i_pert+1}): {Hnew}")
            # Caso contrário, considera aceitar uma solução pior probabilisticamente
            elif _aceita_probabilistico(score_best_iter, score_new, T):
                aceitar_nova_solucao = True
                num_aceitos_nesta_temp +=1
                print(f"Perturbação. {i_pert+1}: Solução PIOR ({score_new:.5f} >= {score_best_iter:.5f}) -----------> ACEITA probabilisticamente. \nHnew: {Hnew}")
            else:
                print(f"Perturbação. {i_pert+1}: Solução PIOR ({score_new:.5f} >= {score_best_iter:.5f}) -----------> REJEITADA. \nHnew: {Hnew}")

            if aceitar_nova_solucao:
                # 7. Hbest_iter ← Hnew
                Hbest_iter = Hnew.copy()
                score_best_iter = score_new
                
                # 8. (fim do if interno)
            
            # 9. if AvaliaRede(Hbest_iter) < AvaliaRede(Hbest_global) then
            if score_best_iter < score_best_global:
                # 10. Hbest_global ← Hbest_iter
                Hbest_global = Hbest_iter.copy()
                score_best_global = score_best_iter
                iter_sem_melhora_global_count = 0 # Reseta contador de não melhora
                print(f"  !!! NOVO MELHOR GLOBAL encontrado !!! Score: {score_best_global:.5f}, H: {Hbest_global}")
            # 11. (fim do if de atualização global)
        
        print(f"\nResumo da Temperatura {T:.5f}: {num_aceitos_nesta_temp}/{max_iter_por_temperatura} soluções piores foram aceitas.")
        
        # 12. T ← T ⋅ α (Resfriamento)
        T *= alpha
        iter_sem_melhora_global_count +=1

    # 13. end (fim do while)
    if T <= T_min:
        print(f"\nParada: Temperatura T ({T:.5f}) atingiu ou passou T_min ({T_min}).")
    if iter_sem_melhora_global_count >= max_iter_sem_melhora_global:
        print(f"\nParada: Sem melhora global por {max_iter_sem_melhora_global} ciclos de resfriamento.")

    # 14. return Hbest_global
    print(f"\n=====================================================================================================")
    print(f"\n--- Otimização Concluída ---")
    print(f"Melhor conjunto de hiperparâmetros global (Hbest_global): {Hbest_global}")
    print(f"Melhor score global: {score_best_global:.5f}")
    return Hbest_global, score_best_global

# --- Abaixo, exemplos de placeholders para as funções customizáveis ---

def avalia_rede_placeholder(hiperparametros, X_train, y_train):
    """
    PLACEHOLDER para AvaliaRede(). 
    
    ESSA FUNÇÃO É APENAS PARA SIMULAR O TREINAMENTO DA REDE!!!!!
    NÃO IMPLEMENTADA!! TEM QUE IMPLEMENTAR ISSO AQUI EH DE SACANAGEM!!!!

    
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
    
    # Adiciona um pouco de ruído para simular a variabilidade do treinamento
    score += random.uniform(-0.001, 0.001)
    # Garante que o score não seja negativo se a função objetivo assim o exigir
    # print(f"    [avalia_rede_placeholder] Score gerado: {max(0, score):.5f}")
    return max(0, score) # Retorna o score (menor é melhor)

def perturba_hiperparametros_placeholder(H_atual, config_hiperparametros):
    """
    PLACEHOLDER para Perturba().
    Gera uma nova configuração de hiperparâmetros (vizinha) a partir da atual. 
    (OU SEJA, DEFINE COMO VAO SER ALTERADOS NA PERTURBAÇÃO, SE SOMA 1, SUBTRAI 0.4, MUDA ATIVACAO ETC)
    NÃO IMPLEMENTADA!! TEM QUE IMPLEMENTAR ISSO AQUI EH DE SACANAGEM!!!!
    """
    H_novo = H_atual.copy()
    # Escolhe um hiperparâmetro aleatório para perturbar
    param_para_perturbar = random.choice(list(config_hiperparametros.keys()))
    config_param = config_hiperparametros[param_para_perturbar]
    
    valor_atual = H_novo[param_para_perturbar]
    novo_valor = valor_atual

    # print(f"    [perturba_placeholder] Perturbando '{param_para_perturbar}' (valor atual: {valor_atual})")

    tipo = config_param['type']
    if tipo == 'float':
        # Perturbação para float: pequena mudança aditiva ou multiplicativa
        # Escala logarítmica é comum para taxa de aprendizado
        if config_param.get('scale') == 'log':
            log_min = math.log(config_param['min'])
            log_max = math.log(config_param['max'])
            log_atual = math.log(valor_atual)
            # Mudança de até 10% da faixa logarítmica total
            perturbacao = random.uniform(-0.1, 0.1) * (log_max - log_min)
            log_novo = np.clip(log_atual + perturbacao, log_min, log_max)
            novo_valor = math.exp(log_novo)
        else: # Escala linear
            perturbacao = random.gauss(0, (config_param['max'] - config_param['min']) * 0.05) # 5% da faixa
            novo_valor = np.clip(valor_atual + perturbacao, config_param['min'], config_param['max'])

    elif tipo == 'int':
        # Perturbação para int: pequena mudança, respeitando steps se houver
        step = config_param.get('step', 1)
        # Muda em +/- alguns steps
        mudanca = random.choice([-2*step, -1*step, 1*step, 2*step])
        novo_valor_temp = valor_atual + mudanca
        
        # Arredonda para o múltiplo de 'step' mais próximo, se 'step' > 1
        if step > 1:
            novo_valor_temp = round(novo_valor_temp / step) * step
            
        novo_valor = int(np.clip(novo_valor_temp, config_param['min'], config_param['max']))

    elif tipo == 'choice':
        # Perturbação para escolha: seleciona uma opção diferente da atual
        opcoes = list(config_param['options'])
        if len(opcoes) > 1: # Só perturba se houver mais de uma opção
            opcoes_validas = [opt for opt in opcoes if opt != valor_atual]
            if opcoes_validas: # Se houver opções diferentes da atual
                novo_valor = random.choice(opcoes_validas)
            # else: novo_valor permanece o mesmo se só houver uma opção ou a atual for a única restante

    H_novo[param_para_perturbar] = novo_valor
    # print(f"    [perturba_placeholder] Novo valor para '{param_para_perturbar}': {novo_valor}")
    return H_novo

# --- Exemplo de Uso ---
if __name__ == '__main__':
    # Dados de treino (dummy)
    X_dummy = np.random.rand(100, 10) # 100 amostras, 10 features
    y_dummy = np.random.randint(0, 2, 100) # Classificação binária

    # 1. Hiperparâmetros iniciais (H0)
    H0_exemplo = {
        'lr': 0.05,
        'camadas': 3,
        'neuronios': 128,
        'ativacao': 'sigmoid'
    }

    # 2. Configuração do espaço de hiperparâmetros (para a função `perturba_func`)
    config_hiperparametros_exemplo = {
        'lr': {'type': 'float', 'min': 0.0001, 'max': 0.1, 'scale': 'log'},
        'camadas': {'type': 'int', 'min': 1, 'max': 5, 'step': 1},
        'neuronios': {'type': 'int', 'min': 32, 'max': 256, 'step': 32}, # Múltiplos de 32
        'ativacao': {'type': 'choice', 'options': ['relu', 'tanh', 'sigmoid', 'elu']}
    }

    # 3. Parâmetros da Tempera Simulada
    T0_exemplo = 1.0      # Temperatura inicial
    alpha_exemplo = 0.90  # Taxa de resfriamento
    T_min_exemplo = 0.001 # Temperatura mínima
    max_iter_sem_melhora_ex = 20
    max_iter_por_temp_ex = 10

    print("Iniciando Tempera Simulada para Otimização de Hiperparâmetros (com placeholders)...")

    # Chamada da função principal
    melhores_hiperparametros, melhor_score_final = tempera_simulada_otimizacao_hiperparametros(
        dados_treino_X=X_dummy,
        dados_treino_y=y_dummy,
        H0=H0_exemplo,
        T0=T0_exemplo,
        alpha=alpha_exemplo,
        T_min=T_min_exemplo,
        avalia_rede_func=avalia_rede_placeholder, # Usando o placeholder
        perturba_func=perturba_hiperparametros_placeholder, # Usando o placeholder
        config_hiperparametros=config_hiperparametros_exemplo,
        max_iter_sem_melhora_global=max_iter_sem_melhora_ex,
        max_iter_por_temperatura=max_iter_por_temp_ex
    )

    print("\n--- Resultado Final da Otimização (Placeholder) ---")
    print(f"Melhores Hiperparâmetros Encontrados: {melhores_hiperparametros}")
    print(f"Melhor Score Obtido: {melhor_score_final:.5f}")
    print("\nLembrete: As funções `avalia_rede_placeholder` e `perturba_hiperparametros_placeholder` são apenas exemplos.")
    print("Você precisará substituí-las por implementações que treinem e avaliem sua rede neural real.")