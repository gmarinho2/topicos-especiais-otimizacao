import random

def gerar_solucao_inicial(definicoes_hiperparametros):
    """
    Gera um vetor (dicionário) inicial de hiperparâmetros com valores aleatórios
    baseados em suas definições de tipo (int ou float) e limites.

    Args:
        definicoes_hiperparametros (dict): Um dicionário onde cada chave é o
            nome de um hiperparâmetro e o valor é outro dicionário
            contendo 'type' ('int' ou 'float'), 'min' e 'max'.
            Exemplo:
            {
                "taxa_aprendizado": {"type": "float", "min": 0.001, "max": 0.1},
                "numero_camadas": {"type": "int", "min": 1, "max": 5}
            }

    Returns:
        dict: Um dicionário onde cada chave é o nome do hiperparâmetro
              e o valor é o valor aleatório gerado.
              Exemplo:
              {
                  "taxa_aprendizado": 0.053,
                  "numero_camadas": 3
              }
    """
    N = {}

    #foreach n em hiperparâmetros
    for nome_hp, definicao_hp in definicoes_hiperparametros.items():
        tipo = definicao_hp.get("type")
        min_val = definicao_hp.get("min")
        max_val = definicao_hp.get("max")
        valor = None

        if tipo == "choice":
            opcoes = definicao_hp.get("options")
            if opcoes is None or not isinstance(opcoes, list) or not opcoes:
                raise ValueError(f"Para o tipo 'choice', a chave 'options' com uma lista não vazia é obrigatória para '{nome_hp}'.")
            
            valor = random.choice(opcoes)
            N[nome_hp] = valor
            continue 

        if tipo is None or min_val is None or max_val is None:
            raise ValueError(
                f"A definição para o hiperparâmetro '{nome_hp}' está incompleta. "
                "É necessário 'type', 'min' e 'max'."
            )

        #Verifição de tipos
        if tipo == "int":
            valor = random.randint(min_val, max_val)
        elif tipo == "float":
            valor = random.uniform(min_val, max_val)
        else:
            raise ValueError(
                f"Tipo de hiperparâmetro '{tipo}' não suportado para '{nome_hp}'. "
                "Use 'int' ou 'float'."
            )

        N[nome_hp] = valor

    return N

def solucao_inicial_tempera(definicoes_hiperparametros):
    return gerar_solucao_inicial(definicoes_hiperparametros)

def solucao_inicial_genetico(definicoes_hiperparametros, tamanho_populacao: int):
    populacao = []
    for _ in range(tamanho_populacao):
        populacao.append(gerar_solucao_inicial(definicoes_hiperparametros))
    return populacao


if __name__ == '__main__':
    
    definicoes_hp_exemplo = {
        "taxa_aprendizado": {"type": "float", "min": 0.0001, "max": 0.1},
        "numero_neuronios_camada_inicial": {"type": "int", "min": 32, "max": 256},
        "numero_camadas_ocultas": {"type": "int", "min": 1, "max": 5},
        "taxa_dropout": {"type": "float", "min": 0.1, "max": 0.5},
        "batch_size": {"type": "int", "min": 16, "max": 128}
    }

    print("Definições dos Hiperparâmetros:")
    for nome, definicao in definicoes_hp_exemplo.items():
        print(f"  {nome}: {definicao}")

    solucao_inicial_tempe = solucao_inicial_tempera(definicoes_hp_exemplo)
    solucao_inicial_genet = solucao_inicial_genetico(definicoes_hp_exemplo, 5)

    print("\nSolução Inicial Gerada (N):")
    for nome, valor_gerado in solucao_inicial_tempe.items():
        print(f"  {nome}: {valor_gerado}")


    for i in range(len(solucao_inicial_genet)):
        print(f"\nSolução {i}:")
        for nome, valor_gerado in solucao_inicial_genet[i].items():
            print(f"  {nome}: {valor_gerado}")