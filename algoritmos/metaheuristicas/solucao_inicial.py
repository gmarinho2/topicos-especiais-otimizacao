import random
import json

def gerar_solucao_inicial(definicoes_hiperparametros):
    """
    Gera um vetor (dicionário) inicial de hiperparâmetros com valores aleatórios
    baseados em suas definições.
    """
    solucao = {}

    for nome_hp, definicao_hp in definicoes_hiperparametros.items():
        tipo = definicao_hp.get("type")
        valor = None

        if tipo == "choice":
            opcoes = definicao_hp.get("options")
            if not isinstance(opcoes, list) or not opcoes:
                raise ValueError(f"Para o tipo 'choice', a chave 'options' com uma lista não vazia é obrigatória para '{nome_hp}'.")
            valor = random.choice(opcoes)

        elif tipo == "int":
            min_val = definicao_hp.get("min")
            max_val = definicao_hp.get("max")
            if min_val is None or max_val is None:
                raise ValueError(f"Para o tipo 'int', 'min' e 'max' são obrigatórios para '{nome_hp}'.")
            valor = random.randint(min_val, max_val)

        elif tipo == "float":
            min_val = definicao_hp.get("min")
            max_val = definicao_hp.get("max")
            if min_val is None or max_val is None:
                raise ValueError(f"Para o tipo 'float', 'min' e 'max' são obrigatórios para '{nome_hp}'.")
            valor = random.uniform(min_val, max_val)
            
        else:
            raise ValueError(f"Tipo de hiperparâmetro '{tipo}' não suportado para '{nome_hp}'. Use 'int', 'float' ou 'choice'.")

        solucao[nome_hp] = valor

    return solucao

def solucao_inicial_tempera(definicoes_hiperparametros):
    """Gera uma única solução inicial para Têmpera Simulada."""
    return gerar_solucao_inicial(definicoes_hiperparametros)

def solucao_inicial_genetico(definicoes_hiperparametros, tamanho_populacao: int):
    """Gera uma lista de soluções (população inicial) para Algoritmo Genético."""
    return [gerar_solucao_inicial(definicoes_hiperparametros) for _ in range(tamanho_populacao)]


if __name__ == '__main__':
    definicoes_hiperparametros = {
        "taxa_aprendizado": {"type": "float", "min": 0.0001, "max": 0.1},
        "numero_camadas": {"type": "int", "min": 1, "max": 15},
        "ativacao": {"type": "choice", "options": ["relu", "tanh", "identity", "logistic"]},
    }
    
    for i in range(1, 16):
        definicoes_hiperparametros[f"neuronios_camada_{i}"] = {"type": "int", "min": 30, "max": 100}
    
    print("--- Geração de Instâncias Iniciais ---")
    print("\nSolução Inicial Gerada para Têmpera Simulada:")
    
    
    solucao_ts = solucao_inicial_tempera(definicoes_hiperparametros)
    print(solucao_ts)

    nome_arquivo_tempera = "/home/gmarinho/code/topicos-especiais-otimizacao/algoritmos/metaheuristicas/solucao_tempera.json"
    with open(nome_arquivo_tempera, 'w', encoding='utf-8') as f:
        json.dump(solucao_ts, f, ensure_ascii=False, indent=4)
    print(f"-> Salvo em '{nome_arquivo_tempera}'")




    tamanho_populacao_ag = 20
    populacao_ag = solucao_inicial_genetico(definicoes_hiperparametros, tamanho_populacao_ag)

    print(f"\nPopulação Inicial com {tamanho_populacao_ag} indivíduos Gerada para Algoritmo Genético:")
    
    
    for i, individuo in enumerate(populacao_ag):
        print(f"  Indivíduo {i+1}: {individuo}")

    nome_arquivo_genetico = "/home/gmarinho/code/topicos-especiais-otimizacao/algoritmos/metaheuristicas/populacao_genetico.json"
    with open(nome_arquivo_genetico, 'w', encoding='utf-8') as f:
        json.dump(populacao_ag, f, ensure_ascii=False, indent=4)
    print(f"-> Salvo em '{nome_arquivo_genetico}'")