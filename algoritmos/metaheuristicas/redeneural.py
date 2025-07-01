import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=ConvergenceWarning)

definicoes_hiperparametros = {
    "taxa_aprendizado": {"type": "float", "min": 0.0001, "max": 0.1},
    "numero_camadas": {"type": "int", "min": 1, "max": 15},
    "ativacao": {"type": "choice", "options": ["relu", "tanh", "identity", "logistic"]},
}
# Adiciona dinamicamente os hiperparâmetros para o número de neurônios
for i in range(1, 16):
    definicoes_hiperparametros[f"neuronios_camada_{i}"] = {"type": "int", "min": 30, "max": 100}


# Variável global para armazenar os dados e evitar recarregamentos desnecessários
diabetes_data = None


def carregar_dados_diabetes():
    """
    Carrega, particiona e pré-processa o dataset Diabetes.
    Utiliza uma cache em memória (variável global) para evitar recargas repetidas,
    o que acelera muito o processo de otimização.
    """
    global diabetes_data
    if diabetes_data is None:
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target

        # Divide os dados em conjuntos de treino e teste
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25, random_state=42)

        # Normaliza os dados (passo crucial para redes neurais)
        scaler = StandardScaler()
        X_treino = scaler.fit_transform(X_treino)
        X_teste = scaler.transform(X_teste)
        
        # Armazena os dados processados na variável global
        diabetes_data = (X_treino, X_teste, y_treino, y_teste)
        
    return diabetes_data

def calcular_fitness_rmse(hiperparametros: dict) -> float:
    """
    Função de fitness que treina uma MLP de Regressão e retorna o RMSE.

    Esta é a função a ser chamada pelo seu algoritmo de Têmpera Simulada ou
    Algoritmo Genético. Ela recebe um dicionário com uma instância de
    hiperparâmetros e retorna o seu "custo" (RMSE).

    Args:
        hiperparametros (dict): Dicionário com os valores dos hiperparâmetros a serem testados.

    Returns:
        float: O valor do Root Mean Squared Error (RMSE) no conjunto de teste.
               Retorna um valor infinito se ocorrer um erro.
    """
    try:
        # 1. Garante que os dados estão carregados
        X_treino, X_teste, y_treino, y_teste = carregar_dados_diabetes()

        # 2. Constrói a arquitetura da rede dinamicamente
        num_camadas_ativas = hiperparametros["numero_camadas"]
        neuronios_por_camada = [
            hiperparametros[f"neuronios_camada_{i+1}"] for i in range(num_camadas_ativas)
        ]
        hidden_layer_sizes = tuple(neuronios_por_camada)

        # 3. Cria a instância do Regressor MLP
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=hiperparametros["ativacao"],
            learning_rate_init=hiperparametros["taxa_aprendizado"],
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42
        )

        # 4. Treina o modelo
        mlp.fit(X_treino, y_treino)

        # 5. Faz predições e calcula o RMSE
        predicoes = mlp.predict(X_teste)
        
        # --- LINHA CORRIGIDA ---
        # Calcula o MSE e depois tira a raiz quadrada para obter o RMSE.
        # Esta abordagem é compatível com todas as versões do scikit-learn.
        rmse = np.sqrt(mean_squared_error(y_teste, predicoes))
        # -----------------------

        return rmse

    except Exception as e:
        # Penaliza combinações de hiperparâmetros que causam erros
        print(f"Erro durante a avaliação: {e}")
        return float('inf')

# --- Bloco de Exemplo de Uso ---
if __name__ == '__main__':
    print("Iniciando teste da função de fitness...")



    '''
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
    
    carregar_dados_diabetes()
    print("-" * 30)

    exemplo_hiperparametros = {
        "taxa_aprendizado": 0.01,
        "numero_camadas": 2,
        "ativacao": "relu",
        "neuronios_camada_1": 80,
        "neuronios_camada_2": 60,
    }
    
    # Adiciona chaves não utilizadas para simular o comportamento do otimizador
    for i in range(3, 16):
        exemplo_hiperparametros[f"neuronios_camada_{i}"] = 50

    print(f"Testando com os hiperparâmetros (dicionário completo simulado):")
    print(exemplo_hiperparametros)
    
    # Chama a função de fitness
    rmse_resultado = calcular_fitness_rmse(exemplo_hiperparametros)

    if rmse_resultado != float('inf'):
        print(f"\nTeste concluído. RMSE obtido: {rmse_resultado:.4f}")
    else:
        print("\nO teste resultou em um erro.")