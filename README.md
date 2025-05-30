# Tópicos Especiais em Otimização

## Documentação do Projeto: Configuração do Ambiente

A seguir, apresentamos o conteúdo do arquivo `README.md` que detalha como configurar o ambiente Python necessário para executar o código que busca hiperparametros de redes neurais usando Algoritmo Genetico e Tempera Simulada.


## Pré-requisitos

* Python 3.x instalado em seu sistema. Você pode verificar isso abrindo um terminal ou prompt de comando e digitando `python --version` ou `python3 --version`.

## Passos para Configuração

Siga os passos abaixo para criar o ambiente virtual, instalar as dependências e executar o código.

**1. Crie um Diretório para o Projeto (se ainda não tiver)**

   Se você ainda não tem o diretório, faça:
   ```bash
   git clone git@github.com:gmarinho2/topicos-especiais-otimizacao.git
   ```
   Se tem, navegue até ele.

**2. Crie o Ambiente Virtual**

   Dentro do diretório do projeto, execute o seguinte comando para criar um ambiente virtual. Vamos chamá-lo de `.virtual_env_teo`:
   ```bash
   python3 -m venv .virtual_env_teo
   ```
   Isso criará um subdiretório chamado `.virtual_env_teo` contendo os arquivos do ambiente virtual.

**3. Ative o Ambiente Virtual**

   A ativação do ambiente virtual modifica o seu shell para usar o interpretador Python e os pacotes instalados nesse ambiente específico.

   * **No Windows (Prompt de Comando ou PowerShell):**
       ```cmd
       .\.virtual_env_teo\Scripts\activate
       ```
       Se estiver usando PowerShell e encontrar um erro de política de execução, pode ser necessário executar `Set-ExecutionPolicy Unrestricted -Scope Process` antes e depois reverter para a política anterior.

   * **No macOS e Linux (Bash/Zsh):**
       ```bash
       source .virtual_env_teo/bin/activate
       ```
   Após a ativação, o nome do ambiente virtual (`.virtual_env_teo`) deverá aparecer no início do prompt do seu terminal, indicando que o ambiente está ativo.



**4. Instale as Dependências**

   Com o ambiente virtual ativo, instale as bibliotecas listadas no arquivo `requirements.txt` usando o `pip`:
   ```bash
   pip install -r requirements.txt
   ```
   Isso garantirá que a biblioteca `numpy` (e quaisquer outras que você adicionar) seja instalada apenas neste ambiente virtual.

**5. Execute o Script Python**

   Agora você está pronto para executar o script Python do algoritmo de Tempera Simulada. Se o seu arquivo se chama `tempera_simulada.py`, execute-o com:
   ```bash
   python tempera_simulada.py
   ```

**6. Desative o Ambiente Virtual (Opcional)**

   Quando terminar de trabalhar no projeto, você pode desativar o ambiente virtual digitando no terminal:
   ```bash
   deactivate
   ```
   Isso restaurará as configurações do seu shell para o interpretador Python global do sistema.