import csv
from dados import dados_frutas
from mlp import RedeNeuralMLP

# Criamos uma lista de configurações para testar automaticamente
# Adicionei uma 4ª configuração para termos bastante dados para mostrar
configuracoes_para_testar = [
    {"ocultos": 2, "taxa": 0.1, "epocas": 5000},  # Configuração mais fraca
    {"ocultos": 3, "taxa": 0.3, "epocas": 10000},  # Configuração média
    {"ocultos": 4, "taxa": 0.2, "epocas": 12000},  # Configuração intermediária
    {"ocultos": 5, "taxa": 0.5, "epocas": 15000},  # Configuração forte/rápida
]

print("Iniciando testes de múltiplas configurações e salvando no CSV...\n")

# Cria e abre o arquivo CSV para salvar os resultados
with open('resultados_testes.csv', 'w', newline='', encoding='utf-8') as arquivo_csv:
    escritor = csv.writer(arquivo_csv)

    # Escreve o cabeçalho da tabela no arquivo
    escritor.writerow(['Teste', 'Neuronios Ocultos', 'Taxa de Aprendizado', 'Epocas', 'Erro Final Total', 'Acertos',
                       'Total de Frutas'])

    # O enumerate ajuda a contar qual é o número do teste (1, 2, 3...)
    for indice, config in enumerate(configuracoes_para_testar):
        numero_teste = indice + 1
        print(
            f"--- Rodando Teste {numero_teste}: {config['ocultos']} Ocultos | Taxa: {config['taxa']} | Épocas: {config['epocas']} ---")

        # Cria uma rede nova, do zero, para cada teste
        rede = RedeNeuralMLP(entradas=2, ocultos=config["ocultos"], saidas=1)

        # Inicia o treinamento (Backpropagation)
        for epoca in range(config["epocas"]):
            for entradas, saida_desejada in dados_frutas:
                ocultas, chute = rede.feedforward(entradas)
                rede.backpropagation(entradas, ocultas, chute, saida_desejada, config["taxa"])

        # Avaliação de desempenho após o treino
        erro_final = 0.0
        acertos = 0
        total_frutas = len(dados_frutas)

        for entradas, saida_desejada in dados_frutas:
            _, chute = rede.feedforward(entradas)
            erro_final += abs(saida_desejada[0] - chute[0])

            # Se o chute for maior ou igual a 0.5, é 1.0 (Boa/Excelente)
            classificacao = 1.0 if chute[0] >= 0.5 else 0.0

            if classificacao == saida_desejada[0]:
                acertos += 1

        print(f"Resultado -> Erro: {erro_final:.4f} | Acertos: {acertos}/{total_frutas}\n")

        # Salva essa linha de resultado diretamente no arquivo CSV
        escritor.writerow([
            numero_teste,
            config["ocultos"],
            config["taxa"],
            config["epocas"],
            round(erro_final, 4),
            acertos,
            total_frutas
        ])

print("Testes concluídos! Verifique o arquivo 'resultados_testes.csv' na sua pasta.")