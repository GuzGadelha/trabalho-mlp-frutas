import csv
from dados import dados_frutas
from mlp import RedeNeuralMLP

# Criamos uma lista de configurações para testar automaticamente
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

# =========================================================
# --- PREPARAÇÃO PARA O MODO AO VIVO ---
# Repare que isso está totalmente encostado na margem esquerda,
# indicando que vai rodar SÓ DEPOIS que todos os testes terminarem.
# =========================================================
print("\n" + "=" * 30)
print("PREPARANDO REDE PARA O MODO AO VIVO")
print("=" * 30)

rede_ao_vivo = RedeNeuralMLP(entradas=2, ocultos=5, saidas=1)

# Treinamento rápido para a demonstração
for epoca in range(15000):
    for entradas, saida_desejada in dados_frutas:
        ocultas, chute = rede_ao_vivo.feedforward(entradas)
        rede_ao_vivo.backpropagation(entradas, ocultas, chute, saida_desejada, 0.5)

print("Rede treinada e pronta!")

while True:
    print("\n--- Classificador de Frutas da Scarlet ---")
    entrada_usuario = input("Digite o TAMANHO e o PESO da fruta (ex: 0.55, 0.60) ou 'sair': ")

    if entrada_usuario.lower() == 'sair':
        print("Encerrando o modo ao vivo. Até logo!")
        break

    try:
        # Converte a entrada em uma lista de floats
        valores = [float(x.strip()) for x in entrada_usuario.split(',')]

        if len(valores) != 2:
            print("Erro: Por favor, insira exatamente dois valores (tamanho e peso).")
            continue

        # A rede faz a predição
        _, chute = rede_ao_vivo.feedforward(valores)

        # Define o resultado
        status = "BOA / EXCELENTE (1.0)" if chute[0] >= 0.5 else "RUIM / PEQUENA (0.0)"

        print(f"\n> Resultado da Rede: {chute[0]:.4f}")
        print(f"> Classificação Final: {status}")

    except ValueError:
        print("Erro: Entrada inválida. Use o formato: 0.5, 0.5")