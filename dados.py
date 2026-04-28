import csv

# Nossa lista vazia que vai guardar os dados
dados_frutas = []

# Abre o arquivo CSV em modo de leitura ('r')
with open('dados.csv', 'r', encoding='utf-8') as arquivo:
    leitor = csv.reader(arquivo)

    # Pula a primeira linha (o cabeçalho)
    next(leitor)

    # Passa por cada linha do arquivo
    for linha in leitor:
        # O leitor traz tudo como texto (string). 
        # Precisamos converter as colunas 0, 1 e 2 para números decimais (float)
        entrada1 = float(linha[0])
        entrada2 = float(linha[1])
        saida_desejada = float(linha[2])

        # Ignoramos a coluna 3 (o significado em texto) porque a rede só entende números!

        # Adicionamos no formato que a MLP exige: [[entradas], [saida]]
        dados_frutas.append([[entrada1, entrada2], [saida_desejada]])