import math
import random

def sigmoide(x):
    """Esmaga o valor para o intervalo entre 0 e 1."""
    return 1.0 / (1.0 + math.exp(-x))

def derivada_sigmoide(x):
    """Calcula a taxa de variação para ajudar a corrigir os erros da rede."""
    return x * (1.0 - x)

class RedeNeuralMLP:
    def __init__(self, entradas, ocultos, saidas):
        # Quantidade de neurônios em cada camada
        self.entradas = entradas
        self.ocultos = ocultos
        self.saidas = saidas
        
        # Pesos: sorteamos números aleatórios entre -1 e 1
        # Conexões da Camada de Entrada para a Camada Oculta
        self.pesos_entrada_oculta = [
            [random.uniform(-1, 1) for _ in range(self.ocultos)] 
            for _ in range(self.entradas)
        ]
        
        # Conexões da Camada Oculta para a Camada de Saída
        self.pesos_oculta_saida = [
            [random.uniform(-1, 1) for _ in range(self.saidas)] 
            for _ in range(self.ocultos)
        ]
        
        # Vieses (Bias): um para cada neurônio oculto e um para o de saída
        self.bias_oculta = [random.uniform(-1, 1) for _ in range(self.ocultos)]
        self.bias_saida = [random.uniform(-1, 1) for _ in range(self.saidas)]


    def feedforward(self, entradas_da_fruta):
        """
        Recebe as medidas de uma fruta, passa pelos neurônios
        e retorna o chute da rede.
        """
        # 1. Calcular os valores da Camada Oculta
        ativacoes_ocultas = []
        for i in range(self.ocultos):
            # Começa com o valor do viés (bias) do neurônio
            soma = self.bias_oculta[i]

            # Multiplica cada entrada pelo peso da sua conexão
            for j in range(self.entradas):
                soma += entradas_da_fruta[j] * self.pesos_entrada_oculta[j][i]

            # Passa a soma na função de ativação e guarda o resultado
            ativacao = sigmoide(soma)
            ativacoes_ocultas.append(ativacao)

        # 2. Calcular os valores da Camada de Saída
        ativacoes_saida = []
        for k in range(self.saidas):
            soma = self.bias_saida[k]

            # Agora a "entrada" são as ativações que vieram da camada oculta
            for i in range(self.ocultos):
                soma += ativacoes_ocultas[i] * self.pesos_oculta_saida[i][k]

            ativacao = sigmoide(soma)
            ativacoes_saida.append(ativacao)

        # Retornamos as duas listas porque vamos precisar das
        # ativações ocultas mais tarde no Backpropagation!
        return ativacoes_ocultas, ativacoes_saida

    def backpropagation(self, entradas, ativacoes_ocultas, ativacoes_saida, saida_desejada, taxa_aprendizado):
        """
        Calcula o erro e ajusta os pesos e vieses de trás para frente.
        """
        # --- 1. Calcular o Erro e o Delta da Camada de Saída ---
        deltas_saida = []
        for k in range(self.saidas):
            erro = saida_desejada[k] - ativacoes_saida[k]
            # O Delta é o erro multiplicado pela derivada da ativação
            delta = erro * derivada_sigmoide(ativacoes_saida[k])
            deltas_saida.append(delta)

        # --- 2. Calcular o Erro e o Delta da Camada Oculta ---
        deltas_oculta = []
        for i in range(self.ocultos):
            erro_oculto = 0.0
            # Pega a culpa que a camada de saída jogou para este neurônio
            for k in range(self.saidas):
                erro_oculto += deltas_saida[k] * self.pesos_oculta_saida[i][k]

            delta = erro_oculto * derivada_sigmoide(ativacoes_ocultas[i])
            deltas_oculta.append(delta)

        # --- 3. Atualizar os Pesos e Vieses (Oculta -> Saída) ---
        for i in range(self.ocultos):
            for k in range(self.saidas):
                # Ajuste = taxa * delta_destino * ativacao_origem
                ajuste = taxa_aprendizado * deltas_saida[k] * ativacoes_ocultas[i]
                self.pesos_oculta_saida[i][k] += ajuste

        for k in range(self.saidas):
            self.bias_saida[k] += taxa_aprendizado * deltas_saida[k]

        # --- 4. Atualizar os Pesos e Vieses (Entrada -> Oculta) ---
        for j in range(self.entradas):
            for i in range(self.ocultos):
                ajuste = taxa_aprendizado * deltas_oculta[i] * entradas[j]
                self.pesos_entrada_oculta[j][i] += ajuste

        for i in range(self.ocultos):
            self.bias_oculta[i] += taxa_aprendizado * deltas_oculta[i]