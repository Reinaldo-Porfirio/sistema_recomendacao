import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Criar um conjunto de dados simulado (usuários x filmes/livros)
data = {
    'Usuario': ['Ana', 'Bruno', 'Clara', 'Daniel'],
    'Filme1': [5, 3, 0, 4],  # 0 indica que o usuário não avaliou
    'Filme2': [4, 0, 5, 2],
    'Filme3': [0, 5, 3, 0],
    'Filme4': [2, 4, 0, 5]
}
df = pd.DataFrame(data)

def recomendar_itens(usuario, df, n_vizinhos=2):
    # Preparar a matriz de avaliações (excluindo a coluna 'Usuario')
    matriz_avaliacoes = df.drop('Usuario', axis=1).values
    # Inicializar o modelo KNN
    knn = NearestNeighbors(n_neighbors=n_vizinhos, metric='cosine')
    knn.fit(matriz_avaliacoes)
    
    # Encontrar o índice do usuário
    idx_usuario = df.index[df['Usuario'] == usuario].tolist()[0]
    
    # Encontrar os vizinhos mais próximos
    distancias, indices = knn.kneighbors([matriz_avaliacoes[idx_usuario]])
    
    # Itens não avaliados pelo usuário
    itens_nao_avaliados = np.where(matriz_avaliacoes[idx_usuario] == 0)[0]
    recomendacoes = []
    
    # Calcular a pontuação média dos vizinhos para itens não avaliados
    for item in itens_nao_avaliados:
        pontuacao = 0
        peso_total = 0
        for i, distancia in zip(indices[0], distancias[0]):
            if matriz_avaliacoes[i][item] != 0:
                peso = 1 - distancia  # Inverter a distância para peso
                pontuacao += matriz_avaliacoes[i][item] * peso
                peso_total += peso
        if peso_total > 0:
            pontuacao /= peso_total
            recomendacoes.append((df.columns[item + 1], round(pontuacao, 2)))
    
    # Ordenar por pontuação
    recomendacoes.sort(key=lambda x: x[1], reverse=True)
    return recomendacoes

# Testar o sistema
usuario = 'Clara'
recomendacoes = recomendar_itens(usuario, df)
print(f"Recomendações para {usuario}:")
for item, pontuacao in recomendacoes:
    print(f"{item}: {pontuacao}")