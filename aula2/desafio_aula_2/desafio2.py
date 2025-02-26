# 1. Definir as listas x e y
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# 2. Calcular as médias de x e y
media_x = sum(x) / len(x)
media_y = sum(y) / len(y)


# 3. Inicializar as variáveis para os somatórios
soma_x = 0
soma_xy = 0

# 4. Utilizar um loop para calcular:
#    - A soma de (x_i - média_x) * (y_i - média_y)
#    - A soma de (x_i - média_x)²
soma_x = sum((xi - media_x) ** 2 for xi in x)
soma_xy = sum((xi - media_x) * (yi - media_y) for xi, yi in zip(x, y))


# 5. Calcular beta1 e beta0 usando as fórmulas dos mínimos quadrados
beta1 = soma_xy / soma_x
beta0 = media_y - beta1 * media_x
# 6. Imprimir os resultados
print(f'Valores do beta1 sao: {beta1}')
print(f'Valores do beta0 sao: {beta0}')