# Árvore de decisão do zero!!! (CART para classificação) 
# Segue os algoritmos dos slides (CresceÁrvore, MelhorCorte, Prediz) usando o índice de Gini.
from collections import Counter


def gini(y):
    # G_m = 1 - soma_k p_mk^2, com p_mk a proporção da classe k no nó
    n = len(y)
    return 1 - sum((c / n) ** 2 for c in Counter(y).values())

def melhor_corte(X, y):
    # Procura a pergunta (j, tau) que minimiza o Gini ponderado dos filhos:
    # s = n_L/n * G(L) + n_R/n * G(R)
    # Devolve None se não houver pergunta não trivial (Cand = vazio).
    n, p = len(X), len(X[0])
    melhor = None  # (j, tau, s)
    for j in range(p):
        valores = sorted(set(linha[j] for linha in X))
        # limiares candidatos: pontos médios entre valores distintos consecutivos
        for a, b in zip(valores, valores[1:]):
            tau = (a + b) / 2
            y_esq = [y[i] for i in range(n) if X[i][j] < tau]
            y_dir = [y[i] for i in range(n) if X[i][j] >= tau]
            s = len(y_esq) / n * gini(y_esq) + len(y_dir) / n * gini(y_dir)
            # desigualdade estrita: em empate fica o primeiro encontrado (menor j)
            if melhor is None or s < melhor[2]:
                melhor = (j, tau, s)
    return melhor

def cresce_arvore(X, y):
    # Folha: dicionário {"classe": ...}; nó interno: {"j", "tau", "esq", "dir"}.
    corte = melhor_corte(X, y)
    # Condição de parada mínima: nó puro (Gini = 0) ou sem pergunta possível
    if gini(y) == 0 or corte is None:
        return {"classe": Counter(y).most_common(1)[0][0]}  # classe majoritária
    j, tau, _ = corte
    esq = [i for i in range(len(X)) if X[i][j] < tau]
    dir_ = [i for i in range(len(X)) if X[i][j] >= tau]
    return {
        "j": j,
        "tau": tau,
        "esq": cresce_arvore([X[i] for i in esq], [y[i] for i in esq]),
        "dir": cresce_arvore([X[i] for i in dir_], [y[i] for i in dir_]),
    }

def prediz(arvore, x):
    # Desce da raiz até uma folha respondendo às perguntas "x_j < tau?"
    while "classe" not in arvore:
        arvore = arvore["esq"] if x[arvore["j"]] < arvore["tau"] else arvore["dir"]
    return arvore["classe"]

def mostra(arvore, nomes, nivel=0):
    # Imprime a árvore com indentação
    pad = "    " * nivel
    if "classe" in arvore:
        print(f"{pad}-> {arvore['classe']}")
        return
    print(f"{pad}{nomes[arvore['j']]} < {arvore['tau']}?")
    mostra(arvore["esq"], nomes, nivel + 1)
    mostra(arvore["dir"], nomes, nivel + 1)
