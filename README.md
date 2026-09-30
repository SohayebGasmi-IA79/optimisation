# TP — Optimisation implémentée avec NumPy

<div align="center">
<h2>Régression, classification et clustering sans framework d'apprentissage</h2>
  <p>
    Comparaison expérimentale de <strong>7 méthodes d'optimisation</strong> sur
    <strong>8 jeux de données</strong>, avec implémentation manuelle des modèles,
    gradients, métriques, recherches de pas et visualisations.
  </p>
  <p>
    <a href="#objectifs">Objectifs</a> ·
    <a href="#méthodes">Méthodes</a> ·
    <a href="#installation">Installation</a> ·
    <a href="#exécution">Exécution</a> ·
    <a href="#résultats">Résultats</a>
  </p>
</div>

---

## Présentation

Ce projet correspond à un travail pratique consacré à l'optimisation numérique avec **NumPy**. L'objectif est de construire un pipeline expérimental complet sans utiliser d'optimiseur prêt à l'emploi de scikit-learn, PyTorch ou TensorFlow.

Le notebook `tp.ipynb` implémente et compare :

- **7 optimiseurs** : GD, GD-LS, SGD, Momentum, AdaGrad, RMSprop et Adam ;

- **3 familles de problèmes** : régression, classification et clustering ;

- **8 jeux de données** : régression non linéaire, Diabetes, Spiral3, Iris, Blobs3, Wine, MNIST et Fashion-MNIST ;

- le calcul manuel des gradients et des fonctions de coût ;

- la recherche d'hyperparamètres sur l'ensemble de validation ;

- l'arrêt anticipé, la détection de divergence et le suivi du coût ;

- des métriques, courbes, matrices de confusion, projections PCA et visualisations de centroïdes ;

- l'export des résultats bruts en JSON et des sections LaTeX pour un rapport.

> **Important :** les valeurs affichées dans ce README sont les sorties enregistrées dans le notebook fourni. Elles peuvent légèrement varier selon les fichiers de données disponibles, les versions des bibliothèques et les paramètres d'exécution.

## Objectifs

1. Implémenter une boucle d'optimisation commune à plusieurs algorithmes.

1. Vérifier les gradients par différences finies centrées.

1. Comparer les méthodes sur des problèmes convexes et non convexes.

1. Mesurer à la fois la qualité finale et le coût de calcul.

1. Étudier l'effet du choix du pas d'apprentissage et de la taille des mini-lots.

1. Produire des résultats reproductibles grâce à des graines aléatoires explicites.

<a name="méthodes"></a>

## Méthodes d'optimisation

| Méthode | Principe utilisé dans le notebook | Type de gradient |
| --- | --- | --- |
| **GD** | Descente de gradient à lot complet | Gradient sur toutes les observations |
| **GD-LS** | GD avec recherche linéaire | Pas exact pour les quadratiques, sinon grille + section dorée |
| **SGD** | Descente avec mini-lots mélangés à chaque époque | Gradient sur un mini-lot |
| **Momentum** | Accumulation exponentielle des gradients | `m = μm + g` |
| **AdaGrad** | Adaptation coordonnée du pas selon l'historique carré | Accumulateur monotone |
| **RMSprop** | Moyenne exponentielle des carrés des gradients | Accumulateur décroissant |
| **Adam** | Momentum du premier et du second moment avec correction de biais | `m`, `s`, corrections `β₁`, `β₂` |

Constantes utilisées :

```python
METHODS = ["GD", "GD-LS", "SGD", "Momentum", "AdaGrad", "RMSprop", "Adam"]
MU = RHO = B1 = 0.9
B2 = 0.999
EPS = 1e-8
```

### Boucle d'optimisation commune

Le cœur du projet est la fonction `run`. Chaque problème fournit une interface uniforme :

- `init(seed)` : initialisation des paramètres ;

- `loss(theta, idx)` : fonction de coût ;

- `grad(theta, idx)` : gradient ;

- `monitor(theta)` : métriques d'entraînement et de validation.

```python
def run(prob, method, eta, epochs, batch, seed, patience=None):
    rng = np.random.default_rng(1000 + seed)
    theta = prob.init(seed).copy()
    m = np.zeros_like(theta)
    s = np.zeros_like(theta)
    t = 0

    for epoch in range(1, epochs + 1):
        if method in {"GD", "GD-LS"}:
            batches = [None]
        else:
            order = rng.permutation(prob.n_train)
            batches = [order[i:i + batch]
                       for i in range(0, prob.n_train, batch)]

        for idx in batches:
            g = prob.grad(theta, idx)
            t += 1

            if method == "GD-LS":
                theta -= line_step(prob, theta, g, counter) * g
            elif method in ("GD", "SGD"):
                theta -= eta * g
            elif method == "Momentum":
                m = MU * m + g
                theta -= eta * m
            elif method == "AdaGrad":
                s += g * g
                theta -= eta * g / (np.sqrt(s) + EPS)
            elif method == "RMSprop":
                s = RHO * s + (1 - RHO) * g * g
                theta -= eta * g / (np.sqrt(s) + EPS)
            elif method == "Adam":
                m = B1 * m + (1 - B1) * g
                s = B2 * s + (1 - B2) * g * g
                m_hat = m / (1 - B1 ** t)
                s_hat = s / (1 - B2 ** t)
                theta -= eta * m_hat / (np.sqrt(s_hat) + EPS)

        # Le notebook conserve le meilleur état selon le critère de validation.
```

### Recherche linéaire GD-LS

Pour une fonction quadratique dont la matrice hessienne `H` est connue, le pas est calculé analytiquement :

```python
def line_step(prob, theta, g, counter):
    if hasattr(prob, "H"):
        counter.hess += 1
        denominator = float(g @ (prob.H @ g))
        return float((g @ g) / denominator) \
            if denominator > 0 else float(prob.eta_max)

    # Pour les autres problèmes : grille géométrique puis raffinement
    etas = prob.eta_max * 0.5 ** np.arange(12)
    values = [prob.loss(theta - eta * g) for eta in etas]
    k = int(np.argmin(values))
    return float(etas[k])
```

## Problèmes étudiés

### 1. Régression

Trois modèles sont comparés :

- **OLS** : régression linéaire sans régularisation ;

- **Ridge** : régression linéaire avec pénalité L2 ;

- **Ridge à noyau RBF** : représentation non linéaire via une matrice de Gram.

La fonction objectif est :

$$
J(\theta) = \frac{1}{2n}\|A\theta-y\|^2
           + \frac{\lambda}{2}\theta^\top R\theta.
$$

Son gradient est :

$$
\nabla J(\theta) = \frac{1}{n}A^\top(A\theta-y)+\lambda R\theta.
$$

Implémentation NumPy utilisée :

```python
class RegProblem:
    def loss(self, theta, idx=None):
        A, y = (self.A, self.Y) if idx is None else \\
               (self.A[idx], self.Y[idx])
        residual = A @ theta - y
        return (0.5 * np.mean(residual ** 2)
                + 0.5 * self.lam * theta @ (self.R @ theta))

    def grad(self, theta, idx=None):
        A, y = (self.A, self.Y) if idx is None else \\
               (self.A[idx], self.Y[idx])
        return (A.T @ (A @ theta - y) / len(y)
                + self.lam * (self.R @ theta))
```

Le noyau RBF est calculé sans boucle sur les paires :

```python
def rbf(A, B, sigma):
    d2 = ((A * A).sum(1)[:, None]
          + (B * B).sum(1)[None, :]
          - 2 * A @ B.T)
    return np.exp(-np.maximum(d2, 0) / (2 * sigma ** 2))
```

Métriques : **MSE**, **RMSE**, **MAE** et **R²**. Le critère de sélection indiqué dans les sorties est le **RMSE de validation**.

### 2. Classification par MLP à trois couches

Le réseau possède deux couches cachées `tanh` et une sortie `softmax` :

```
Entrée → Dense + tanh → Dense + tanh → Dense + softmax
```

Architectures utilisées :

| Jeu de données | Architecture |
| --- | --- |
| Spiral3 | `[2, 32, 16, 3]` |
| Iris | `[4, 32, 16, 3]` |
| MNIST | `[784, 128, 64, 10]` |
| Fashion-MNIST | `[784, 128, 64, 10]` |

La perte est l'entropie croisée avec régularisation L2 sur les poids :

```python
def softmax_probabilities(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def cross_entropy(logits, y):
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=1))
    return np.mean(log_sum_exp - shifted[np.arange(len(y)), y])
```

La rétropropagation est codée explicitement :

```python
D3 = probabilities.copy()
D3[np.arange(batch_size), y] -= 1
D3 /= batch_size

D2 = (D3 @ W3.T) * (1 - H2 ** 2)
D1 = (D2 @ W2.T) * (1 - H1 ** 2)
```

Métriques calculées :

- exactitude (`accuracy`) ;

- précision macro ;

- rappel macro ;

- F1 macro et F1 par classe ;

- entropie croisée ;

- matrice de confusion.

### 3. Clustering par optimisation des centroïdes

Le clustering utilise `K` centroïdes optimisés par gradient. La fonction de coût est une version différentiable de l'inertie, avec affectations souples contrôlées par `tau` :

```python
class Cluster:
    @staticmethod
    def D(C, X):
        # Distances euclidiennes au carré, vectorisées.
        return np.maximum(
            (X * X).sum(1)[:, None]
            + (C * C).sum(1)[None, :]
            - 2 * X @ C.T,
            0,
        )

    def grad(self, theta, idx=None):
        X = self.Xtr if idx is None else self.Xtr[idx]
        C = theta.reshape(self.K, -1)
        distances = self.D(C, X)
        Q = np.exp(-distances / self.tau)
        Q /= Q.sum(axis=1, keepdims=True)
        return (2 / len(X) *
                (Q.sum(0)[:, None] * C - Q.T @ X)).ravel()
```

Une référence indépendante est également fournie avec **Lloyd**, l'équivalent du k-means à affectations dures :

```python
def lloyd(prob, seed, iters):
    centers = prob.init(seed).reshape(prob.K, -1).copy()
    for _ in range(iters):
        assignments = prob.D(centers, prob.Xtr).argmin(axis=1)
        for k in range(prob.K):
            mask = assignments == k
            if mask.any():
                centers[k] = prob.Xtr[mask].mean(axis=0)
    return centers
```

Métriques : **inertie**, nombre de clusters non vides, silhouette, **ARI** et **NMI** lorsque des étiquettes de référence existent.

## Jeux de données

| Famille | Jeux de données | Format |
| --- | --- | --- |
| Régression | `regression_nonlinear.csv`, `regression_diabetes.csv` | CSV avec `split` |
| Classification | `classification_spiral3.csv`, `classification_iris.csv` | CSV avec étiquette et `split` |
| Classification image | `mnist_50k_10k_10k.npz`, `fashion_mnist_50k_10k_10k.npz` | NPZ : 50k train, 10k validation, 10k test |
| Clustering | `clustering_blobs3.csv`, `clustering_wine.csv` | CSV avec caractéristiques et éventuellement `true_label` |

Le notebook cherche les données dans les emplacements suivants :

```
.
optimisation_datasets/
data/
optimisation_datasets.zip
```

Les variables sont standardisées à partir de l'ensemble d'entraînement uniquement :

```python
def std_fit(X_train, *others):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1
    return [(X - mu) / sd for X in (X_train,) + others], mu, sd
```

Si un fichier réel manque, le notebook peut reconstruire Diabetes, Iris et Wine avec scikit-learn, et tenter de télécharger MNIST/Fashion-MNIST. Les découpages reconstruits sont aléatoires avec la graine `0` et ne garantissent donc pas de reproduire exactement les partitions de l'enseignant.

## Installation

<a name="installation"></a>

### Prérequis

- Python 3.9 ou plus récent ;

- Jupyter Notebook ou JupyterLab ;

- NumPy ;

- Matplotlib ;

- IPython ;

- scikit-learn uniquement pour la reconstruction optionnelle de jeux de données manquants.

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\\Scripts\\activate    # Windows

python -m pip install --upgrade pip
python -m pip install numpy matplotlib ipython scikit-learn jupyter
```

### Organisation recommandée

```
.
├── tp.ipynb
├── README.md
├── optimisation_datasets/
│   ├── regression_diabetes.csv
│   ├── classification_iris.csv
│   ├── clustering_wine.csv
│   ├── mnist_50k_10k_10k.npz
│   └── fashion_mnist_50k_10k_10k.npz
├── regression_nonlinear.csv
├── classification_spiral3.csv
├── clustering_blobs3.csv
└── results/
    ├── figs/
    ├── raw/
    └── sections/
```

## Exécution

<a name="exécution"></a>

1. Placer `tp.ipynb` et les jeux de données à la racine du projet.

1. Lancer Jupyter :

   ```bash
   jupyter notebook
   ```

1. Ouvrir `tp.ipynb`.

1. Exécuter les cellules dans l'ordre.

1. Pour une vérification rapide, modifier :

   ```python
   QUICK = True
   ```

1. Pour les résultats complets du rapport, utiliser :

   ```python
   QUICK = False
   SEEDS = [42, 123, 2024]
   EP_REG, EP_CLS, EP_CLU, EP_IMG = 500, 300, 300, 8
   OUT = "results"
   ```

La configuration complète enregistrée dans le notebook est :

```python
from types import SimpleNamespace

A = SimpleNamespace(
    seeds=[42, 123, 2024],
    ep_reg=500,
    ep_cls=300,
    ep_clu=300,
    ep_img=8,
    out="results",
)
```

### Lancer les expériences

```python
run_reg(A)   # OLS, Ridge et Kernel sur nonlinear + diabetes
run_cls(A)   # MLP sur Spiral3, Iris, MNIST et Fashion-MNIST
run_clu(A)   # Centroïdes sur Blobs3, Wine, MNIST et Fashion-MNIST
```

Pour afficher les figures déjà produites :

```python
show("reg_nonlinear")
show("cls_spiral3")
show("clu_mnist")
```

## Résultats enregistrés

Le critère `crit` est le meilleur critère de validation moyen enregistré pour chaque méthode. Il correspond à :

- **RMSE de validation** pour la régression ;

- **entropie croisée de validation** pour la classification ;

- **coût différentiable ****`J_tau`**** de validation** pour le clustering.

### Régression non linéaire

| Modèle | Meilleure méthode observée | Critère enregistré |
| --- | --- | --- |
| OLS | SGD | 0.7176 |
| Ridge | SGD | 0.7176 |
| Kernel RBF | Adam | 0.1901 |

Le noyau RBF améliore fortement la représentation sur ce jeu de données non linéaire. La meilleure valeur enregistrée est obtenue avec Adam pour le modèle Kernel.

### Diabetes

| Modèle | Meilleure méthode observée | Critère enregistré |
| --- | --- | --- |
| OLS | AdaGrad | 50.78 |
| Ridge | Adam | 50.57 |
| Kernel RBF | Adam | 51.09 |

Sur Diabetes, les modèles linéaires régularisés ou non sont plus compétitifs que le noyau RBF dans cette configuration.

### Classification

| Jeu | Meilleure méthode observée | Critère de validation |
| --- | --- | --- |
| Spiral3 | RMSprop | 0.002003 |
| Iris | Momentum | 0.0008412 |
| MNIST | Momentum | 0.09585 |
| Fashion-MNIST | Adam | 0.3271 |

### Clustering

| Jeu | Meilleure méthode observée | Critère de validation |
| --- | --- | --- |
| Blobs3 | Toutes, ex æquo | 0.5643 |
| Wine | Pratiquement ex æquo | 7.454 |
| MNIST | AdaGrad | 49.21 |
| Fashion-MNIST | Momentum | 43.00 |

Pour Blobs3 et Wine, les méthodes convergent vers des solutions proches dans les sorties du notebook. Sur les images, les mini-lots et les optimiseurs adaptatifs ont un impact plus visible.

## Fichiers générés

Après exécution, le dossier `results/` contient :

```
results/
├── figs/
│   ├── reg_*_curves.png
│   ├── reg_*_pred.png
│   ├── reg_*_resid.png
│   ├── cls_*_curves.png
│   ├── cls_*_cm.png
│   ├── cls_*_f1.png
│   ├── clu_*_curves.png
│   ├── clu_*_pca.png
│   └── clu_*_centroids.png
├── raw/
│   └── *.json
└── sections/
    └── *.tex
```

Les paramètres `theta` sont volontairement retirés des JSON bruts pour conserver des fichiers lisibles et compacts :

```python
def dump(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    compact = {
        method: [
            {key: value for key, value in run.items() if key != "theta"}
            for run in runs
        ]
        for method, runs in results.items()
    }
    json.dump(compact, open(path, "w"), default=jd)
```

## Vérification des gradients

Le notebook compare le gradient analytique à une approximation par différences finies centrales :

```python
def gradcheck(prob, theta, idx=None, k=8, h=1e-5, seed=0):
    analytic = prob.grad(theta, idx)
    rng = np.random.default_rng(seed)
    errors = []

    for i in rng.choice(theta.size, min(k, theta.size), replace=False):
        direction = np.zeros_like(theta)
        direction[i] = h
        numerical = (
            prob.loss(theta + direction, idx)
            - prob.loss(theta - direction, idx)
        ) / (2 * h)
        errors.append(
            abs(numerical - analytic[i])
            / max(abs(numerical), abs(analytic[i]), 1e-12)
        )

    return float(max(errors))
```

Cette vérification est particulièrement importante pour :

- la régularisation Ridge ;

- la rétropropagation du MLP ;

- le gradient des centroïdes avec affectations souples ;

- la stabilité numérique de la softmax et du `log-sum-exp`.

## Reproductibilité et limites

- Les graines principales sont `[42, 123, 2024]`.

- La permutation des mini-lots utilise `1000 + seed`.

- Le meilleur état selon la validation est conservé, plutôt que le dernier état.

- Les données sont standardisées sans utiliser les ensembles validation/test pour ajuster les statistiques.

- Les images sont aplaties en 784 variables puis ramenées dans une échelle comparable à `[0, 1]`.

- La silhouette est coûteuse en mémoire et en temps car elle utilise une matrice de distances.

- Les résultats affichés proviennent d'une exécution déjà enregistrée ; ils ne constituent pas une étude statistique exhaustive.

- `EP_IMG = 8` est volontairement faible pour limiter le temps de calcul. Il peut être augmenté si les courbes continuent à décroître.

- Les reconstructions automatiques de datasets ne doivent pas être mélangées silencieusement avec les partitions officielles : il faut l'indiquer dans un rapport.

<details>
<summary><strong>Liste des fonctions principales du notebook</strong></summary>

| Domaine | Fonctions/classes |
| --- | --- |
| Optimisation | `Counter`, `golden`, `line_step`, `run`, `gradcheck` |
| Régression | `rbf`, `RegProblem`, `exp_reg` |
| Classification | `MLP`, `cls_metrics`, `exp_cls`, `confusion_fig` |
| Clustering | `Cluster`, `lloyd`, `ari`, `nmi`, `silhouette`, `cluster_metrics`, `exp_clu` |
| Données | `find_file`, `read_csv`, `split_masks`, `std_fit`, `load_npz`, `build_missing` |
| Expériences | `eta_grid`, `tune`, `run_all`, `best_method`, `run_reg`, `run_cls`, `run_clu` |
| Sorties | `curves`, `dump`, `Sec`, `table`, `show` |

</details> <details>
<summary><strong>Exemple minimal d'une exécution sur un seul problème</strong></summary>

```python
# Exemple conceptuel : prob doit fournir init, grad, loss et monitor.
result = run(
    prob=prob,
    method="Adam",
    eta=1e-2,
    epochs=100,
    batch=32,
    seed=42,
    patience=20,
)

print("Meilleur critère :", result["best_crit"])
print("Époque optimale :", result["best_ep"])
print("Temps :", result["time"])
print("Compteurs :", result["cnt"])
```

</details>

## Références conceptuelles

- Descente de gradient et recherche linéaire ;

- Momentum, AdaGrad, RMSprop et Adam ;

- régression OLS et régularisation Ridge ;

- noyaux RBF et méthodes à noyau ;

- rétropropagation, `tanh`, softmax et entropie croisée ;

- k-means/Lloyd, silhouette, ARI et NMI ;

- normalisation des données et séparation train/validation/test.


</div>