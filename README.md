# creditRisk_IRB_IFRS9

**Credit risk modeling for Bâle IRB & IFRS 9**  
*Python implementation for retail portfolio*

---

##  Description

Projet complet de modélisation du risque de crédit pour portefeuille retail, conforme aux normes :

| Norme | Périmètre |
|-------|-----------|
| **Bâle IRB** | Capital réglementaire (RWA, EL, UL) |
| **IFRS 9** | Pertes de crédit attendues (ECL), staging, PD lifetime |

---

## Fonctionnalités

| Composant | Description |
|-----------|-------------|
| **PD 1 an** | Scorecard Logit + XGBoost challenger. Validation : AUC, Gini, KS, ROC |
| **PD TTC / PIT** | Distinction **PD TTC** (cycle long pour Bâle) et **PD PIT** (cycle court avec macro pour IFRS 9) |
| **PD Lifetime** | GLM avec shift macro + Analyse de survie (Cox PH) + Matrices de transition |
| **LGD** | Approche micro-structure (probabilité de guérison + sévérité) + downturn |
| **EAD** | Produits engagés + CCF pour lignes non utilisées |
| **Marge de Conservatisme (MoC)** | Intégration des incertitudes d'estimation, de scénario et de modèle |
| **Staging (IFRS 9)** | Allocation Stage 1 / 2 / 3 (SICR + backstop 30 jours) |
| **ECL** | Multi-scénarios (baseline / upside / downside) avec pondération |
| **Stress Testing** | Chocs macro + Reverse stress test (seuil critique) |
| **Capital (Bâle III)** | RWA, EL, UL, shortfall, ratios de capital |

---



---

## Résumé du projet

Ce projet implémente une modélisation complète du risque de crédit conforme aux normes **Bâle IRB** et **IFRS 9**.

### Probabilité de défaut (PD)

| Type | Usage | Méthode |
|------|-------|---------|
| **PD brute** | Score de référence | Logit / XGBoost |
| **PD TTC** | Bâle IRB (capital réglementaire) | Moyenne lissée sur cycle complet |
| **PD PIT** | IFRS 9 (ECL, stress tests) | Ajustement macro `PD_TTC × exp(β × Δmacro)` |
| **PD Lifetime** | ECL Stages 2 & 3 | GLM + Survie (Cox PH) + matrices de transition |

### LGD & EAD

- **LGD** : Approche microstructure (guérison + sévérité) + downturn
- **EAD** : Produits engagés + CCF pour lignes non utilisées

### Marge de Conservatisme (MoC)

- Incertitude d'estimation
- Incertitude de scénario
- Incertitude de modèle

### IFRS 9

- **Staging** : Stage 1 / 2 / 3 (SICR + backstop 30 jours)
- **ECL** : Multi-scénarios (baseline / upside / downside) avec pondération

### Bâle III

- RWA, EL, UL, shortfall, ratios de capital
- Stress tests et reverse stress tests

---


## 🔮 Axes d'amélioration non pris en compte

Le projet actuel est fonctionnel et complet, mais les axes suivants n'ont **pas été implémentés** (réservés pour des versions futures) :

| Axe | Description | Priorité |
|-----|-------------|----------|
| **API REST** | Exposer les modèles via FastAPI/Flask pour utilisation en production | Haute |
| **Base de données** | Stocker les résultats (ECL, RWA) dans PostgreSQL ou MongoDB | Haute |
| **Dashboard interactif** | Visualisation des métriques avec Power BI, Tableau ou Plotly Dash | Haute |
| **CI/CD** | Automatiser les tests et le déploiement avec GitHub Actions | Moyenne |
| **Containerisation** | Dockeriser l'application pour faciliter le déploiement | Moyenne |
| **Modèles alternatifs** | Tester LightGBM, CatBoost, réseaux de neurones pour la PD | Moyenne |
| **Scoring comportemental** | Intégrer des données transactionnelles en temps réel | Moyenne |
| **Validation croisée temporelle** | Backtesting sur plusieurs cycles économiques | Haute |
| **Modèle de corrélation** | Ajouter une matrice de corrélation PD/LGD/EAD (modèle copule) | Haute |
| **Documentation automatique** | Générer la doc avec Sphinx ou MkDocs | Basse |
| **Monitoring** | Mettre en place un suivi des dérives de modèles (concept drift) | Haute |
| **Scénarios personnalisés** | Interface utilisateur pour définir ses propres scénarios macro | Basse |
| **Passage à l'échelle** | Traitement de millions de lignes avec Spark ou Dask | Haute |
| **Cloud** | Déploiement sur AWS SageMaker, Azure ML ou GCP | Moyenne |

---

## 👤 Auteur

**Rodrigue KOFFI**

| Titre / Statut | Description |
|----------------|-------------|
| **Actuarial Analyst** | Provisions techniques, best estimate, passif prudentiel (Solvabilité II) |
| **Credit & Financial Modeling** | Modèles de risque de crédit (Bâle IRB, IFRS 9) et modélisation financière |
| **FRM Part I Candidate** | Financial Risk Manager (GARP) |

### Contact

- **GitHub** : [@rodrigue-koffi](https://github.com/rodrigue-koffi)
- **Email** : kkanrodrigue@gmail.com

---
## Installation

```bash
# Cloner le dépôt
git clone https://github.com/rodrigue-koffi/creditRisk_IRB_IFRS9.git
cd creditRisk_IRB_IFRS9

# Installer les dépendances
pip install -r requirements.txt
---
**Made with by Rodrigue KOFFI**
