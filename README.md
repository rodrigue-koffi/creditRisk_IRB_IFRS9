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

## 📁 Structure du projet
---
creditRisk_IRB_IFRS9/
├── data/ # Données brutes et traitées
│ ├── raw/ # Données brutes
│ ├── processed/ # Données traitées
│ ├── external/ # Données externes
│ └── macro/ # Données macroéconomiques
├── logs/ # Logs d'exécution
├── notebooks/ # Jupyter notebooks
├── reports/ # Rapports et figures
│ ├── figures/ # Graphiques
│ ├── tables/ # Tableaux
│ └── html/ # Rapports HTML
├── src/ # Code source principal
│ ├── dataPreparation/ # Chargement et nettoyage
│ ├── irb/ # Bâle IRB (PD 1 an, RWA)
│ ├── ifrs9/ # IFRS 9 (PD lifetime, ECL)
│ ├── stressTesting/ # Stress test & reverse stress
│ ├── resilience/ # Métriques de résilience
│ ├── validation/ # Validation des modèles
│ ├── utils/ # Utilitaires
│ └── orchestration/ # Orchestrateur principal
├── tests/ # Tests unitaires
│ ├── unit/ # Tests unitaires
│ ├── integration/ # Tests d'intégration
│ └── fixtures/ # Fixtures de test
├── requirements.txt # Dépendances Python
└── README.md # Documentation

---

