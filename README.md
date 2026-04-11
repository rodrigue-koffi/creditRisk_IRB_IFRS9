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

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/rodrigue-koffi/creditRisk_IRB_IFRS9.git
cd creditRisk_IRB_IFRS9

# Installer les dépendances
pip install -r requirements.txt
