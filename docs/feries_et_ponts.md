# Fériés, vacances et ponts : comparaison de variantes

`python main_holidays.py` compare quatre variantes du modèle global multi-horizon. La validation couvre huit fenêtres récentes de cinq jours et douze fériés historiques, chacun dans une fenêtre dédiée ; le modèle est réentraîné avant chaque fenêtre. Les scores utilisent uniquement les cibles observées communes aux quatre variantes.

| Variante | Variables calendaires enrichies | Coefficients et régime manuels des ponts |
|---|---|---|
| `reference` | non | oui |
| `sans_pont` | non | non |
| `calendrier` | oui | oui |
| `calendrier_sans_pont` | oui | non |

L'enrichissement ajoute les distances bornées au férié précédent et suivant, le type fixe ou mobile du férié le plus proche, son jour de semaine et une interaction entre proximité d'un férié et vacances scolaires. Le jour de semaine de la date cible figurait déjà dans le modèle. À égale distance entre deux fériés, le prochain est choisi. Au-delà de sept jours, les distances valent 8 et le type et le jour de semaine sont neutralisés. Le calendrier de jours fériés du projet est limité aux dates 2025–2026 répertoriées dans `src/features.py`.

L'ablation des ponts neutralise `bridge_intensity`, l'indicateur et le régime de pont qui en découlent, ainsi que leur correction dans `smart_expected`. La correction propre au 11 novembre reste active si ses données historiques sont disponibles. La variante ne supprime pas les informations calendaires connues comme les jours de semaine et les fériés.

## Huit fenêtres récentes

Fenêtres du 9 septembre au 1er novembre 2026, 934 heures communes par rue et cible. Les MAE du débit sont en véhicules/h et celles de l'occupation en points.

| Rue et cible | Référence | Sans pont | Calendrier | Calendrier sans pont |
|---|---:|---:|---:|---:|
| Convention, q | 48,42 | 47,86 | 48,52 | 48,58 |
| Convention, k | 1,134 | 1,135 | 1,132 | 1,133 |
| Saint-Pères, q | 67,65 | 66,72 | 67,49 | 67,17 |
| Saint-Pères, k | 1,637 | 1,626 | 1,614 | 1,619 |

Les écarts sont faibles. Pour le débit, la variante sans pont gagne sur six des huit fenêtres à Convention et à Saint-Pères, mais la moyenne ne résume pas sa performance autour des événements.

## Jours fériés et alentours

Les tableaux ci-dessous sont pondérés par heure observée. Chaque score compare exactement les mêmes heures. Les douze jours fériés de novembre 2025 à novembre 2026 fournissent 275 heures à Convention et 288 à Saint-Pères.

| Jour férié, débit | Référence | Sans pont | Calendrier | Calendrier sans pont |
|---|---:|---:|---:|---:|
| Convention, q | 74,52 | 74,62 | 74,59 | 77,43 |
| Saint-Pères, q | 84,47 | 85,46 | 84,46 | 86,15 |

Les veilles, la MAE de débit à Convention passe de 47,51 à 50,68 véhicules/h sans pont (285 heures). Les lendemains, elle passe de 40,63 à 52,25 véhicules/h (259 heures). La hausse des lendemains est concentrée notamment sur les 26 décembre 2025, 2 janvier et 15 mai 2026 ; ces dates correspondent à des vendredis après un jeudi férié dans le calendrier de l'expérience. À Saint-Pères, la MAE du débit le lendemain passe de 49,48 à 54,78 véhicules/h (261 heures). Les lendemains du 1er novembre 2026 ne sont pas couverts par les données disponibles.

Aux Champs-Élysées, seuls deux jours fériés de novembre 2025 sont évaluables. Le 11 novembre 2025, la MAE de débit est de 274,16 véhicules/h pour la référence et de 260,38 avec le calendrier enrichi, sur 24 heures ; ces valeurs restent élevées et n'établissent pas la qualité de prévision pour 2026.

## Limites et suite

Les fenêtres fériées peuvent se chevaucher ; certaines observations et certains événements sont rares. Les huit fenêtres récentes ont déjà servi à d'autres analyses et l'une recoupe le 1er novembre. Ces chiffres sont descriptifs, sans test final indépendant ; les petites différences ne permettent pas de conclure à une amélioration robuste. La météo n'est pas utilisée.

Conserver la référence pour le moment. Tester les variables calendaires sur d'autres années et analyser séparément les vendredis de pont avant de retirer les règles. Une comparaison plus ciblée pourrait distinguer l'effet des coefficients numériques de celui du regroupement en régime « pont ». Il reste prioritaire de trouver des mesures récentes pour les Champs-Élysées.

Les scores par événement, période, rue et cible, les prévisions et les effectifs sont dans `data/processed/holiday_analysis/`. Les variantes sont expérimentales et aucun modèle final n'est remplacé.
