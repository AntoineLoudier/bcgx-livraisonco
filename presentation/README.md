# LivraisonCo — Revue des résultats et décision de pilote

La présentation a été réécrite en huit slides principales, avec neuf annexes techniques. La version précédente est conservée dans `archive_v1.zip`.

- **LivraisonCo_synthese.pdf** : les huit slides pour l’oral.
- **LivraisonCo_business.pdf** : le même récit, suivi des annexes.
- **LivraisonCo_business.pptx** : version modifiable, avec notes orateur.
- **notes_orateur.md** : notes et explications des calculs, consultables séparément.

Les graphiques et maquettes des huit slides principales sont composés d’objets PowerPoint éditables. Les neuf annexes reprennent des images du PDF d’origine ; leurs chiffres et figures internes ne sont pas éditables. Le PDF original `LivraisonCo_CTM.pdf` est exclu du dépôt ; les extraits nécessaires sont conservés dans `assets/`.

## Choix de fond et de forme

Le support utilise des titres qui énoncent une conclusion, des comparaisons sur une même échelle, une analyse de sensibilité et des tableaux de décision. La présentation évite les slogans, les grandes cartes répétées et les chiffres de gains détachés de leurs hypothèses.

Le fil de présentation est le suivant :

1. Tester la réduction des heures de livraison, en commençant sur Convention.
2. Expliquer l’économie avec les heures réellement récupérables et un seuil de rentabilité.
3. Montrer les performances par rue et la contribution de l’ancrage puis de la correction.
4. Expliquer le modèle en une slide.
5. Identifier le travail manquant entre trafic prévu et coût de tournée.
6. Présenter le rôle de l’exploitation et du livreur dans un outil concret.
7. Proposer un protocole prospectif comparant au planning actuel.
8. Demander les données, le référent métier et l’accord sur le pilote.

La partie modèle est conservée en détail dans les annexes. Les prévisions du 7–11 novembre sont également conservées, avec une mention explicite du statut exploratoire des créneaux et restrictions.

## Provenance et portée des résultats

Les scores sont transcrits de **la page 13 du PDF fourni** ; ils sont centralisés dans `scores_source.json`. Aucun entraînement ni nouveau backtest n’a été exécuté pour cette adaptation.

Les gains sur q sont de 21,0 % sur Convention et 3,3 % sur Saint-Pères. Sur k, ils sont de 24,9 % et 14,8 %. Ce sont des réductions de RMSE face à J−7, pas des taux d’accuracy ni des économies. Les pourcentages publiés sont repris tels quels : les valeurs arrondies des RMSE ne permettent pas toujours de les reconstruire exactement.

Les barres de débit partagent une échelle commençant à zéro. Elles montrent que, sur Saint-Pères, l’ancrage seul est moins bon que J−7 ; la correction rattrape d’abord cette faiblesse. Le score global est conservé en annexe, car il mélange notamment les mesures 2025 des Champs-Élysées et les périodes 2026 des autres rues.

La validation Champs-Élysées ne démontre pas la performance de novembre 2026. Le dépôt local consulté était sur `main` et ne contenait pas les artefacts associés au support : cette adaptation ne constitue pas un nouvel audit de causalité. La disponibilité historique des prévisions météo et l’évaluation sur des cibles réelles sont à confirmer dans ces artefacts.

Les horaires de fermeture évoqués dans le support initial n’ont pas été vérifiés pour 2026. Dans l’annexe de prévisions, ils doivent être lus comme les hypothèses de ce support, à confirmer avant toute décision opérationnelle.

## Économie : hypothèses et calculs

Les scénarios sont **des simulations, pas des gains constatés ni des prévisions de rentabilité**. Les taux de 10 €/h le jour et 14 €/h la nuit viennent du cas. La base de 100 heures payées par jour, les 250 jours par an, la part de conduite et les gains de durée/récupération sont illustratifs.

```text
Base annuelle = 100 h/jour × 10 €/h × 250 jours = 250 000 €
Part conduite = 40 % = 40 h/jour
Temps libéré dans le scénario repère = 40 × 20 % = 8 h/jour
Heures effectivement évitées = 8 × 70 % = 5,6 h/jour
Économie brute = 5,6 × 10 × 250 = 14 000 €/an
```

La matrice fait varier séparément la réduction du temps de conduite (10/20/30 %) et la part récupérable (30/50/70 %), à part conduite constante. Aucune probabilité n’est associée à ces scénarios. Le coût horaire utilisé n’est pas nécessairement le coût employeur complet.

La version précédente inventait 6 k€ initiaux et 4 k€/an récurrents pour illustrer le gain net. **Ces montants sont retirés de la version actuelle.** Le coût de la solution reste à chiffrer. Les 14 k€ constituent uniquement le plafond de coût en année 1 à l’équilibre dans le scénario repère, en supposant une année complète de gain ; ni budget recommandé, ni devis. La montée en charge et les coûts complets modifieraient ce seuil.

À effectif et heures payées constants, le temps gagné représente de la capacité et ne réduit pas automatiquement les dépenses. Aucun lien empirique entre baisse de RMSE et baisse du temps de conduite n’est estimé.

Le graphique jour/nuit utilise une tournée de nuit de 5 h à 14 €/h et une durée de jour variable à 10 €/h. Les coûts sont égaux pour 7 h le jour ; la nuit est moins chère au-delà, à travail et autres coûts identiques. Ce graphique est calculé, il ne provient pas de trajets GPS.

## Régénération et vérification

```bash
uv pip install --python .venv/bin/python python-pptx reportlab pymupdf
.venv/bin/python presentation/build_presentation.py
```

Le générateur écrit les PDF, le PowerPoint, les notes, `hypotheses_business.json`, `scenarios_business.csv`, `scores_source.json` et `layout_bounds.json`. Les fichiers JSON/CSV sont des sorties, pas les entrées du script. Pour faire évoluer les paramètres, modifier le bloc `assumptions` et les légendes narratives correspondantes dans le script. Les calculs et les cellules se recalculent ; certaines formulations explicatives restent écrites pour le scénario actuel.

La régénération utilise les figures de `assets/` et fonctionne sans le PDF original. Si celui-ci est fourni temporairement à la racine, le script peut en réextraire les figures. Les chemins de la version archivée ne sont pas maintenus : utiliser le générateur courant.

Le PDF utilise Arimo, compatible en métriques avec Arial ; le PowerPoint demande Arial. Les fontes ne sont pas intégrées au PowerPoint. Le PDF a été rendu et inspecté visuellement, les coordonnées et calculs vérifiés. Le rendu dans Microsoft PowerPoint n’a pas été contrôlé ici.
