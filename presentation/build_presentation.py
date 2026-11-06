"""Build the review deck in PDF and editable PowerPoint from explicit source data.

Run: .venv/bin/python presentation/build_presentation.py
The original PDF is the source of model scores; no model is rerun here.
Its extracted figures in assets/ allow regeneration without the original PDF.
"""
from pathlib import Path
import csv
import json
import pymupdf
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'LivraisonCo_CTM.pdf'
ASSETS = ROOT / 'assets'
ASSETS.mkdir(exist_ok=True)
W, H = 1280, 720
INK, GREEN, GRAY = '#1D2B32', '#007A53', '#5A686E'
LIGHT, RULE, PALE = '#EFF3F1', '#D2DAD7', '#E0EEE7'
MID, AMBER, WHITE = '#81938D', '#94631F', '#FFFFFF'
FONTS = Path('/usr/share/fonts/truetype/croscore')
for name, filename in [('Body', 'Arimo-Regular.ttf'), ('Bold', 'Arimo-Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / filename)))

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W/96), Inches(H/96)
prs.core_properties.title = 'LivraisonCo — Résultats et décision de pilote'
prs.core_properties.author = 'Les Architectes'
pdf = canvas.Canvas(str(ROOT / 'LivraisonCo_business.pdf'), pagesize=(W, H))
pdf.setTitle(prs.core_properties.title)
pdf.setAuthor('Les Architectes')
slide = None
page = 0
notes = []
text_bounds = []


def rgb(color):
    return RGBColor.from_string(color.removeprefix('#'))


def rect(x, y, w, h, color, border=None, radius=0):
    assert w > 0 and h > 0
    pdf.setFillColor(color)
    pdf.setStrokeColor(border or color)
    if radius:
        pdf.roundRect(x, H-y-h, w, h, radius, fill=1, stroke=int(bool(border)))
    else:
        pdf.rect(x, H-y-h, w, h, fill=1, stroke=int(bool(border)))
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                              Inches(x/96), Inches(y/96), Inches(w/96), Inches(h/96))
    if radius:
        s.adjustments[0] = min(radius/min(w, h), .5)
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(color)
    if border:
        s.line.color.rgb = rgb(border)
        s.line.width = Pt(.6)
    else:
        s.line.fill.background()


def line(x1, y1, x2, y2, color=RULE, width=1, dashed=False):
    pdf.setStrokeColor(color)
    pdf.setLineWidth(width)
    pdf.setDash([5, 4] if dashed else [])
    pdf.line(x1, H-y1, x2, H-y2)
    pdf.setDash([])
    s = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                  Inches(x1/96), Inches(y1/96), Inches(x2/96), Inches(y2/96))
    s.line.color.rgb = rgb(color)
    s.line.width = Pt(width*.75)
    if dashed:
        from pptx.enum.dml import MSO_LINE_DASH_STYLE
        s.line.dash_style = MSO_LINE_DASH_STYLE.DASH


def dot(x, y, r, color=GREEN):
    pdf.setFillColor(color)
    pdf.circle(x, H-y, r, fill=1, stroke=0)
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches((x-r)/96), Inches((y-r)/96),
                              Inches(2*r/96), Inches(2*r/96))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(color)
    s.line.fill.background()


def wrap(value, w, size, bold):
    result = []
    for para in str(value).split('\n'):
        current = ''
        for word in para.split():
            candidate = (current + ' ' + word).strip()
            if current and pdfmetrics.stringWidth(candidate, 'Bold' if bold else 'Body', size) > w:
                result.append(current)
                current = word
            else:
                current = candidate
        result.append(current)
    return result


def txt(x, y, value, size=21, color=INK, bold=False, w=1152, leading=None):
    leading = leading or size*1.22
    lines = wrap(value, w, size, bold)
    height = len(lines)*leading + 4
    assert y+height < H+1, (page, value, y+height)
    assert 0 <= x and x+w <= W+1, (page, value, x+w)
    pdf.setFillColor(color)
    pdf.setFont('Bold' if bold else 'Body', size)
    for i, value_line in enumerate(lines):
        pdf.drawString(x, H-y-size-i*leading, value_line)
    s = slide.shapes.add_textbox(Inches(x/96), Inches(y/96), Inches(w/96), Inches(height/96))
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = False
    for i, value_line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = value_line
        p.font.name = 'Arial'
        p.font.size = Pt(size*.75)
        p.font.bold = bold
        p.font.color.rgb = rgb(color)
        p.line_spacing = Pt(leading*.75)
        p.space_before = p.space_after = Pt(0)
    # Retain content rectangles for layout inspection independently of PowerPoint.
    text_bounds.append({'page': page, 'text': str(value), 'x': x, 'y': y, 'w': w, 'h': height})
    return height


def image(path, x, y, w, h):
    pdf.drawImage(str(path), x, H-y-h, w, h)
    slide.shapes.add_picture(str(path), Inches(x/96), Inches(y/96), Inches(w/96), Inches(h/96))


def new(section, title, subtitle='', source=''):
    global slide, page
    if page:
        pdf.showPage()
    page += 1
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    rect(0, 0, W, H, WHITE)
    txt(56, 25, 'LIVRAISONCO', 13, GREEN, True, w=200)
    txt(855, 25, section, 13, GRAY, w=370)
    txt(56, 74, title, 35, INK, True, w=1168, leading=40)
    line(56, 173, 1224, 173, GREEN, 2)
    if subtitle:
        txt(56, 190, subtitle, 18, GRAY, w=1168)
    if source:
        txt(56, 656, source, 12.5, GRAY, w=1168, leading=15)
    line(56, 688, 1224, 688)
    txt(56, 697, 'Les Architectes  |  Datathon BCG X × CentraleSupélec  |  6 novembre 2026', 11, GRAY, w=1000)
    txt(1185, 694, str(page), 14, GRAY, w=40)


def note(title, body):
    slide.notes_slide.notes_text_frame.text = body
    notes.append(f'## {page}. {title}\n\n{body}\n')


def conclusion(value, y=600):
    line(56, y, 1224, y)
    rect(56, y+18, 4, 33, GREEN)
    txt(72, y+16, value, 22, INK, True, w=1148)


def dec(n, places=1):
    return f'{n:.{places}f}'.replace('.', ',')


def pct(n):
    return dec(100*n)+' %'


# A sensitivity analysis replaces the unsupported implementation-cost estimate.
assumptions = {
    'statut': 'Simulation illustrative ; aucun gain de temps ou de coût mesuré',
    'heures_payees_par_jour': 100,
    'cout_horaire_du_cas_eur': 10,
    'jours_par_an': 250,
    'part_conduite': .40,
    'reductions_conduite': [.10, .20, .30],
    'parts_recuperables': [.30, .50, .70],
    'repere': {'reduction_conduite': .20, 'part_recuperable': .70},
    'cout_solution': 'À chiffrer ; aucun devis disponible',
}
annual_base = assumptions['heures_payees_par_jour']*assumptions['cout_horaire_du_cas_eur']*assumptions['jours_par_an']
rows = []
for reduction in assumptions['reductions_conduite']:
    for recoverable in assumptions['parts_recuperables']:
        rate = assumptions['part_conduite']*reduction*recoverable
        rows.append({'reduction_conduite': reduction, 'part_recuperable': recoverable,
                     'baisse_couts_horaires': round(rate, 4),
                     'gain_brut_annuel_eur': round(annual_base*rate)})
repere = assumptions['repere']
saving_rate = assumptions['part_conduite']*repere['reduction_conduite']*repere['part_recuperable']
annual_saving = round(saving_rate*annual_base)
drive_hours = assumptions['heures_payees_par_jour']*assumptions['part_conduite']
released_hours = drive_hours*repere['reduction_conduite']
avoided_hours = released_hours*repere['part_recuperable']
(ROOT/'hypotheses_business.json').write_text(json.dumps(assumptions, ensure_ascii=False, indent=2)+'\n')
with (ROOT/'scenarios_business.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

scores = [
    {'rue': 'Convention', 'q_j7': 79.1, 'q_ancrage': 74.3, 'q_final': 62.5, 'gain_q': 21.0,
     'k_j7': 2.81, 'k_final': 2.11, 'gain_k': 24.9},
    {'rue': 'Saint-Pères', 'q_j7': 83.2, 'q_ancrage': 91.4, 'q_final': 80.4, 'gain_q': 3.3,
     'k_j7': 2.60, 'k_final': 2.22, 'gain_k': 14.8},
]
(ROOT/'scores_source.json').write_text(json.dumps({
    'source': 'LivraisonCo_CTM.pdf (support initial non distribué), page 13',
    'statut': 'Transcription des résultats du support ; pas de nouveau calcul de backtest',
    'unites': {'q': 'RMSE en véhicules/heure', 'k': 'RMSE en points de pourcentage'},
    'scores': scores,
}, ensure_ascii=False, indent=2)+'\n')

# 1 — Recommendation memo, evidence and economic question.
new('Synthèse et recommandation',
    'Réduire les heures de livraison en adaptant les créneaux ;\ncommencer le pilote sur Convention',
    'Le modèle améliore la prévision du trafic. La prochaine étape est de mesurer son effet sur les tournées.',
    'Sources : support du 5 novembre, p. 7 et 13. Simulation économique ; gains de temps et économies non observés.')
line(841, 247, 841, 577)
for y, n, title, body in [
    (247, '01', 'Un premier cas d’usage sur Convention',
     '−21 % de RMSE sur le débit et −25 % sur l’occupation face à J−7. Saint-Pères permet de tester un cas où le gain est plus faible.'),
    (362, '02', 'Une décision précise à tester',
     'Comparer les créneaux d’une même tournée, puis choisir le moins coûteux parmi ceux compatibles avec les contraintes clients.'),
    (477, '03', 'Un périmètre à exclure du pilote autonome',
     'Champs-Élysées : aucune mesure récente. Une validation de 2025 ne suffit pas pour automatiser une décision en novembre 2026.'),
]:
    txt(56, y, n, 24, GREEN, True, w=45)
    txt(111, y, title, 24, INK, True, w=690)
    txt(111, y+38, body, 21, GRAY, w=680)
txt(877, 250, 'ORDRE DE GRANDEUR', 14, GRAY, True, w=335)
txt(877, 291, '1 h payée évitée\npar jour', 31, GREEN, True, w=335)
txt(877, 386, '= 2 500 €/an', 32, INK, True, w=335)
txt(877, 441, 'À 10 €/h et 250 jours/an.\nHors coût de la solution.', 21, GRAY, w=335)
txt(877, 521, 'À vérifier : le temps libéré peut-il réduire les heures payées ?', 22, INK, True, w=335)
conclusion('Décision proposée : autoriser un pilote de planification et l’accès aux données de tournées.')
note('La recommandation',
     'Ouvrir sur le coût horaire de la livraison et la décision à prendre, sans annoncer d’économies déjà réalisées. '
     'Convention est le premier candidat sur la base des gains reportés, pas une garantie de rentabilité ou de significativité statistique. '
     'Les gains arrondis 21 % et 25 % correspondent à 21,0 % et 24,9 % dans le support source. Saint-Pères peut être un second site, '
     'avec son propre groupe témoin : ce n’est pas un contrôle interchangeable de Convention. '
     '1 h/jour × 10 €/h × 250 jours = 2 500 €/an, avant coût de solution. Le taux 10 €/h vient du cas ; les 250 jours sont une hypothèse. '
     'À heures payées constantes, le gain est de la capacité ; ne pas l’appeler économie de trésorerie. '
     'Équipe : Antoine Loudier, Amin Belfkira, Baptiste Dejardin, Joseph Bouteiller, Louis Jamin.')

# 2 — Unit economics: derivation + sensitivity, no invented investment quote.
new('Économie | Simulation',
    f'{dec(avoided_hours)} h payées évitées par jour représenteraient {annual_saving//1000} k€/an ;\nla récupération du temps est l’hypothèse décisive',
    'Base illustrative : 100 h payées/jour, dont 40 % de conduite ; 10 €/h et 250 jours/an.',
    'Calcul : part de conduite × réduction de conduite × part récupérable. Coûts horaires seuls ; coût de solution à chiffrer.')
txt(56, 243, 'Du temps de conduite aux heures évitées', 23, INK, True, w=555)
# Floating bars: starting drive time, theoretical release, recoverable part.
chart_x, chart_y, scale, chart_w = 70, 475, 4.25, 510
for tick in [0, 20, 40]:
    yy = chart_y-tick*scale
    line(chart_x, yy, chart_x+chart_w, yy, RULE)
    txt(chart_x, yy-19, str(tick)+' h', 13, GRAY, w=60)
for x, amount, color, lab in [
    (136, drive_hours, MID, 'Conduite\nactuelle'),
    (304, released_hours, '#A8BCB3', 'Temps libéré\nsi −20 %'),
    (472, avoided_hours, GREEN, 'Heures évitées\nsi 70 % récupérés'),
]:
    rect(x, chart_y-amount*scale, 68, amount*scale, color)
    txt(x-2, chart_y-amount*scale-32, dec(amount, 0 if amount.is_integer() else 1)+' h', 25, INK, True, w=130)
    txt(x-40, 492, lab, 18, GRAY, w=154)
txt(56, 556, f'Scénario repère : {pct(saving_rate)} des coûts horaires évitables.', 21, GREEN, True, w=570)
line(646, 244, 646, 580)
txt(681, 243, 'Sensibilité de la baisse des coûts horaires', 23, INK, True, w=540)
txt(826, 286, 'Part du temps libéré réellement récupérée', 15, GRAY, w=390)
for j, recoverable in enumerate(assumptions['parts_recuperables']):
    txt(858+j*123, 316, f'{recoverable:.0%}', 20, GRAY, True, w=110)
txt(681, 346, 'Baisse du temps\nde conduite', 15, GRAY, w=152)
for i, reduction in enumerate(assumptions['reductions_conduite']):
    y = 388+i*55
    txt(697, y+13, f'−{reduction:.0%}', 23, INK, True, w=110)
    for j, recoverable in enumerate(assumptions['parts_recuperables']):
        x = 827+j*123
        selected = reduction == repere['reduction_conduite'] and recoverable == repere['part_recuperable']
        rect(x, y, 116, 48, GREEN if selected else LIGHT)
        txt(x+24, y+10, pct(assumptions['part_conduite']*reduction*recoverable), 23,
            WHITE if selected else INK, selected, w=90)
txt(681, 556, 'Valeurs simulées ; aucun scénario n’est une prévision.', 17, GRAY, w=535)
conclusion(f'À ce niveau de gain, un coût total de solution supérieur à {annual_saving//1000} k€ en année 1 annulerait les économies.')
note('L’économie dépend de la récupération des heures',
     'La sensibilité fait varier deux hypothèses indépendamment. Aucune probabilité n’est associée aux cellules. '
     'La part conduite est fixée à 40 % uniquement pour la simulation ; elle doit être mesurée. '
     'Le scénario repère est 100 × 40 % × 20 % × 70 % = 5,6 h/jour évitées, soit 5,6 % des heures payées et 14 000 €/an au taux du cas. '
     'Le coût horaire de 10 € n’est pas nécessairement le coût employeur complet. À coût et effectif constants, parler de capacité, pas de cash. '
     'Le plafond de 14 k€ n’est ni un budget recommandé ni un devis : c’est le seuil de rentabilité en année 1 de ce scénario, '
     'en supposant le gain sur 250 jours, sans montée en charge, actualisation ni coûts oubliés. '
     'Les 6 k€ initiaux et 4 k€/an de la version précédente sont retirés : ils n’étaient étayés par aucun chiffrage technique. '
     'Aucun lien empirique entre gain de RMSE et réduction du temps de conduite n’est estimé.')

# 3 — Show heterogeneity and baseline difficulty instead of pooling mixed periods.
new('Performance | Résultats reportés',
    'Le gain sur le débit atteint 21 % sur Convention,\nmais seulement 3 % sur Saint-Pères',
    'RMSE sur quatre fenêtres de cinq jours ; comparaison au trafic de la même heure à J−7.',
    'Source : support initial, p. 13. Pourcentages reportés avant arrondi. Validation sur cibles observées selon le support ; incertitude non fournie.')
txt(56, 243, 'Débit q — RMSE (véhicules/heure)', 23, INK, True, w=725)
# Grouped horizontal bars on a shared zero-based scale.
for x, col, name in [(56, '#BDC7C2', 'J−7'), (182, MID, 'Ancrage'), (368, GREEN, 'Modèle final')]:
    rect(x, 284, 16, 12, col)
    txt(x+25, 279, name, 18, GRAY, w=160)
bar_x, scale = 259, 4.35
for i, data in enumerate(scores):
    y = 333+i*143
    txt(56, y+22, data['rue'], 23, INK, True, w=195)
    for j, (key, color) in enumerate([('q_j7', '#BDC7C2'), ('q_ancrage', MID), ('q_final', GREEN)]):
        yy = y+j*28
        rect(bar_x, yy, data[key]*scale, 18, color)
        txt(bar_x+data[key]*scale+9, yy-3, dec(data[key]), 18, color if j==2 else GRAY,
            j==2, w=80)
    txt(56, y+57, f'−{dec(data["gain_q"])} % vs J−7', 18, GREEN, True, w=197)
for tick in [0, 20, 40, 60, 80, 100]:
    xx = bar_x+tick*scale
    line(xx, 584, xx, 589, GRAY)
    txt(xx-8, 591, str(tick), 13, GRAY, w=45)
line(bar_x, 584, bar_x+100*scale, 584, GRAY)
line(790, 244, 790, 622)
txt(822, 243, 'Occupation k', 23, INK, True, w=400)
txt(822, 276, 'RMSE en points de pourcentage', 17, GRAY, w=400)
for i, data in enumerate(scores):
    y = 319+i*86
    txt(822, y, data['rue'], 21, INK, True, w=390)
    txt(822, y+31, f'{dec(data["k_j7"], 2)} → {dec(data["k_final"], 2)}', 23, INK, w=200)
    txt(1059, y+31, f'−{dec(data["gain_k"])} %', 23, GREEN, True, w=164)
line(822, 497, 1224, 497)
txt(822, 515, 'Champs-Élysées : à part', 21, AMBER, True, w=400)
txt(822, 549, 'Validation en 2025 ; capteur muet depuis le 19/11/2025. Performance 2026 non démontrée.', 20, GRAY, w=395)
note('Le score moyen masque les différences',
     'Tous les chiffres de cette slide sont transcrits de la page 13 du PDF original. Les barres partent de zéro et partagent la même échelle. '
     'Convention : q 79,1 (J−7), 74,3 (ancrage), 62,5 (final). Saint-Pères : 83,2, 91,4, 80,4. '
     'Sur Saint-Pères, l’ancrage est moins bon que J−7 : la correction rattrape surtout cette faiblesse avant de produire un gain limité face à J−7. '
     'C’est la raison de garder une référence simple dans l’expérimentation opérationnelle. '
     'k : 2,81 à 2,11 sur Convention ; 2,60 à 2,22 sur Saint-Pères. Les variations publiées reposent sur des valeurs avant arrondi. '
     'La RMSE mesure une erreur ; ces pourcentages ne sont pas des taux d’accuracy. '
     'Le gain global −9,3 % q / −12,4 % k est conservé en annexe et non utilisé comme preuve de performance contemporaine, '
     'car il agrège notamment la période 2025 des Champs-Élysées. Pas d’intervalle de confiance ni de nouveau backtest exécuté ici.')

# 4 — Technical mechanism, causal timeline, and limits in one page.
new('Modèle | Principe',
    'Le profil hebdomadaire fournit le niveau attendu ;\nles arbres apprennent la correction à apporter',
    'Du 7 au 11 novembre 2026 : 3 rues × 120 heures = 360 lignes, avec débit q et occupation k.',
    'Source : support initial, p. 4 et 14. Architecture reportée ; disponibilité historique de la météo à vérifier dans les backtests.')
rect(56, 249, 1168, 85, LIGHT)
txt(77, 272, 'Prévision', 29, INK, True, w=185)
txt(262, 271, '=', 30, GRAY, w=40)
txt(320, 272, 'Ancrage calendaire', 29, INK, True, w=355)
txt(693, 271, '+', 30, GRAY, w=40)
txt(756, 272, 'Correction LightGBM / XGBoost', 29, GREEN, True, w=450)
line(705, 360, 705, 517)
txt(56, 360, 'Une référence explicite', 23, INK, True, w=620)
txt(56, 400, '60 % × J−7 + 25 % × J−14 + 15 % × J−21', 24, INK, w=625)
txt(56, 443, 'Puis ajustement calendaire, notamment pour le pont et le 11 novembre, à partir de l’historique 2025.', 21, GRAY, w=605)
txt(751, 360, 'Une correction conditionnelle', 23, INK, True, w=470)
txt(751, 400, 'Calendrier, météo, historique et état du trafic : 48 variables dans le support initial.', 21, GRAY, w=470)
txt(751, 465, 'Moyenne des deux corrections apprises ; pondération plus forte des heures récentes.', 21, GRAY, w=470)
line(56, 547, 1224, 547)
txt(56, 565, 'Règle de validation', 18, GREEN, True, w=205)
txt(281, 561, 'À chaque date de prévision, n’utiliser que l’information disponible à cette date.\nÉvaluer sur les mesures réelles, y compris les jours où les données sont moins fiables.', 21, INK, w=942)
note('Expliquer le modèle en une minute',
     'Décrire la référence, puis la correction. L’ajustement de calendrier et les coefficients de pont hérités de 2025 sont des hypothèses à tester, '
     'pas des lois de trafic. La moyenne LightGBM/XGBoost concerne les corrections de l’ancrage. '
     'Il y a six couples rue/cible, chacun associant les deux familles : éviter la formulation ambiguë « six arbres ». '
     'Le dépôt local consulté est main et ne contient pas les artefacts qui ont produit les scores du support. Cette slide décrit le support fourni, '
     'elle n’est pas un nouvel audit du pipeline. La validation annoncée doit être confirmée par les artefacts, notamment les vintages météo '
     'et le traitement des imputations. Les cibles non observées ne sont pas évaluables. '
     'Ne pas confondre disponibilité du calendrier, causalité des lags et disponibilité d’une prévision météo historique.')

# 5 — The missing operational link; economic threshold shown as a real curve.
new('Passage à la décision',
    'Le trafic prévu doit être converti en durée de tournée\navant de recommander un changement de créneau',
    'Débit et occupation décrivent le trafic ; ils ne donnent pas directement le temps ou le coût d’une livraison.',
    'Tarifs du cas : 10 €/h le jour, 14 €/h la nuit (support initial, p. 7). Calcul à travail et autres coûts identiques ; nuit fixée à 5 h.')
txt(56, 243, 'Ce qui manque entre la prévision et le planning', 23, INK, True, w=660)
for y, number, title, body in [
    (292, '1', 'Trafic par rue et par heure', 'Disponible : q et k, sur trois rues et cinq jours.'),
    (386, '2', 'Durées de conduite et d’arrêt', 'À calibrer : GPS, arrêts, charge et créneau de passage.'),
    (480, '3', 'Tournées et coûts sous contraintes', 'À intégrer : clients, capacités, horaires et coûts payés.'),
]:
    dot(72, y+16, 15, GREEN if number=='1' else MID)
    txt(67, y+3, number, 19, WHITE, True, w=24)
    txt(105, y, title, 23, INK, True, w=550)
    txt(105, y+36, body, 20, GRAY, w=540)
line(706, 244, 706, 574)
txt(741, 243, 'Exemple : jour ou nuit ?', 23, INK, True, w=480)
txt(741, 279, 'Coût de la tournée (€) · heures seules', 16, GRAY, w=480)
# x: day duration 4..9 h; y: total cost 40..90 euros.
ox, oy, cw, ch = 783, 521, 385, 189
def cost_point(hours, euros):
    return ox+(hours-4)/5*cw, oy-(euros-40)/50*ch
for euros in [40, 60, 80]:
    _, yy = cost_point(4, euros)
    line(ox, yy, ox+cw, yy)
    txt(746, yy-10, str(euros), 14, GRAY, w=35)
for hours in range(4, 10):
    xx, _ = cost_point(hours, 40)
    txt(xx-5, 534, str(hours), 15, GRAY, w=30)
line(ox, oy, ox+cw, oy, GRAY)
x1, y1 = cost_point(4, 40)
x2, y2 = cost_point(9, 90)
line(x1, y1, x2, y2, GREEN, 3)
x1, y1 = cost_point(4, 70)
x2, y2 = cost_point(9, 70)
line(x1, y1, x2, y2, MID, 2, True)
xx, yy = cost_point(7, 70)
dot(xx, yy, 5, INK)
line(xx, yy, xx, oy, GRAY, 1, True)
txt(806, 345, 'Jour : 10 € × durée', 17, GREEN, True, w=275)
txt(806, 417, 'Nuit : 14 € × 5 h = 70 €', 17, GRAY, w=300)
txt(783, 561, 'Durée de la tournée de jour (h)', 16, GRAY, w=430)
conclusion('La nuit n’est moins chère que si la tournée de jour dépasse 7 h, contre 5 h la nuit.')
note('Mesurer le maillon manquant',
     'Cette slide explique pourquoi le gain de prédiction n’est pas directement un gain financier. '
     'La durée dépend aussi des arrêts, du nombre de colis, des clients et de l’itinéraire. Un faible q ne prouve pas la fluidité : '
     'le taux d’occupation k est utile pour distinguer route vide et congestion. '
     'L’exemple est un calcul : jour 10 €/h × durée variable ; nuit 14 €/h × 5 h supposées. Égalité à 7 h, soit un surcroît de durée de 40 %. '
     'La nuit devient moins chère au-delà, à travail identique et en supposant aucune différence d’autre coût. '
     'Le coût employeur complet, les contraintes horaires et les restrictions locales doivent être intégrés avant une recommandation. '
     'Le graphe est calculé à partir des hypothèses affichées, pas à partir d’un historique GPS. Aucun moteur de tournée opérationnel n’est revendiqué.')

# 6 — Actual workflow and mockup with decision states, no generic app cards.
new('Produit | Maquette de travail',
    'L’exploitation choisit le créneau ; le livreur reçoit\nune tournée validée et remonte les temps réalisés',
    'Un site de planification avec une vue mobile ; la direction suit le coût par colis et la ponctualité.',
    'Maquettes conceptuelles. Tournée et horaires illustratifs ; aucune intégration métier ni application déployée à ce stade.')
# Desktop pane: more workbench than marketing dashboard.
rect(56, 248, 807, 321, WHITE, RULE)
rect(56, 248, 807, 42, LIGHT)
txt(74, 258, 'Planning  /  Lundi 9 novembre', 19, INK, True, w=615)
txt(76, 310, 'Tournée A · Convention', 23, INK, True, w=620)
txt(76, 346, 'Créneau', 16, GRAY, True, w=130)
txt(245, 346, 'Lecture trafic', 16, GRAY, True, w=215)
txt(495, 346, 'Contraintes clients', 16, GRAY, True, w=260)
for y, slot, traffic, client, select in [
    (384, '08 h – 12 h', 'Option issue du modèle', 'À confirmer', True),
    (429, '12 h – 16 h', 'Option à comparer', 'À confirmer', False),
]:
    if select:
        rect(72, y-7, 773, 43, PALE)
    txt(77, y, slot, 18, INK, select, w=165)
    txt(245, y, traffic, 18, GRAY, w=245)
    txt(495, y, client, 18, AMBER, w=235)
line(76, 480, 840, 480)
txt(76, 495, 'Champs-Élysées : donnée ancienne → revue manuelle', 18, AMBER, w=750)
rect(588, 528, 252, 28, LIGHT)
txt(604, 532, 'Validation après contrôle', 16, GRAY, w=225)
# Mobile with verified workflow label; statuses are fictitious explicitly.
rect(917, 243, 287, 336, INK, radius=15)
rect(929, 255, 263, 310, WHITE, radius=6)
txt(949, 278, 'MA TOURNÉE', 13, GREEN, True, w=220)
txt(949, 310, 'Tournée A', 26, INK, True, w=220)
txt(949, 352, 'Convention\n08 h – 12 h', 23, INK, w=220)
txt(949, 417, 'Après validation du planning', 15, GRAY, w=220)
line(949, 452, 1171, 452)
for y, word in [(467, 'Enregistrer l’arrivée'), (511, 'Signaler un incident')]:
    txt(949, y, word, 19, GREEN, True, w=220)
conclusion('La priorité du premier outil : comparer, valider et enregistrer les durées réellement observées.')
note('Un outil qui produit aussi les données manquantes',
     'Parcours proposé : l’exploitation prépare le planning, confronte les options aux engagements clients et valide. '
     'Le livreur reçoit ensuite la tournée validée et remonte les heures d’arrivée, départ et incident ; les saisies doivent demander peu d’effort. '
     'Ces événements complètent le GPS pour distinguer circulation et arrêts. La direction consulte le coût par colis livré et la ponctualité. '
     'Le tableau est une maquette : Tournée A et les statuts sont fictifs, les créneaux ne sont pas optimisés. '
     'La plage 8–12 h est inspirée du support initial, pas une décision validée. Les contraintes doivent être renseignées avant publication. '
     'La vue mobile est la suite du flux après validation et non un moyen de contourner le contrôle exploitation. '
     'Les formes et textes sont éditables dans PowerPoint. Cette tâche livre un support de présentation, pas une application fonctionnelle.')

# 7 — Prospective experimental design with observable pass criteria.
new('Pilote | Protocole proposé',
    'Comparer au planning actuel sur des tournées équivalentes,\npuis décider sur le coût par colis et la ponctualité',
    'Deux niveaux de validation : la qualité de prévision, puis l’effet de la décision sur l’exploitation.',
    'Protocole proposé. Durée et taille d’échantillon à dimensionner à partir de la variabilité des coûts ; aucun résultat de pilote disponible.')
# Evidence ledger rather than decorative roadmap.
columns = [(56, 231), (293, 598), (905, 319)]
for x, width, title in [(56, 220, 'QUESTION'), (293, 588, 'MÉTHODE'), (905, 319, 'CRITÈRE DE DÉCISION')]:
    txt(x, 247, title, 15, GRAY, True, w=width)
for y, question, method, criterion in [
    (288, 'Le signal est-il\nutilisable ?',
     'Prévisions gelées avant chaque fenêtre ; cibles observées ; météo disponible à la date de prévision.',
     'Gain stable par rue et par horizon face à J−7.'),
    (390, 'Le planning\ncoûte-t-il moins ?',
     'Dans une même zone, comparer planning actuel et assisté sur des groupes de tournées de charge comparable.',
     'Baisse du coût par colis livré, avec intervalle d’incertitude.'),
    (503, 'Le gain peut-il\nêtre déployé ?',
     'Vérifier heures payées, retards, échecs de livraison et coût complet de la solution.',
     'Rentabilité positive ; ponctualité dans la tolérance fixée avant le test.'),
]:
    line(56, y-11, 1224, y-11)
    txt(56, y, question, 23, INK, True, w=220)
    txt(293, y, method, 21, INK, w=572)
    txt(905, y, criterion, 21, GREEN, True, w=319)
conclusion('Affecter aléatoirement les groupes si possible ; ne pas choisir les meilleures journées après le test.', y=610)
note('Une comparaison prospective, pas une démonstration sélectionnée',
     'Séparer validation prédictive et expérimentation opérationnelle. Les fenêtres déjà utilisées pour choisir les modèles sont des données de développement. '
     'Prévoir une période chronologique ultérieure gelée ; examiner erreurs par horizon, rue, type de jour et qualité des données. '
     'Pour l’impact business, le contrôle est le planning réellement utilisé par l’exploitation. '
     'Randomiser des groupes de tournées comparables au sein d’une zone, à une granularité limitant les échanges de livreurs et les interférences. '
     'Ne pas comparer simplement Convention traitée à Saint-Pères témoin : les différences de rue seraient confondues avec l’effet. '
     'Si la randomisation est impossible, utiliser un appariement explicite et annoncer les limites d’interprétation causale. '
     'Définir avant le test le coût complet par colis effectivement livré, une marge de non-dégradation de la ponctualité et le traitement des fériés/incidents. '
     'Estimer l’incertitude en tenant compte du groupement des tournées et des jours. Dimensionner l’échantillon selon la variance observée et l’effet minimal rentable. '
     'Aucun seuil numérique ou délai de significativité n’est inventé. Ne pas extrapoler une journée fériée à une année.')

# 8 — Concrete recommendation with gates, no fabricated six-week delivery promise.
new('Décision et conditions de passage',
    'Engager le pilote sur Convention ; étendre seulement\nsi les gains couvrent les coûts de la solution',
    'Saint-Pères sert de second cas d’usage ; Champs-Élysées reste sous supervision jusqu’au retour de données fiables.',
    'Recommandation fondée sur les scores reportés. Ordre des étapes proposé ; charge, budget et calendrier à établir avec l’exploitation.')
for x, title, width in [(56, 'ÉTAPE', 250), (370, 'LIVRABLE ATTENDU', 510), (932, 'RESPONSABLE', 292)]:
    txt(x, 247, title, 15, GRAY, True, w=width)
for y, stage, deliverable, who in [
    (291, '1. Instrumenter', 'Tournées, GPS, arrêts, colis et heures payées reliés ; coût de référence calculé.', 'Exploitation + Data'),
    (388, '2. Tester les décisions', 'Durées calibrées ; comparaison des créneaux sous contraintes ; pilote avec contrôle.', 'Exploitation + Data Science'),
    (499, '3. Décider du déploiement', 'Gains nets et ponctualité estimés ; budget et règles de repli documentés.', 'Sponsor opérations'),
]:
    line(56, y-11, 1224, y-11)
    txt(56, y, stage, 23, INK, True, w=290)
    txt(370, y, deliverable, 22, INK, w=508)
    txt(932, y, who, 21, GRAY, w=292)
conclusion('À obtenir aujourd’hui : un référent exploitation, l’accès aux données et l’accord sur le protocole.')
note('Terminer par une demande opérationnelle',
     'Demander l’autorisation de préparer le pilote, les données nécessaires et un interlocuteur responsable des plannings. '
     'Ne pas demander un budget au montant inventé ; établir les coûts d’intégration, d’exploitation et d’accompagnement après cadrage. '
     'Convention est la première priorité selon les résultats disponibles. Saint-Pères peut être un second terrain avec comparaison interne, '
     'utile pour tester un gain prédictif moindre. Les Champs-Élysées nécessitent une stratégie de données actuelle avant autonomie. '
     'Le solveur et les modèles de durée viennent après l’instrumentation. L’absence de preuve économique ou une dégradation du service '
     'doit conduire à adapter ou arrêter le déploiement, sans avoir à défendre le modèle à tout prix. '
     'Ne pas promettre un calendrier de six semaines ni une puissance statistique avant de connaître les volumes.')

# Original technical evidence in appendices; figures are retained as images.
source_doc = pymupdf.open(SOURCE) if SOURCE.exists() else None
appendices = [(2, 'Sources et qualité des données'), (3, 'Spécification du modèle'),
              (5, 'Prévisions du 7 au 11 novembre : scénarios à confirmer'),
              (8, 'Réel et prédit par rue'), (9, 'Profils horaires'),
              (10, 'Relation entre débit et occupation'), (11, 'Importance des variables'),
              (12, 'Tableau complet des scores'), (13, 'Liste des 48 variables')]
for idx, title in appendices:
    new('Annexe | Support du 5 novembre', title,
        source=(f'Source : LivraisonCo_CTM.pdf, p. {idx+1}. Créneaux exploratoires ; pont et restrictions du 11 novembre 2026 à confirmer.'
                if idx == 5 else
                f'Source : LivraisonCo_CTM.pdf, p. {idx+1}. Figure ou tableau d’origine ; résultats non recalculés.'))
    path = ASSETS/f'annexe_source_{idx+1:02d}.png'
    if source_doc is not None:
        p = source_doc[idx]
        clip = pymupdf.Rect(p.rect.width*.035, p.rect.height*.15, p.rect.width*.965, p.rect.height*.935)
        p.get_pixmap(matrix=pymupdf.Matrix(3.4, 3.4), clip=clip, alpha=False).save(path)
    if not path.exists():
        raise FileNotFoundError(f'Figure manquante : {path}. Restaurer cette figure ou fournir {SOURCE}.')
    with pymupdf.open(path) as figure:
        aspect = figure[0].rect.width / figure[0].rect.height
    h = 451
    w = h*aspect
    image(path, (W-w)/2, 190, w, h)
    note(title, f'Extrait de la page {idx+1} du support fourni, sans modification des chiffres ni recalcul. '
         'Titre et pied de page d’origine retirés ; corps conservé comme image. Les chiffres et figures internes à cette image ne sont pas éditables. '
         'Pour interpréter les scores, rappeler que la période des Champs-Élysées est 2025. '
         'La disponibilité réelle de la météo et l’ensemble du protocole doivent être vérifiés dans les artefacts de validation. '
         'Pour la prévision du 11 novembre, la fermeture mentionnée dans la figure provient du scénario du support initial : '
         'aucune restriction 2026 n’a été confirmée ici. Les créneaux affichés restent à valider avec les contraintes clients et les restrictions réelles.')

pdf.save()
prs.save(ROOT/'LivraisonCo_business.pptx')
(ROOT/'notes_orateur.md').write_text('# LivraisonCo — Notes de présentation\n\n'+ '\n'.join(notes))
(ROOT/'layout_bounds.json').write_text(json.dumps(text_bounds, ensure_ascii=False, indent=2)+'\n')
# A short PDF is convenient when sharing only the decision narrative.
complete = pymupdf.open(ROOT/'LivraisonCo_business.pdf')
short = pymupdf.open()
short.insert_pdf(complete, from_page=0, to_page=7)
short.set_metadata({'title': prs.core_properties.title, 'author': 'Les Architectes'})
short.save(ROOT/'LivraisonCo_synthese.pdf')
print(f'{page} slides : 8 principales et {len(appendices)} annexes. PDF court : 8 pages.')
