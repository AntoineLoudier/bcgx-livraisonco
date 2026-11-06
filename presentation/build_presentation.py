"""Build the review deck in PDF and editable PowerPoint from explicit source data.

Run: .venv/bin/python presentation/build_presentation.py
Scores and figures come from refresh_evidence.py after acquisition and training.
The renderer does not rerun models or modify the evidence snapshots.
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
ASSETS = ROOT / 'assets'
ASSETS.mkdir(exist_ok=True)
RESULTS = ROOT / 'results'
evidence = json.loads((ROOT/'scores_source.json').read_text())
manifest = json.loads((RESULTS/'manifest.json').read_text())
scores = evidence['scores']
assert [row['rue'] for row in scores] == ['Convention', 'Saint-Pères']
global_scores = {row['Target']:row for row in evidence['global_2_rues']}
revision = manifest['model_revision'][:7]
feature_count = manifest['feature_count']
score_by_street = {row['rue']:row for row in scores}
cv = score_by_street['Convention']
sp = score_by_street['Saint-Pères']
W, H = 1280, 720
INK, GREEN, GRAY = '#1D2B32', '#007A53', '#5A686E'
LIGHT, RULE, PALE = '#EFF3F1', '#D2DAD7', '#E0EEE7'
MID, AMBER, WHITE = '#81938D', '#94631F', '#FFFFFF'
FONTS = Path('/usr/share/fonts/truetype/croscore')
font_files = [('Body', 'Arimo-Regular.ttf'), ('Bold', 'Arimo-Bold.ttf')]
if not all((FONTS / filename).is_file() for _, filename in font_files):
    from matplotlib import get_data_path
    FONTS = Path(get_data_path()) / 'fonts' / 'ttf'
    font_files = [('Body', 'DejaVuSans.ttf'), ('Bold', 'DejaVuSans-Bold.ttf')]
for name, filename in font_files:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / filename)))

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W/96), Inches(H/96)
prs.core_properties.title = 'LivraisonCo — Résultats et décision de pilote'
prs.core_properties.author = 'Les Architectes'
pdf = canvas.Canvas(str(ROOT / 'LivraisonCo.pdf'), pagesize=(W, H))
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


def image_viewport(path, x, y, w, h, top, bottom):
    """Display a cropped viewport; preserve the approved image file unchanged."""
    full_h = h/(1-top-bottom)
    pdf.saveState()
    clip = pdf.beginPath()
    clip.rect(x, H-y-h, w, h)
    pdf.clipPath(clip, stroke=0)
    pdf.drawImage(str(path), x, H-(y-full_h*top)-full_h, w, full_h)
    pdf.restoreState()
    pic = slide.shapes.add_picture(str(path), Inches(x/96), Inches(y/96), Inches(w/96), Inches(h/96))
    pic.crop_top, pic.crop_bottom = top, bottom


def new(section, title, subtitle='', source='', compact=False):
    global slide, page
    if page:
        pdf.showPage()
    page += 1
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    rect(0, 0, W, H, WHITE)
    txt(56, 25, 'LIVRAISONCO', 13, GREEN, True, w=200)
    txt(855, 25, section, 13, GRAY, w=370)
    txt(56, 74, title, 35, INK, True, w=1168, leading=40)
    if not compact:
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

# 1 — Recommendation memo, evidence and economic question.
new('Synthèse et recommandation',
    'Réduire les heures de livraison en adaptant les créneaux ;\ncommencer le pilote sur Convention',
    'Le modèle améliore la prévision du trafic. La prochaine étape est de mesurer son effet sur les tournées.',
    f'Source : acquisition du 6 novembre, modèle {revision}. Gains face à J−7 prétraité. Simulation économique ; économies non observées.')
line(841, 247, 841, 577)
for y, n, title, body in [
    (247, '01', 'Un premier cas d’usage sur Convention',
     f'−{dec(cv["gain_q"],0)} % de RMSE sur le débit et −{dec(cv["gain_k"],0)} % sur l’occupation face à J−7 prétraité. Gain de débit plus modeste sur Saint-Pères.'),
    (362, '02', 'Une décision précise à tester',
     'Comparer les créneaux d’une même tournée, puis choisir le moins coûteux parmi ceux compatibles avec les contraintes clients.'),
    (477, '03', 'Deux rues dans le périmètre évalué',
     'Convention et Saint-Pères : 240 lignes horaires à prévoir du 7 au 11 novembre, avec q et k. Champs-Élysées est hors évaluation.'),
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
     'Convention est le premier candidat sur la base du backtest recalculé, pas une garantie de rentabilité ou de significativité statistique. '
     f'Les gains exacts sont {dec(cv["gain_q"],2)} % sur q et {dec(cv["gain_k"],2)} % sur k face au J−7 prétraité. Saint-Pères peut être un second site, '
     'avec son propre groupe témoin : ce n’est pas un contrôle interchangeable de Convention. '
     '1 h/jour × 10 €/h × 250 jours = 2 500 €/an, avant coût de solution. Le taux 10 €/h vient du cas ; les 250 jours sont une hypothèse. '
     'À heures payées constantes, le gain est de la capacité ; ne pas l’appeler économie de trésorerie. '
     'Selon la précision de périmètre donnée par l’utilisateur, les Champs-Élysées sont exclus de tous les scores et graphiques de résultat du support. '
     'Le pipeline officiel conserve son export de 360 lignes, mais le périmètre évalué correspond à 240 lignes, chacune portant q et k. '
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

# 3 — Refreshed two-street performance, with explicit reference and scope.
new('Performance | Deux rues évaluées',
    f'L’erreur de débit baisse de {dec(cv["gain_q"],0)} % sur Convention,\ncontre {dec(sp["gain_q"],0)} % sur Saint-Pères face à J−7 prétraité',
    'Quatre fenêtres de cinq jours par rue ; cibles observées uniquement ; acquisition actualisée le 6 novembre.',
    f'Source : results/scores_2_rues.csv · modèle {revision}. Backtest de développement avec météo historique observée.')
txt(56, 243, 'Débit q — RMSE (véhicules/heure)', 23, INK, True, w=725)
for x, col, name in [(56, '#BDC7C2', 'J−7 prétraité'), (244, MID, 'Ancrage'), (412, GREEN, 'Modèle final')]:
    rect(x, 284, 16, 12, col)
    txt(x+25, 279, name, 18, GRAY, w=165)
bar_x, scale = 259, 4.35
for i, data in enumerate(scores):
    y = 333+i*143
    txt(56, y+22, data['rue'], 23, INK, True, w=195)
    for j, (key, color) in enumerate([('q_j7', '#BDC7C2'), ('q_ancrage', MID), ('q_final', GREEN)]):
        yy = y+j*28
        rect(bar_x, yy, data[key]*scale, 18, color)
        txt(bar_x+data[key]*scale+9, yy-3, dec(data[key]), 18, color if j==2 else GRAY, j==2, w=80)
    txt(56, y+57, f'−{dec(data["gain_q"])} %', 19, GREEN, True, w=197)
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
txt(822, 514, 'Ensemble des deux rues', 21, INK, True, w=400)
txt(822, 548, f'q : −{dec(global_scores["q"]["Gain_vs_J7_pretraite_pct"])} %   |   k : −{dec(global_scores["k"]["Gain_vs_J7_pretraite_pct"])} %', 23, GREEN, True, w=400)
txt(822, 587, f'{global_scores["q"]["N"]} heures observées · hors Champs-Élysées', 17, GRAY, w=400)
note('Résultats actualisés sur le périmètre évalué',
     'Les scores sont recalculés après réacquisition du trafic et de la météo, prétraitement, features et entraînement du code main. '
     'Les quatre fenêtres sont sélectionnées par le protocole existant, avec au moins 85 % de cibles observées ; les dates exactes figurent en annexe. '
     f'Convention : q {dec(cv["q_j7"],2)} → {dec(cv["q_final"],2)} ; k {dec(cv["k_j7"],2)} → {dec(cv["k_final"],2)}. '
     f'Saint-Pères : q {dec(sp["q_j7"],2)} → {dec(sp["q_final"],2)} ; k {dec(sp["k_j7"],2)} → {dec(sp["k_final"],2)}. '
     'Le J−7 du pipeline est lissé et corrigé des jours spéciaux : ce n’est pas une simple mesure brute de J−7. '
     'Une référence non lissée sur la série nettoyée/imputée est fournie séparément dans les résultats, sur les mêmes lignes. '
     'Le score global est calculé sur toutes les cibles des deux rues réunies, pas en moyennant les pourcentages. '
     'Il s’agit d’un backtest de développement, avec météo historique observée plutôt que prévisions archivées ; ne pas annoncer un test indépendant '
     'ni une performance opérationnelle certifiée. La comparaison ancienne/nouvelle version à données identiques est détaillée en annexe.')

# 4 — Model updated to the implementation pulled from main.
new('Modèle | Version actualisée',
    'Le profil hebdomadaire est lissé ; la correction\nest régularisée différemment selon la rue',
    f'{feature_count} variables · quatre couples rue/cible évalués · prévisions horaires du 7 au 11 novembre 2026.',
    f'Source : src/features.py et src/model.py, révision {revision}. Backtest avec météo observée ; validation prospective à compléter.')
rect(56, 249, 1168, 85, LIGHT)
txt(77, 272, 'Prévision', 29, INK, True, w=185)
txt(262, 271, '=', 30, GRAY, w=40)
txt(320, 272, 'Ancrage stabilisé', 29, INK, True, w=355)
txt(693, 271, '+', 30, GRAY, w=40)
txt(756, 272, 'Correction LightGBM / XGBoost', 29, GREEN, True, w=450)
line(705, 360, 705, 529)
txt(56, 360, 'Réduire le bruit des historiques', 23, INK, True, w=620)
txt(56, 400, 'Lissage de J−7 à ±1 h ; remplacement des jours spéciaux ; médiane sur quatre semaines.', 22, GRAY, w=605)
txt(56, 468, 'L’ancrage mêle la moyenne 60/25/15 et la médiane, puis applique les règles de calendrier.', 21, GRAY, w=605)
txt(751, 360, 'Limiter les corrections instables', 23, INK, True, w=470)
txt(751, 400, 'Résidus extrêmes écrêtés à l’entraînement ; profondeur et poids adaptés à chaque rue.', 21, GRAY, w=470)
txt(751, 468, 'Saint-Pères : correction atténuée à 0,40 sur q. Convention : poids de 1,05 sur q.', 21, GRAY, w=470)
line(56, 552, 1224, 552)
txt(56, 572, 'À démontrer ensuite', 18, GREEN, True, w=205)
txt(281, 567, 'Stabilité sur des fenêtres nouvelles, avec les prévisions météo disponibles\nau moment de la décision et les délais réels de publication du trafic.', 21, INK, w=942)
note('Ce qui a changé dans la version main',
     'Le lissage utilise les heures t−169, t−168 et t−167 avec des poids 0,10/0,80/0,10. '
     'Les jours spéciaux des lags sont remplacés en cascade J−21, puis J−14, puis J−7. '
     'La moyenne 60/25/15 est mélangée à la médiane J−7/J−14/J−21/J−28 : 90/10 sur Convention et 82/18 sur Saint-Pères. '
     'Le modèle ajoute les médianes q et k et retire is_summer_holiday de la liste des entrées : 49 variables. '
     'Les résidus d’entraînement sont écrêtés aux centiles 1 et 99 sur les deux rues. '
     'La moyenne des corrections LightGBM et XGBoost est pondérée : Convention 1,05 pour q et 1,00 pour k ; Saint-Pères 0,40 pour q et 0,85 pour k. '
     'L2=4 sur Convention, 5 sur Saint-Pères ; profondeur et nombre d’arbres réduits sur Saint-Pères. '
     'Le ratio de pont est borné entre 0,75 et 0,95. Pour les deux rues évaluées, un férié applique 0,80 à l’ancrage ; '
     'le profil de défilé de 2025 ne s’applique qu’aux Champs-Élysées, hors évaluation. Ces coefficients restent des hypothèses de modèle. '
     'Les variables météo historiques de ce backtest sont observées ; la nouvelle acquisition ne constitue pas une archive des prévisions émises dans le passé.')

# 5 — The missing operational link; economic threshold shown as a real curve.
new('Passage à la décision',
    'Le trafic prévu doit être converti en durée de tournée\navant de recommander un changement de créneau',
    'Débit et occupation décrivent le trafic ; ils ne donnent pas directement le temps ou le coût d’une livraison.',
    'Tarifs du cas : 10 €/h le jour, 14 €/h la nuit (support initial, p. 7). Calcul à travail et autres coûts identiques ; nuit fixée à 5 h.')
txt(56, 243, 'Ce qui manque entre la prévision et le planning', 23, INK, True, w=660)
for y, number, title, body in [
    (292, '1', 'Trafic par rue et par heure', 'Disponible : q et k, sur deux rues évaluées et cinq jours.'),
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

# 6 — Approved image, embedded unchanged; viewport trims only the artboard margins.
new('Produit | Maquette validée', '', compact=True)
txt(56, 45, 'Illustration des usages exploitation / livreur · Données fictives · Champs-Élysées hors évaluation', 13, GRAY, w=1168)
mockup = ASSETS/'maquette_exploitation_livreur.png'
viewport_h = 612
viewport_w = viewport_h*1586/(992-100-40)
image_viewport(mockup, (W-viewport_w)/2, 65, viewport_w, viewport_h, 100/992, 40/992)
note('Maquette validée : du planning à l’exécution',
     'La maquette graphique séparée validée par l’utilisateur est intégrée comme image dans cette slide. '
     'Le fichier source n’a pas été modifié : seuls le titre et les marges extérieures de la planche sont masqués dans la mise en page. '
     'L’exploitation compare les créneaux et vérifie les contraintes ; le livreur consulte ensuite un planning validé et enregistre arrivée ou retard. '
     'Les deux écrans représentent des étapes successives. La courbe, les horaires et la carte sont illustratifs et ne reprennent pas les prévisions recalculées. '
     'La ligne Champs-Élysées reste visible dans cette maquette approuvée à titre d’exemple de revue manuelle ; elle ne fait pas partie de l’évaluation. '
     'Aucune application opérationnelle ni optimisation de tournée n’est revendiquée. L’image est remplaçable dans PowerPoint mais ses textes internes ne sont pas éditables.')

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
    'Convention et Saint-Pères constituent le périmètre évalué ; Champs-Élysées est exclu des scores.',
    'Recommandation fondée sur le backtest actualisé. Charge, budget et calendrier à établir avec l’exploitation.')
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
     'utile pour tester un gain prédictif moindre. Les Champs-Élysées sont hors évaluation selon le périmètre précisé. '
     'Le solveur et les modèles de durée viennent après l’instrumentation. L’absence de preuve économique ou une dégradation du service '
     'doit conduire à adapter ou arrêter le déploiement, sans avoir à défendre le modèle à tout prix. '
     'Ne pas promettre un calendrier de six semaines ni une puissance statistique avant de connaître les volumes.')

# Appendices rebuilt from the current two-street evidence; no legacy score image.
def read_rows(name):
    with (RESULTS/name).open() as f:
        return list(csv.DictReader(f))


def short_arc(name):
    return 'Ensemble 2 rues' if name.startswith('GLOBAL') else name


def percent_reduction(value):
    value = float(value)
    return ('−' if value >= 0 else '+')+dec(abs(value))+' %'


def figure_slide(title, filename, subtitle, source, explanation):
    new('Annexe | Deux rues évaluées', title, subtitle, source)
    path = ASSETS/filename
    with pymupdf.open(path) as doc:
        ratio = doc[0].rect.width/doc[0].rect.height
    h = min(416, 1168/ratio)
    w = h*ratio
    image(path, (W-w)/2, 232, w, h)
    note(title, explanation)


comparison_rows = read_rows('comparaison_versions.csv')
comparison_global = {r['Target']:r for r in comparison_rows if r['Arc'].startswith('GLOBAL')}
new('Annexe | Comparaison des versions',
    f'À données identiques, la RMSE baisse encore de {dec(float(comparison_global["q"]["Gain_nouvelle_vs_ancienne_pct"]))} %\nsur q et de {dec(float(comparison_global["k"]["Gain_nouvelle_vs_ancienne_pct"]))} % sur k',
    'Deux versions réentraînées sur les mêmes données actualisées et les mêmes fenêtres ; 944 heures observées.',
    f'Source : results/comparaison_versions.csv. Ancienne version {manifest["comparison_revision"][:7]} ; nouvelle {revision}.')
for x, label, w in [(56,'RUE',260),(342,'CIBLE',100),(492,'ANCIENNE RMSE',220),(768,'NOUVELLE RMSE',220),(1054,'BAISSE',170)]:
    txt(x,247,label,15,GRAY,True,w=w)
for i,row in enumerate(comparison_rows):
    y=296+i*47
    line(56,y-9,1224,y-9)
    txt(56,y,short_arc(row['Arc']),21,INK,True,w=260)
    txt(342,y,row['Target'],21,INK,w=100)
    places=1 if row['Target']=='q' else 2
    txt(492,y,dec(float(row['RMSE_ancienne_version']),places),22,GRAY,w=220)
    txt(768,y,dec(float(row['RMSE_nouvelle_version']),places),22,INK,True,w=220)
    txt(1054,y,percent_reduction(row['Gain_nouvelle_vs_ancienne_pct']),22,GREEN,True,w=170)
conclusion('Le progrès le plus net concerne le débit de Saint-Pères ; les gains restent à confirmer hors développement.')
note('Isoler le progrès du code',
     'Les versions ancienne et nouvelle sont réentraînées sur les mêmes données fraîchement téléchargées, prétraitées avec le même pipeline. '
     'Les dates, cibles et historiques des plis sont identiques. Les valeurs ne comparent donc pas directement les chiffres de l’ancien PDF aux nouvelles fenêtres. '
     'Cette comparaison mesure l’effet de la version sur ce backtest ; elle ne constitue pas un test indépendant de la sélection des hyperparamètres. '
     'Les sorties détaillées et les empreintes de données sont dans results/. Un gain agrégé peut masquer des pertes sur certains plis.')

summary_rows = read_rows('scores_2_rues.csv')
new('Annexe | Scores détaillés',
    'Les résultats globaux agrègent uniquement\nConvention et Saint-Pères',
    'q : RMSE et MAE en véhicules/heure. k : RMSE et MAE en points de pourcentage.',
    'Source : results/scores_2_rues.csv. Référence = J−7 prétraité. Le J−7 non lissé est aussi disponible dans ce fichier.')
for x,label,w in [(56,'RUE',250),(322,'CIBLE',90),(424,'N',80),(526,'J−7',140),(726,'MODÈLE',160),(928,'BAISSE',145),(1090,'MAE',134)]:
    txt(x,247,label,15,GRAY,True,w=w)
for i,row in enumerate(summary_rows):
    y=296+i*47
    line(56,y-9,1224,y-9)
    places=1 if row['Target']=='q' else 2
    txt(56,y,short_arc(row['Arc']),21,INK,True,w=250)
    txt(322,y,row['Target'],21,INK,w=90)
    txt(424,y,row['N'],21,GRAY,w=80)
    txt(526,y,dec(float(row['RMSE_J7_pretraite']),places),22,GRAY,w=140)
    txt(726,y,dec(float(row['RMSE_Modele']),places),22,INK,True,w=160)
    txt(928,y,percent_reduction(row['Gain_vs_J7_pretraite_pct']),22,GREEN,True,w=145)
    txt(1090,y,dec(float(row['MAE_Modele']),places),22,INK,w=134)
conclusion('Le J−7 prétraité intègre déjà le lissage et les jours spéciaux : c’est une référence plus exigeante ici.')
note('Définition exacte des scores',
     'La RMSE globale est calculée en concaténant les erreurs des deux rues, avec pondération implicite par le nombre de mesures réelles. '
     'Convention a 464 heures et Saint-Pères 480, soit 944 heures par cible. Aucun score des Champs-Élysées ne contribue aux valeurs. '
     'Le gain est 1 − RMSE modèle / RMSE référence, calculé avant arrondi. Le J−7 prétraité reprend les colonnes baseline_q/k de la validation officielle. '
     'Le CSV contient aussi la référence J−7 non lissée de la série nettoyée/imputée, pour éviter d’assimiler la référence prétraitée à une observation brute.')

fold_rows = [r for r in read_rows('scores_par_fenetre.csv') if r['Target']=='q']
new('Annexe | Fenêtres de validation',
    'Quatre fenêtres par rue couvrent plusieurs saisons ;\nle gain n’est pas uniforme sur toutes les périodes',
    'Plages des premières et dernières cibles observées ; chaque fenêtre nominale couvre cinq jours.',
    'Source : results/scores_par_fenetre.csv. Heures locales Europe/Paris ; observations imputées exclues.')
for x,label,w in [(56,'RUE / PLI',240),(320,'PREMIÈRE CIBLE',220),(570,'DERNIÈRE CIBLE',220),(840,'N',60),(946,'RMSE q',110),(1100,'J−7 q',124)]:
    txt(x,247,label,15,GRAY,True,w=w)
for i,row in enumerate(fold_rows):
    y=289+i*39
    line(56,y-7,1224,y-7)
    txt(56,y,row['Arc']+' / '+row['Fold'],19,INK,True,w=240)
    for x,key in [(320,'Start'),(570,'End')]:
        val=row[key]
        label=val[8:10]+'/'+val[5:7]+'/'+val[:4]+' '+val[11:16]
        txt(x,y,label,18,GRAY,w=220)
    txt(840,y,row['N'],19,GRAY,w=60)
    txt(946,y,dec(float(row['RMSE_Modele'])),20,INK,True,w=110)
    txt(1100,y,dec(float(row['RMSE_J7_pretraite'])),20,GRAY,w=124)
txt(56,615,'Saint-Pères, août : la référence reste meilleure sur q et k. Un gain moyen ne garantit pas chaque fenêtre.',19,AMBER,w=1168)
note('Lire les fenêtres sans confondre les périodes des rues',
     'Les plages affichées sont celles des cibles effectivement observées. Des heures manquantes en bord de fenêtre expliquent qu’une plage paraisse plus courte '
     'que les 120 heures nominales. Les fenêtres sont choisies par le protocole existant, au moins 85 % de mesures et entraînement sur le passé. '
     'Les horaires ne doivent pas être remplacés par le minimum/maximum commun aux deux rues ou aux Champs-Élysées : ces périodes peuvent différer. '
     'Saint-Pères perd face à sa référence lors du pli d’août. Les détails q et k se trouvent dans le CSV ; aucune sélection des plis après le calcul.')

new('Annexe | Sources et périmètre',
    'Les deux rues évaluées disposent de mesures récentes ;\nla météo couvre les 120 heures à prévoir',
    'Acquisition Open Data Paris et Open-Meteo relancée le 6 novembre 2026.',
    'Source : results/manifest.json. Dernière mesure à 00 h le 6 novembre en heure de Paris, soit 23 h UTC le 5 novembre.')
for x,label,w in [(56,'RUE',270),(370,'MESURES q ET k',270),(710,'ENTRAÎNEMENT',230),(1000,'ARC ID',200)]:
    txt(x,247,label,15,GRAY,True,w=w)
for i,row in enumerate(manifest['data_quality']):
    y=296+i*62
    line(56,y-9,1224,y-9)
    txt(56,y,row['rue'],23,INK,True,w=270)
    txt(370,y,str(row['heures_reelles_q_k'])+' h',23,INK,w=270)
    txt(710,y,str(row['heures_train'])+' h',23,INK,w=230)
    txt(1000,y,str(row['troncons'][0]['iu_ac']),23,INK,w=200)
line(56,430,1224,430)
txt(56,450,'Périmètre évalué',22,GREEN,True,w=310)
txt(390,450,'Convention et Saint-Pères uniquement ; Champs-Élysées exclu selon la précision reçue.',21,INK,w=820)
txt(56,517,'Météo de la cible',22,GREEN,True,w=310)
txt(390,517,f'{manifest["weather_target_hours"]} heures du 7 au 11 novembre, sans valeur manquante. Prévisions actuelles disponibles.',21,INK,w=820)
txt(56,588,'Limite du backtest',22,AMBER,True,w=310)
txt(390,588,'Les fenêtres historiques utilisent la météo observée, pas des prévisions archivées à leur date.',21,INK,w=820)
note('Sources et limites de disponibilité',
     'Les identifiants et les couples de nœuds exacts sont conservés dans le manifeste. Les données brutes ont été retéléchargées plutôt que réutilisées. '
     'Les comptes de mesures sont calculés après nettoyage et agrégation horaire ; les doublons de changement d’heure expliquent un éventuel écart au compte brut. '
     'La complétude de la météo du 7–11 novembre concerne la prévision actuelle. Elle ne résout pas la question des vintages météo des backtests. '
     'L’export officiel reste compatible avec les trois arcs du code ; ce support et son agrégat d’évaluation portent exclusivement sur les deux rues demandées.')

new('Annexe | Paramètres de la version main',
    'La nouvelle version stabilise l’ancrage et réduit\nl’amplitude des corrections sur Saint-Pères',
    f'Implémentation {revision} ; coefficients fixés par le code et évalués dans le backtest.',
    'Sources : src/features.py, src/model.py ; versions et empreintes dans results/manifest.json.')
for x,label,w in [(56,'PARAMÈTRE',620),(710,'CONVENTION',230),(1000,'SAINT-PÈRES',224)]:
    txt(x,247,label,15,GRAY,True,w=w)
parameter_rows=[
    ('Moyenne 3 semaines / médiane 4 semaines','90 % / 10 %','82 % / 18 %'),
    ('Poids de la correction q / k','1,05 / 1,00','0,40 / 0,85'),
    ('Arbres par famille','400','350'),
    ('Profondeur LightGBM / XGBoost','5 / 4','4 / 3'),
    ('Régularisation L2','4','5'),
    ('Résidus : écrêtage entraînement','P1 – P99','P1 – P99'),
]
for i,(name,a,b) in enumerate(parameter_rows):
    y=290+i*45
    line(56,y-8,1224,y-8)
    txt(56,y,name,21,INK,w=620)
    txt(710,y,a,22,INK,True,w=230)
    txt(1000,y,b,22,INK,True,w=224)
txt(56,584,'Lissage J−7 : 10 % à h−1, 80 % à h, 10 % à h+1. Férié : ancrage × 0,80 sur les deux rues.\nLe ratio de pont est borné entre 0,75 et 0,95 ; ces règles restent des hypothèses de modèle.',19,GRAY,w=1168)
note('Détail de la correction',
     'La correction est la moyenne 50/50 LightGBM et XGBoost multipliée par le poids de la rue et de la cible. '
     'Le lissage horaire à J−7 utilise uniquement des valeurs âgées de plus de six jours et reste disponible à horizon cinq jours hors délai de publication. '
     'Les heures spéciales sont remplacées en cascade inverse. Les règles calendrier ne doivent pas être présentées comme des effets causalement estimés. '
     'Le profil Armistice 2025 spécifique aux Champs-Élysées ne s’applique pas aux deux rues évaluées.')

figure_slide('Prévisions actualisées du 7 au 11 novembre', 'current_forecast.png',
    '240 lignes horaires sur deux rues ; zone beige = 11 novembre. Prévisions, pas observations futures.',
    f'Source : results/previsions_7_11_novembre_2_rues.csv · modèle {revision}. Hypothèses de pont et de férié intégrées au modèle.',
    'Les courbes proviennent des modèles finaux réentraînés après acquisition. Aucun repli sur l’ancrage n’a été utilisé pour dessiner ces résultats. '
    'La courbe de la maquette produit reste illustrative et distincte de ces prévisions. Aucun itinéraire ni créneau optimal n’est déduit automatiquement de ces seules courbes.')
figure_slide('Dernière fenêtre : mesures et prédictions', 'current_oof.png',
    'Fenêtre du 1er au 6 novembre ; seules les heures dont les cibles sont observées sont tracées.',
    'Source : results/predictions_validation_2_rues.csv. Référence = J−7 prétraité ; même backtest que les scores.',
    'Deux rues et deux cibles, même dernière fenêtre nominale. Les trous éventuels correspondent à des cibles non observées. '
    'Il s’agit de prédictions hors entraînement temporel dans ce backtest de développement, pas d’un holdout indépendant du travail de sélection.')
figure_slide('Profils horaires sur les mesures actualisées', 'current_profiles.png',
    'Moyennes des jours de semaine et des week-ends ; description de l’historique utilisé.',
    'Source : features_train.csv après réacquisition et prétraitement ; empreinte conservée dans le manifeste.',
    'Profils descriptifs calculés sur les cibles réelles des deux rues. Ils ne constituent pas une évaluation du modèle et ne tiennent pas lieu de prévision de tournée.')
figure_slide('Débit et occupation apportent des informations différentes', 'current_fundamental.png',
    'Mesures réelles des deux rues. Un faible débit peut correspondre à une route vide ou à une congestion.',
    'Source : features_train.csv après réacquisition et prétraitement ; pas de seuil universel de saturation imposé.',
    'Les nuages de points sont descriptifs. L’occupation n’est ni une vitesse ni directement un temps de parcours. '
    'Un seuil de congestion opérationnel doit être calibré avec des mesures terrain.')
figure_slide('Les variables expliquent la correction, pas tout le trafic', 'current_importance.png',
    'Impact absolu moyen de la composante LightGBM, après son poids de 50 % et le poids résiduel de la rue.',
    'Source : results/importance_variables.csv. Échantillon des 2 000 dernières lignes d’entraînement par rue ; XGBoost et ancrage non expliqués.',
    'Les valeurs SHAP du pipeline portent sur le résidu LightGBM. Ici elles sont multipliées par 0,5 et par le poids résiduel du modèle pour exprimer '
    'la contribution de cette composante avant clipping de la prédiction. Elles ne couvrent ni XGBoost, ni l’ancrage, ni le post-traitement non linéaire. '
    'Ce graphique décrit les modèles finaux sur un échantillon d’entraînement, pas la causalité des variables ni un effet économique.')

new('Annexe | Variables du modèle',
    f'{feature_count} variables entrent dans les arbres ;\nles deux médianes hebdomadaires ont été ajoutées',
    'is_summer_holiday est retiré des entrées ; le calendrier, la météo et les historiques restent disponibles.',
    f'Source : FEATURE_COLS dans src/model.py, modèle {revision}. Liste complète dans results/manifest.json.')
feature_groups=[
    ('Calendrier · 13',manifest['feature_columns'][:13]),
    ('Événements · 4',manifest['feature_columns'][13:17]),
    ('Météo / interactions · 8',manifest['feature_columns'][17:25]),
    ('Historique q / k · 20',manifest['feature_columns'][25:43]+manifest['feature_columns'][-2:]),
    ('Variables croisées · 4',manifest['feature_columns'][43:47]),
]
assert sum(len(vals) for _,vals in feature_groups)==feature_count
for y,(name,vals) in zip([247,312,354,416,555],feature_groups):
    line(56,y-9,1224,y-9)
    txt(56,y,name,21,INK,True,w=285)
    txt(360,y,', '.join(vals),17,GRAY,w=864,leading=22)
note('Variables exactes',
     'La liste est extraite du code exécuté, conservée dans le manifeste, puis répartie entre familles. '
     'Les variables is_road_closed, is_paris_respire et is_armistice_parade_hours peuvent être constantes sur certaines rues ; '
     'leur présence dans la liste ne signifie pas qu’elles contribuent à leurs modèles. Aucun gain causal n’est attribué individuellement à l’ajout des médianes.')

appendix_count = page-8

pdf.save()
prs.save(ROOT/'LivraisonCo.pptx')
(ROOT/'notes_orateur.md').write_text('# LivraisonCo — Notes de présentation\n\n'+ '\n'.join(notes))
print(f'{page} slides : 8 principales et {appendix_count} annexes. PDF et PowerPoint complets.')
