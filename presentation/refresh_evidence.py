"""Export auditable presentation evidence after refreshing the official pipeline.

Run in order from repository root:
  .venv/bin/python src/data_loader.py
  .venv/bin/python src/preprocessing.py
  .venv/bin/python src/features.py
  .venv/bin/python main_train.py
  .venv/bin/python main_predict.py
  .venv/bin/python presentation/refresh_evidence.py

The official pipeline can still export three streets. Presentation scores and
figures explicitly use Convention and Saint-Pères only, per the updated brief.
"""
from pathlib import Path
import sys, os, json, hashlib, subprocess, types, importlib.metadata
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/livraisonco-mpl')
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src import model, features
from src.evaluate import get_temporal_cv_splits

ARCS = ['Convention', 'Saint-Pères']
BASE_REVISION = 'e6c9818234eb18e90850f3825774852218792d72'
RESULTS = ROOT / 'results'
ASSETS = ROOT / 'assets'
RESULTS.mkdir(exist_ok=True)
ASSETS.mkdir(exist_ok=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def old_module(path, name):
    source = subprocess.check_output(['git', 'show', BASE_REVISION+':'+path], cwd=REPO, text=True)
    mod = types.ModuleType(name)
    mod.__file__ = f'{BASE_REVISION}:{path}'
    exec(compile(source, mod.__file__, 'exec'), mod.__dict__)
    return mod


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def write_csv(df, name):
    df.to_csv(RESULTS/name, index=False, lineterminator='\n')


# Require outputs newer than the data and the model implementation. Never silently
# publish stale reports after merely pulling new source code.
required = ['data/processed/features_train.csv', 'data/processed/features_test_nov7_11.csv',
            'data/processed/cv_predictions_oof.csv', 'models/ensemble_models_q_k.joblib',
            'reports/rmse_by_arc_and_outcome.csv', 'reports/feature_importances_by_arc_outcome.csv']
latest_input = max((REPO/p).stat().st_mtime for p in [
    'data/raw/traffic_raw.csv', 'data/raw/weather_raw.csv', 'src/model.py', 'src/features.py',
    'src/preprocessing.py', 'src/evaluate.py', 'main_train.py'])
for name in required:
    if not (REPO/name).exists() or (REPO/name).stat().st_mtime < latest_input:
        raise RuntimeError(f'Artifact absent or stale: {name}; rerun preprocessing, features and main_train.')

train = pd.read_csv(REPO/'data/processed/features_train.csv', parse_dates=['t_1h'])
clean = pd.read_csv(REPO/'data/processed/traffic_clean.csv', parse_dates=['t_1h'])
weather = pd.read_csv(REPO/'data/raw/weather_raw.csv', parse_dates=['t_1h'])
oof = pd.read_csv(REPO/'data/processed/cv_predictions_oof.csv', parse_dates=['t_1h'])
oof = oof[oof.arc.isin(ARCS)].copy()
assert not oof.duplicated(['arc', 't_1h']).any()
assert (oof[['is_imputed_q', 'is_imputed_k']]==0).all().all()
assert np.isfinite(oof[['q','k','pred_q','pred_k','baseline_q','baseline_k']]).all().all()

# An unchanged, unsmoothed weekly reference is supplied separately. The pipeline's
# "baseline_q/k" is a cleaned and smoothed lag, not a naive raw J-7 observation.
lag = clean[['arc','t_1h','q','k']].copy().sort_values(['arc','t_1h'])
for target in ['q','k']:
    lag['unadjusted_j7_'+target] = lag.groupby('arc')[target].shift(168)
oof = oof.merge(lag[['arc','t_1h','unadjusted_j7_q','unadjusted_j7_k']],
                on=['arc','t_1h'], how='left', validate='one_to_one')
assert oof[['unadjusted_j7_q','unadjusted_j7_k']].notna().all().all()

# Replay the previous implementation on the same refreshed inputs and CV keys.
# This comparison isolates the code revision from changes in acquisition/window selection.
old_features = old_module('src/features.py', 'previous_features')
old_model = old_module('src/model.py', 'previous_model')
old_full = old_features.extend_grid_to_test_period(clean, weather.groupby('t_1h', as_index=False).mean(numeric_only=True))
old_full = old_features.add_calendar_and_event_features(old_full)
old_full = old_features.add_weather_interaction_features(old_full)
old_full = old_features.add_causal_lags_and_sncf_patterns(old_full)
old_aligned = old_full.set_index(['arc','t_1h']).loc[
    pd.MultiIndex.from_frame(train[['arc','t_1h']])].reset_index()
assert old_aligned[['arc','t_1h']].equals(train[['arc','t_1h']].reset_index(drop=True))
previous_predictions = []
for fold, (train_idx, val_idx, _, _) in enumerate(get_temporal_cv_splits(train, n_splits=4, val_days=5), start=1):
    print(f'Comparaison ancienne version : fenêtre {fold}/4', flush=True)
    for arc in ARCS:
        sub_train = old_aligned.loc[train_idx]
        sub_train = sub_train[sub_train.arc.eq(arc)].sort_values('t_1h').reset_index(drop=True)
        sub_val = old_aligned.loc[val_idx]
        sub_val = sub_val[sub_val.arc.eq(arc)].copy()
        row = sub_val[['arc','t_1h']].copy()
        row['fold'] = fold
        for target in ['q','k']:
            m = old_model.ResidualEnsembleModel(target=target, arc=arc).fit(sub_train)
            row['previous_'+target] = m.predict(sub_val)
        previous_predictions.append(row)
old_oof = pd.concat(previous_predictions, ignore_index=True)
oof = oof.merge(old_oof, on=['arc','t_1h','fold'], how='left', validate='one_to_one')
assert oof[['previous_q','previous_k']].notna().all().all()

summary_rows, fold_rows, comparisons = [], [], []
for arc in ARCS+['GLOBAL (2 rues évaluées)']:
    sub = oof if arc.startswith('GLOBAL') else oof[oof.arc.eq(arc)]
    for target in ['q','k']:
        final = rmse(sub[target], sub['pred_'+target])
        baseline = rmse(sub[target], sub['baseline_'+target])
        prev = rmse(sub[target], sub['previous_'+target])
        raw = rmse(sub[target], sub['unadjusted_j7_'+target])
        summary_rows.append({
            'Arc': arc, 'Target': target, 'N': len(sub),
            'RMSE_J7_pretraite': baseline,
            'RMSE_J7_non_lisse': raw,
            'RMSE_Ancrage': rmse(sub[target], sub[target+'_smart_expected']),
            'RMSE_Modele': final,
            'MAE_Modele': float(np.abs(sub[target]-sub['pred_'+target]).mean()),
            'Gain_vs_J7_pretraite_pct': 100*(1-final/baseline),
            'Gain_vs_J7_non_lisse_pct': 100*(1-final/raw),
        })
        comparisons.append({'Arc':arc, 'Target':target, 'N':len(sub),
                            'RMSE_ancienne_version':prev, 'RMSE_nouvelle_version':final,
                            'Gain_nouvelle_vs_ancienne_pct':100*(1-final/prev)})
for (arc, fold), sub in oof.groupby(['arc','fold']):
    for target in ['q','k']:
        fold_rows.append({'Arc':arc, 'Fold':int(fold), 'Target':target,
                          'Start':sub.t_1h.min().isoformat(), 'End':sub.t_1h.max().isoformat(),
                          'N':len(sub), 'RMSE_Modele':rmse(sub[target],sub['pred_'+target]),
                          'RMSE_J7_pretraite':rmse(sub[target],sub['baseline_'+target]),
                          'RMSE_ancienne_version':rmse(sub[target],sub['previous_'+target])})
summary = pd.DataFrame(summary_rows)
folds = pd.DataFrame(fold_rows)
comparison = pd.DataFrame(comparisons)
write_csv(summary,'scores_2_rues.csv')
write_csv(folds,'scores_par_fenetre.csv')
write_csv(comparison,'comparaison_versions.csv')
# Source evidence retained with the deck, independent of ignored reports/.
cols = ['arc','t_1h','fold','q','k','pred_q','pred_k','baseline_q','baseline_k',
        'unadjusted_j7_q','unadjusted_j7_k','previous_q','previous_k',
        'is_imputed_q','is_imputed_k','q_smart_expected','k_smart_expected']
write_csv(oof[cols], 'predictions_validation_2_rues.csv')

# Forecast directly with the freshly fitted models; no fallback-to-baseline plots.
final_models = model.load_models()
for (arc,target), trained in final_models.items():
    assert trained.residual_weight == model.ResidualEnsembleModel(target,arc).residual_weight
forecast_features = pd.read_csv(REPO/'data/processed/features_test_nov7_11.csv', parse_dates=['t_1h'])
forecast = forecast_features[forecast_features.arc.isin(ARCS)].copy()
pq, pk = model.predict_with_models(final_models, forecast)
forecast = forecast[['arc','t_1h']].assign(pred_q=pq, pred_k=pk)
assert len(forecast)==240 and np.isfinite(forecast[['pred_q','pred_k']]).all().all()
write_csv(forecast, 'previsions_7_11_novembre_2_rues.csv')

# Source-quality report contains only assessed streets.
quality = []
raw = pd.read_csv(REPO/'data/raw/traffic_raw.csv', sep=';')
for arc in ARCS:
    sub = clean[clean.arc.eq(arc)]
    real = sub[(sub.is_imputed_q==0)&(sub.is_imputed_k==0)]
    original = raw[raw.arc.eq(arc)]
    nodes = original[['iu_ac','libelle_nd_amont','libelle_nd_aval']].drop_duplicates()
    quality.append({'rue':arc, 'heures_grille':len(sub), 'heures_reelles_q_k':len(real),
                    'debut_mesures':real.t_1h.min().isoformat(),
                    'derniere_mesure':real.t_1h.max().isoformat(),
                    'heures_train':int(train.arc.eq(arc).sum()), 'troncons':nodes.to_dict('records')})
current_revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
tracked_inputs = ['data/raw/traffic_raw.csv','data/raw/weather_raw.csv','data/processed/traffic_clean.csv',
                  'data/processed/features_train.csv','data/processed/cv_predictions_oof.csv',
                  'src/model.py','src/features.py','src/preprocessing.py','src/evaluate.py','main_train.py']
metadata = {
    'generated_at_utc':datetime.now(timezone.utc).isoformat(),
    'model_revision':current_revision, 'comparison_revision':BASE_REVISION,
    'evaluated_streets':ARCS, 'excluded_street':'Champs-Elysées',
    'scope_source':'Précision utilisateur : Champs-Élysées ne fait pas partie de l’évaluation.',
    'acquisition_utc':{p:datetime.fromtimestamp((REPO/p).stat().st_mtime,timezone.utc).isoformat()
                       for p in ['data/raw/traffic_raw.csv','data/raw/weather_raw.csv']},
    'inputs_sha256':{p:sha(REPO/p) for p in tracked_inputs},
    'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','lightgbm','xgboost','scikit-learn']},
    'feature_count':len(model.FEATURE_COLS), 'feature_columns':model.FEATURE_COLS,
    'data_quality':quality,
    'weather_target_hours':int(weather.t_1h.between('2026-11-07','2026-11-11 23:00:00').sum()),
    'weather_target_missing':int(weather.loc[weather.t_1h.between('2026-11-07','2026-11-11 23:00:00')].isna().sum().sum()),
    'limits':[
        'Backtest de développement sur quatre fenêtres par rue ; pas un holdout indépendant.',
        'Météo historique observée, pas une archive de prévisions émises aux dates des plis : performance opérationnelle non certifiée.',
        'J−7 du pipeline désigne un lag prétraité (lissé et corrigé des jours spéciaux). Le J−7 non lissé est fourni séparément.',
        'Les statistiques globales concatènent les cibles observées des deux rues ; aucune observation des Champs-Élysées.',
        'La comparaison de versions réentraîne les deux codes sur les mêmes données et fenêtres actualisées.',
        'Les gains de précision ne sont pas des économies mesurées.',
    ],
}
(ROOT/'results'/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
score_items=[]
for arc in ARCS:
    q=summary[(summary.Arc==arc)&(summary.Target=='q')].iloc[0]
    k=summary[(summary.Arc==arc)&(summary.Target=='k')].iloc[0]
    cq=comparison[(comparison.Arc==arc)&(comparison.Target=='q')].iloc[0]
    ck=comparison[(comparison.Arc==arc)&(comparison.Target=='k')].iloc[0]
    score_items.append({'rue':arc,'q_j7':q.RMSE_J7_pretraite,'q_ancrage':q.RMSE_Ancrage,
                        'q_final':q.RMSE_Modele,'gain_q':q.Gain_vs_J7_pretraite_pct,
                        'k_j7':k.RMSE_J7_pretraite,'k_ancrage':k.RMSE_Ancrage,
                        'k_final':k.RMSE_Modele,'gain_k':k.Gain_vs_J7_pretraite_pct,
                        'q_previous':cq.RMSE_ancienne_version,'k_previous':ck.RMSE_ancienne_version,
                        'gain_previous_q':cq.Gain_nouvelle_vs_ancienne_pct,
                        'gain_previous_k':ck.Gain_nouvelle_vs_ancienne_pct,'n':int(q.N)})
(ROOT/'scores_source.json').write_text(json.dumps({
    'source':'results/scores_2_rues.csv', 'statut':'Recalcul après acquisition ; backtest de développement',
    'revision':current_revision,'baseline':'J−7 prétraité : lissage et remplacement des jours spéciaux',
    'scores':score_items,'global_2_rues':summary[summary.Arc.str.startswith('GLOBAL')].to_dict('records'),
},ensure_ascii=False,indent=2)+'\n')

# Rebuild figures using only the assessed streets and actual current model outputs.
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                     'font.size':12,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelcolor':'#1D2B32','text.color':'#1D2B32','axes.edgecolor':'#AAB7B1'})
colors={'Convention':'#007A53','Saint-Pères':'#315B7B'}
def save(fig,name):
    fig.savefig(ASSETS/name,dpi=160,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def style(ax):
    ax.grid(axis='y',color='#E5EAE7',lw=.6)
    ax.set_axisbelow(True)

fig,axes=plt.subplots(2,1,figsize=(13,6.1),sharex=True,layout='constrained')
for ax,target,label in zip(axes,['q','k'],['Débit (véh/h)','Occupation (%)']):
    for arc in ARCS:
        sub=forecast[forecast.arc.eq(arc)].sort_values('t_1h')
        ax.plot(sub.t_1h,sub['pred_'+target],label=arc,color=colors[arc],lw=1.7)
    ax.axvspan(pd.Timestamp('2026-11-11'),pd.Timestamp('2026-11-11 23:00'),color='#F3E8D4',alpha=.7)
    ax.set_ylabel(label)
    style(ax)
axes[0].legend(ncol=2,frameon=False)
axes[1].xaxis.set_major_locator(mdates.DayLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
axes[1].set_xlim(pd.Timestamp('2026-11-07'),pd.Timestamp('2026-11-11 23:00'))
save(fig,'current_forecast.png')

fig,axes=plt.subplots(2,2,figsize=(13,6.1),layout='constrained')
for i,arc in enumerate(ARCS):
    sub=oof[(oof.arc==arc)&(oof.fold==oof.fold.max())].sort_values('t_1h')
    for j,target in enumerate(['q','k']):
        ax=axes[i,j]
        ax.plot(sub.t_1h,sub['baseline_'+target],color='#A0AAA5',ls='--',label='J−7 prétraité')
        ax.plot(sub.t_1h,sub[target],color='#1D2B32',lw=1.6,label='Mesure')
        ax.plot(sub.t_1h,sub['pred_'+target],color=colors[arc],lw=1.4,label='Modèle actuel')
        ax.set_title(f'{arc} · {"débit (véh/h)" if target=="q" else "occupation (%)"}',fontsize=12)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        style(ax)
axes[0,0].legend(fontsize=9,ncol=3,frameon=False)
save(fig,'current_oof.png')

fig,axes=plt.subplots(2,2,figsize=(13,6.1),layout='constrained')
for j,arc in enumerate(ARCS):
    sub=train[train.arc==arc]
    for i,target in enumerate(['q','k']):
        ax=axes[i,j]
        for weekend,label,col in [(0,'Semaine',colors[arc]),(1,'Week-end','#A88959')]:
            grp=sub[sub.is_weekend==weekend].groupby('hour')[target]
            avg=grp.mean()
            ax.plot(avg.index,avg,label=label,color=col,lw=1.7)
        ax.set_title(f'{arc} · {"débit (véh/h)" if target=="q" else "occupation (%)"}',fontsize=12)
        ax.set_xticks(range(0,24,4)); style(ax)
axes[0,0].legend(frameon=False)
save(fig,'current_profiles.png')

fig,axes=plt.subplots(1,2,figsize=(13,5.3),layout='constrained')
for ax,arc in zip(axes,ARCS):
    sub=train[train.arc==arc]
    ax.scatter(sub.q,sub.k,s=7,alpha=.22,color=colors[arc],edgecolors='none')
    ax.set(xlabel='Débit (véh/h)',ylabel='Occupation (%)',title=arc)
    style(ax)
save(fig,'current_fundamental.png')

importance=pd.read_csv(REPO/'reports/feature_importances_by_arc_outcome.csv')
importance=importance[importance.Arc.isin(ARCS)].copy()
write_csv(importance,'importance_variables.csv')
fig,axes=plt.subplots(2,2,figsize=(13,6.5),layout='constrained')
for i,arc in enumerate(ARCS):
    for j,target in enumerate(['q','k']):
        sub=importance[(importance.Arc==arc)&(importance.Target_Code==target)].copy()
        sub['contribution']=sub.Impact_SHAP_Moyen_Unite * .5 * final_models[(arc,target)].residual_weight
        sub=sub.nlargest(6,'contribution').sort_values('contribution')
        ax=axes[i,j]
        ax.barh(sub.Feature,sub.contribution,color=colors[arc],height=.65)
        ax.tick_params(axis='y',labelsize=9)
        ax.set_title(f'{arc} · {target}',fontsize=12)
        ax.set_xlabel('Contribution LGBM (véh/h)' if target=='q' else 'Contribution LGBM (points de %)')
        ax.grid(axis='x',color='#E5EAE7'); ax.set_axisbelow(True)
save(fig,'current_importance.png')
print(summary.round(3).to_string(index=False),flush=True)
print('\nCOMPARAISON À DONNÉES ET FENÊTRES IDENTIQUES\n'+comparison.round(3).to_string(index=False),flush=True)
print('EVIDENCE_COMPLETE',flush=True)
