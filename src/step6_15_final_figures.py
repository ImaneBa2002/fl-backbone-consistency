from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
IN_DIR = ROOT / 'step6_14_multiseed_evaluation_CKA'
OUT_DIR = ROOT / 'step6_15_final_figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERF = IN_DIR / 'test_performance_mean_std.csv'
PER_CLASS = IN_DIR / 'test_per_class_mean_std.csv'
CKA = IN_DIR / 'CKA_mean_std_3seeds.csv'
DELTA = IN_DIR / 'CKA_paired_delta_mean_std.csv'

METHODS = ['FedAvg', 'MOON']
LAYERS = ['Conv1','Conv2','Conv3','Conv4','Final']
ROUNDS = [1,2,5,10,15,20]
CLASSES = ['NORM','MI','STTC','CD','HYP']

print('='*80)
print('STEP 6.15 - FINAL CHAPTER FIGURES')
print('='*80)

for name, path in {'Performance':PERF,'Per-class':PER_CLASS,'CKA':CKA,'CKA delta':DELTA}.items():
    print(f'{name:<20}: {"FOUND" if path.exists() else "MISSING"}')
    if not path.exists():
        raise FileNotFoundError(path)

perf = pd.read_csv(PERF)
per_class = pd.read_csv(PER_CLASS)
cka = pd.read_csv(CKA)
delta = pd.read_csv(DELTA)

def savefig(stem):
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'{stem}.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUT_DIR / f'{stem}.svg', bbox_inches='tight')
    plt.close()
    print('Saved:', stem)

def metric_values(metric):
    means, stds = [], []
    for method in METHODS:
        row = perf[(perf.method==method) & (perf.metric==metric)]
        if len(row) != 1:
            raise RuntimeError(f'Bad row for {method} {metric}')
        means.append(float(row.iloc[0]['mean']))
        stds.append(float(row.iloc[0]['std']))
    return np.array(means), np.array(stds)

def performance_bar(metric, ylabel, title, stem):
    means, stds = metric_values(metric)
    x = np.arange(len(METHODS))
    plt.figure(figsize=(6.2,4.6))
    plt.bar(x, means, yerr=stds, capsize=6)
    plt.xticks(x, METHODS)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y', alpha=.25)
    top = max(means + stds)
    bottom = 0.0 if metric not in ['macro_auroc','exact_match'] else max(0.0, min(means)-0.08)
    plt.ylim(bottom, max(top*1.18, top+0.08))
    for i,(m,s) in enumerate(zip(means,stds)):
        plt.text(i, m+s+0.01, f'{m:.3f} ± {s:.3f}', ha='center', fontsize=9)
    savefig(stem)

performance_bar('macro_f1','Macro-F1','Final Test Macro-F1 Across Three Seeds','Fig01_MacroF1_mean_std')
performance_bar('macro_auroc','Macro-AUROC','Final Test Macro-AUROC Across Three Seeds','Fig02_MacroAUROC_mean_std')
performance_bar('loss','BCE Loss','Final Test BCE Loss Across Three Seeds','Fig03_BCE_Loss_mean_std')
performance_bar('exact_match','Exact Match','Final Test Exact Match Across Three Seeds','Fig04_ExactMatch_mean_std')

for mean_col,std_col,title,stem in [
    ('cc_mean','cc_std','Final Representation Client-Client CKA','Fig05_Final_CKA_ClientClient_trajectory'),
    ('lg_mean','lg_std','Final Representation Local-Global CKA','Fig06_Final_CKA_LocalGlobal_trajectory'),
]:
    plt.figure(figsize=(7,4.8))
    for method in METHODS:
        d = cka[(cka.method==method) & (cka.layer=='Final')].set_index('round').loc[ROUNDS].reset_index()
        plt.errorbar(d['round'], d[mean_col], yerr=d[std_col], marker='o', capsize=4, linewidth=2, label=method)
    plt.xlabel('Communication Round')
    plt.ylabel('Linear CKA')
    plt.title(title)
    plt.xticks(ROUNDS)
    plt.ylim(0,1.05)
    plt.grid(alpha=.25)
    plt.legend()
    savefig(stem)

def layerwise(mean_col,std_col,title,stem):
    x = np.arange(len(LAYERS)); width=.36
    plt.figure(figsize=(7.2,4.8))
    for i,method in enumerate(METHODS):
        d = cka[(cka.method==method) & (cka['round']==20)].copy()
        d['layer'] = pd.Categorical(d['layer'], categories=LAYERS, ordered=True)
        d = d.sort_values('layer')
        plt.bar(x+(i-.5)*width, d[mean_col], width, yerr=d[std_col], capsize=4, label=method)
    plt.xticks(x,LAYERS)
    plt.ylabel('Linear CKA')
    plt.title(title)
    plt.ylim(0,1.05)
    plt.grid(axis='y', alpha=.25)
    plt.legend()
    savefig(stem)

layerwise('cc_mean','cc_std','Round-20 Layer-wise Client-Client Backbone Similarity','Fig07_Round20_Layerwise_ClientClient_CKA')
layerwise('lg_mean','lg_std','Round-20 Layer-wise Local-Global Backbone Similarity','Fig08_Round20_Layerwise_LocalGlobal_CKA')

def delta_plot(mean_col,std_col,title,stem):
    d = delta[delta['round']==20].copy()
    d['layer'] = pd.Categorical(d['layer'], categories=LAYERS, ordered=True)
    d = d.sort_values('layer')
    x = np.arange(len(LAYERS))
    plt.figure(figsize=(7.2,4.8))
    plt.bar(x, d[mean_col], yerr=d[std_col], capsize=5)
    plt.axhline(0, linewidth=1)
    plt.xticks(x,LAYERS)
    plt.ylabel('Δ CKA (MOON − FedAvg)')
    plt.title(title)
    plt.grid(axis='y', alpha=.25)
    for i,(m,s) in enumerate(zip(d[mean_col], d[std_col])):
        plt.text(i, m+s+0.01, f'{m:+.3f}', ha='center', fontsize=9)
    savefig(stem)

delta_plot('delta_cc_mean','delta_cc_std','Round-20 MOON Gain in Client-Client CKA','Fig09_Round20_Delta_ClientClient_CKA')
delta_plot('delta_lg_mean','delta_lg_std','Round-20 MOON Gain in Local-Global CKA','Fig10_Round20_Delta_LocalGlobal_CKA')

for metric,ylabel,title,stem in [
    ('f1','F1 Score','Per-Class Test F1 Across Three Seeds','Fig11_PerClass_F1_mean_std'),
    ('auroc','AUROC','Per-Class Test AUROC Across Three Seeds','Fig12_PerClass_AUROC_mean_std'),
]:
    x = np.arange(len(CLASSES)); width=.36
    plt.figure(figsize=(8,5))
    for i,method in enumerate(METHODS):
        d = per_class[(per_class.method==method) & (per_class.metric==metric)].copy()
        d['class'] = pd.Categorical(d['class'], categories=CLASSES, ordered=True)
        d = d.sort_values('class')
        plt.bar(x+(i-.5)*width, d['mean'], width, yerr=d['std'], capsize=4, label=method)
    plt.xticks(x,CLASSES)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0,1.05)
    plt.grid(axis='y', alpha=.25)
    plt.legend()
    savefig(stem)

pngs = sorted(OUT_DIR.glob('*.png'))
svgs = sorted(OUT_DIR.glob('*.svg'))
print('\n'+'='*80)
print('STEP 6.15 COMPLETED SUCCESSFULLY')
print('='*80)
print('PNG figures :', len(pngs))
print('SVG figures :', len(svgs))
print('Output      :', OUT_DIR)
if len(pngs)!=12 or len(svgs)!=12:
    raise RuntimeError('Expected 12 PNG and 12 SVG figures.')
print('All figures use mean ± sample standard deviation across 3 seeds.')
print('='*80)
