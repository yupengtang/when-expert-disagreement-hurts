"""Render revised tables and vector figures from the audited CSVs only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent/'outputs'
NAMES = {
 'openai/gpt-4o-mini':'GPT-4o-mini', 'anthropic/claude-sonnet-4':'Claude-Sonnet-4',
 'google/gemini-2.0-flash-001':'Gemini-2.0-Flash', 'meta-llama/Llama-3.1-8B-Instruct':'Llama-3.1-8B',
 'mistralai/Mistral-7B-Instruct-v0.3':'Mistral-7B', 'openai/gpt-5.6-terra':'GPT-5.6-terra',
 'x-ai/grok-4.5':'Grok-4.5', 'google/gemini-3.6-flash':'Gemini-3.6-Flash',
 'anthropic/claude-opus-4.8':'Claude-Opus-4.8', 'google/gemini-2.5-flash':'Gemini-2.5-Flash'}
ORDER=list(NAMES)[:9]
NAMES.update({'meta-llama/llama-3.1-8b-instruct':'Llama-3.1-8B',
              'meta-llama/llama-3.3-70b-instruct':'Llama-3.3-70B'})
plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'paired-audit-v2',
                     'font.size':9,'axes.spines.top':False,'axes.spines.right':False})


def interval(row, metric):
    return f"${row[metric]:+.2f}$ [{row[metric+'_lo']:+.2f}, {row[metric+'_hi']:+.2f}]"


def table(name, caption, label, header, rows, columns):
    text=[r'\begin{table}[t]',r'\caption{'+caption+'}',r'\label{'+label+'}',
          r'\centering\small',r'\setlength{\tabcolsep}{3.5pt}',r'\begin{tabular}{'+columns+'}',
          r'\toprule',' & '.join(header)+r' \\',r'\midrule']
    text += [' & '.join(row)+r' \\' if isinstance(row,list) else row for row in rows]
    text += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    rendered = '\n'.join(text)+'\n'
    assert not any(ord(c) < 32 and c != '\n' for c in rendered), 'Unexpected TeX control character'
    (OUT/name).write_text(rendered)


def draw_effects(data, path, boundary):
    y=np.arange(len(data),dtype=float)
    y[boundary:]+=.7
    fig, axes=plt.subplots(1,2,figsize=(6.8,3.25),sharey=True)
    fig.subplots_adjust(left=.24,right=.98,bottom=.18,top=.88,wspace=.23)
    for ax,key,title,color in zip(axes,['P_R_pp','P_Acc_pp'],['Reversal effect','Accuracy effect'],['#245a81','#6c4d76']):
        ax.axvline(0,color='.5',linewidth=.8)
        ax.axhspan(boundary-.1,y[-1]+.6,color='#f3f6f8')
        ax.errorbar(data[key],y,xerr=[data[key]-data[key+'_lo'],data[key+'_hi']-data[key]],
                    fmt='o',color=color,capsize=2.5,markersize=4,linewidth=1)
        ax.set_yticks(y,[NAMES[x] for x in data.model_id],fontsize=8.8)
        ax.set_ylim(y[-1]+.6,-.7)
        ax.set_title(title,fontsize=10)
        ax.set_xlabel('Expert − anonymous (pp)',fontsize=9)
        ax.tick_params(axis='y',length=0)
        ax.grid(axis='x',alpha=.15)
        ax.set_axisbelow(True)
    for extension in ['pdf','svg']:
        metadata = {'Date': None} if extension == 'svg' else {'CreationDate': None, 'ModDate': None}
        fig.savefig(OUT/f'{path}.{extension}', metadata=metadata)
    plt.close(fig)


def main():
    all_results=pd.read_csv(OUT/'accuracy_decomposition.csv')
    pairs=all_results[all_results.intersection.isin(['A1/A3','A1_rat_clean/A3_rat_clean'])]
    core=pairs[pairs.cohort.isin(['main','controls_four'])].set_index('model_id').loc[ORDER].reset_index()
    rows=[]
    for i,(_,r) in enumerate(core.iterrows()):
        if i==5: rows.append(r'\midrule')
        rows.append([NAMES[r.model_id],str(r.n),interval(r,'P_R_pp'),interval(r,'P_Acc_pp')])
    table('source_effects.tex',r'Paired effects of the expert label on reversal and accuracy (pp), with 95\% item-bootstrap intervals (20,000 resamples), on items whose initial answers match across branches. The first five models use the main benchmark and the last four the 250-item pool. Both protocols are padded, so filler differences also enter these contrasts.',
          'tab:reversal',['Model','$N$',r'$P_R$ [95\% CI]',r'$P_{\rm Acc}$ [95\% CI]'],rows,'lrcc')
    clean_order=ORDER[:2]+['google/gemini-2.5-flash']+ORDER[5:]
    clean=pairs[pairs.cohort.isin(['clean_api','clean_four'])].set_index('model_id').loc[clean_order].reset_index()
    rows=[]
    for i,(_,r) in enumerate(clean.iterrows()):
        if i==3: rows.append(r'\midrule')
        rows.append([NAMES[r.model_id],str(r.n),interval(r,'P_R_pp'),interval(r,'P_Acc_pp')])
    table('clean_effects.tex',r'No-padding identical-rationale contrasts (pp): expert minus anonymous, with 95\% paired item-bootstrap intervals. The two branches of each pair were collected together; rows below the rule come from the September run. Coverage and document-cluster sensitivity are in Appendix~\ref{app:reanalysis}.',
          'tab:paired_api',['Model','$N$',r'$P_R$ [95\% CI]',r'$P_{\rm Acc}$ [95\% CI]'],rows,'lrcc')
    draw_effects(core,'source_accuracy_effects',5)
    draw_effects(clean,'clean_accuracy_effects',3)
    sensitivity=pd.read_csv(OUT/'evidence_coverage_sensitivity.csv')
    clusters=pd.read_csv(OUT/'document_cluster_sensitivity.csv')
    counts=pd.read_csv(OUT/'domain_sample_counts.csv')
    rows=[]
    for model in ORDER[:5]:
        for subset,short in [('historical','All'),('exclude_known_incomplete','Coverage exclusion'),('medicine_science','Medicine/Science')]:
            r=sensitivity[(sensitivity.model_id==model)&(sensitivity.cohort=='main')&(sensitivity.subset==subset)].iloc[0]
            c=clusters[(clusters.model_id==model)&(clusters.cohort=='main')&(clusters.subset==subset)].iloc[0]
            rows.append([NAMES[model] if subset=='historical' else '',short,str(r.n),interval(r,'P_R_pp'),interval(c,'P_Acc_pp')])
    table('coverage_sensitivity.tex',r'Main-audit sensitivity. $P_R$ intervals resample items; $P_{\rm Acc}$ intervals resample source documents within domains, with domain weights held fixed. Coverage exclusion drops the 48 Law items with incomplete excerpts before estimation; it does not certify the remaining labels.',
          'tab:coverage_sensitivity',['Model','Sample','$N$','$P_R$ [item CI]',r'$P_{\rm Acc}$ [document CI]'],rows,'llrcc')
    rows=[]
    for model in clean_order:
        cohort='clean_api' if model in clean_order[:3] else 'clean_four'
        for subset,short in [('historical','All'),('exclude_known_incomplete','Coverage exclusion')]:
            r=sensitivity[(sensitivity.model_id==model)&(sensitivity.cohort==cohort)&(sensitivity.subset==subset)].iloc[0]
            c=clusters[(clusters.model_id==model)&(clusters.cohort==cohort)&(clusters.subset==subset)].iloc[0]
            rows.append([NAMES[model] if subset=='historical' else '',short,str(r.n),interval(r,'P_R_pp'),interval(c,'P_Acc_pp')])
    table('clean_sensitivity.tex',r'No-padding rationale sensitivity. Coverage exclusion drops the follow-up items with incomplete excerpts from each model\textquotesingle s paired intersection. Item intervals and document-cluster intervals address different sampling questions.',
          'tab:clean_sensitivity',['Model','Sample','$N$','$P_R$ [item CI]',r'$P_{\rm Acc}$ [document CI]'],rows,'llrcc')
    a=pd.read_csv(OUT/'agreement_conformity_interaction.csv')
    rows=[]
    for _,r in a[a.cohort.isin(['controls_api','controls_open','controls_four'])].iterrows():
        rows.append([NAMES[r.model_id],str(r.n),interval(r,'I_R_pp'),interval(r,'I_F_pp')])
    table('conformity_interaction.tex',r'Four-condition paired interactions (pp). $I_R=P_{\rm disagree}-P_{\rm agree}$ is defined on reversal and $I_F=P_{\rm disagree}+P_{\rm agree}$ on source following; only $I_F$ compares the same outcome under disagreement and agreement. Item-bootstrap percentile intervals; padded suite.',
          'tab:conformity_interaction',['Model','$N$',r'$I_R$ [95\% CI]',r'$I_F$ [95\% CI]'],rows,'lrcc')
    card=sensitivity[(sensitivity.cohort=='clean_api')&(sensitivity.model_id=='openai/gpt-4o-mini')&(sensitivity.subset=='exclude_known_incomplete')].iloc[0]
    rows=[['Endpoint / collection',r'GPT-4o-mini / July 2026 (reported)'],
          ['Protocol / coverage',r'No padding, identical rationale; affected inputs excluded'],
          ['Paired items / initial accuracy',f'{card.n} / {100*card.a:.1f}\\%'],
          [r'Expert wrong-revision / correction rates',f'{100*card.h_A3:.1f}\\% / {100*card.b_A3:.1f}\\%'],
          [r'$P_R$ / $P_{\rm Acc}$ (pp)',f'{card.P_R_pp:+.2f} / {card.P_Acc_pp:+.2f}'],
          [r'$P_{\rm Acc}$ [95\% item CI] (pp)',interval(card,'P_Acc_pp')],
          [r'Expert versus stored Pass~1 (pp)',interval(card,'delta_acc_A3_pp')]]
    table('audit_card.tex',r'Example audit card: GPT-4o-mini on the no-padding identical-rationale suite (July 2026), after dropping items with incomplete excerpts ('+str(card.n)+r' of 249 paired items kept). Rationales were checked for answer direction only, and the delivered requests and exact generation times for this suite are not available. A card describes one measured setting; it does not certify others.',
          'tab:audit_card',['Field','Recorded result'],rows,r'p{.40\linewidth}p{.55\linewidth}')
    rows=[]
    for cohort in ['main','clean_api','clean_four']:
        for model in counts[(counts.cohort==cohort)&(counts.subset=='historical')].model_id.unique():
            d=counts[(counts.cohort==cohort)&(counts.subset=='historical')&(counts.model_id==model)]
            rows.append([NAMES[model],cohort.replace('_',r'\_')]+[f'{int(d[d.domain==domain].iloc[0].n)}/{int(d[d.domain==domain].iloc[0].n_documents)}' for domain in ['Medicine','Science','Law']])
    table('document_counts.tex',r'Paired questions / distinct source documents by domain. PubMed IDs, SciFact document IDs, and ContractNLI document filenames define the clusters. Cohorts are not pooled.',
          'tab:document_counts',['Model','Cohort','Medicine','Science','Law'],rows,'llccc')
    table('protocol_registry.tex',r'Protocol registry. Sample sizes are designed pools, not valid-pair counts. Reported dates lack complete per-request timestamps. No row represents a new evaluation on evidence-complete v2 inputs.',
          'tab:protocol_registry',['Cohort','Pool','Filler','Pass 1 / Pass 2'],[
          ['Main (Jan., reported)','1,989','Active','Same-run (reported)'],
          ['API/open controls (July, reported)','500','Active','Stored initial answers'],
          ['Four-model / $T=0$ (July, reported)','250','Active','Stored initial answers'],
          ['Clean API (July, reported)','250','None','Exact times unavailable'],
          ['Clean four-model (Sept.)','250','None','July / September'],
          ['Temporal GPT-4o-mini (July, reported)','224','Original','Same-run (reported)']], 'lrll')
    # Compact vector overview; the author's detailed PowerPoint remains untouched.
    fig,ax=plt.subplots(figsize=(7.2,1.85));ax.set_xlim(0,10);ax.set_ylim(0,3);ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.09',linewidth=.8,edgecolor=color,facecolor='#f8fafb'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9,color='#263442')
    box(.15,.75,2.5,1.3,'Fixed across branches\nQuestion + evidence\nShared initial answer','#607b8c')
    for y,label,color in [(2.12,'A₀  Neutral reflection','#87929d'),(1.12,'A₁  Anonymous disagreement','#245a81'),(.12,'A₃  Expert-labeled disagreement','#d55e00')]:
        box(3.55,y,3.2,.62,label,color)
        ax.add_patch(FancyArrowPatch((2.8,1.4),(3.4,y+.31),arrowstyle='->',mutation_scale=10,color='#82919b'))
        ax.add_patch(FancyArrowPatch((6.9,y+.31),(7.5,1.4),arrowstyle='->',mutation_scale=10,color='#82919b'))
    box(7.65,.65,2.15,1.5,'Paired source effects\nReversal: A₃ − A₁\nAccuracy: A₃ − A₁','#607b8c')
    fig.tight_layout(pad=.3)
    fig.savefig(OUT/'audit_flow.pdf');fig.savefig(OUT/'audit_flow.svg');plt.close(fig)
    print('Rendered audited tables and figures:',OUT)


if __name__=='__main__':
    main()
