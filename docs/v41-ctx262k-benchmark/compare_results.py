#!/usr/bin/env python3
"""Generate the context comparison directly from verified raw result rows."""
from pathlib import Path
import collections
import csv
import datetime
import hashlib
import json
import statistics as st

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'v41-local-benchmark'
old=json.loads((BASE/'results_ab.json').read_text())
new=json.loads((HERE/'results_ab.json').read_text())
assert old['config']['ctx']==8192 and new['config']['ctx']==262144
cells=[(x,y) for x in (1024,2048,4096) for y in (128,256)]
comparison=[]
for arm in ('mtp','nomtp'):
    sets=[]
    for data in (old,new):
        a=data['arms'][arm]
        assert len(a['warmup'])==2 and len(a['results'])==18 and a['engine']['exit_code']==0
        assert collections.Counter((r['prefill_tokens'],r['max_new_tokens']) for r in a['results'])=={k:3 for k in cells}
        for r in a['results']:
            assert r['accepted'] and not r['problems'] and r['done_reused']==0
            assert r['done_total_prompt']==r['prefill_tokens']
            assert r['done_generated']==r['max_new_tokens']
            content=Path(r['output_ids_file']).read_bytes()
            assert len(content)==r['max_new_tokens']*4
            assert hashlib.sha256(content).hexdigest()==r['output_ids_sha256']
        sets.append(a['results'])
    assert {r['case']:r['ids_sha256'] for r in sets[0]}=={r['case']:r['ids_sha256'] for r in sets[1]}
    for pp,out in cells:
        row={'arm':arm,'prefill':pp,'output':out,'n':3}
        for metric in ('prompt_tok_s','decode_tok_s','wall_seconds_including_request','prompt_expert_file','decode_expert_file'):
            for name,rows in zip(('8k','262k'),sets):
                vals=[r[metric] for r in rows if (r['prefill_tokens'],r['max_new_tokens'])==(pp,out)]
                row[f'{metric}_{name}_mean']=st.mean(vals)
                row[f'{metric}_{name}_sd']=st.stdev(vals)
            row[f'{metric}_delta_pct']=100*(row[f'{metric}_262k_mean']/row[f'{metric}_8k_mean']-1)
        comparison.append(row)

lines=['DeepSeek V4.1 Flash GSQ: Kontext 8192 gegen 262144',
       'Generiert aus results_ab.json beider Laeufe; gleiche GPU, Engine, Fixtures und Arena.',
       '', 'Methode: Ein persistenter nativer --serve-Prozess pro Arm, gleiche Reihenfolge,',
       'zwei ausgeschlossene Warm-ups und drei Messungen je Zelle. MTP an und aus.',
       'Alle 36 neuen Messungen akzeptiert, exakte Prompt-/Output-Laengen; tatsaechlich',
       '0 wiederverwendete Tokens in jeder gemessenen Anfrage. Output-Dateien/Hashes geprueft.',
       '60 GiB Expert-RAM-Arena, MemoryMax 110 GiB, MemorySwapMax 0, CPU-Slot GPU PCI 00000000:2B:00.0.',
       'Tok/s sind Engine-Phasenraten; Modellladezeit nicht enthalten. SD ist Stichproben-SD.',
       'Ungecacht bezieht sich auf Prompt/KV-Reuse, nicht auf kalten Linux-Datei-/Expert-Cache.',
       'Kein HTTP-Frontend im Messpfad, entsprechend dem vorherigen nativen Server-Benchmark.', '']
for arm in ('mtp','nomtp'):
    lines += [('MTP AN' if arm=='mtp' else 'MTP AUS'),
              'Prompt Out | Prefill 8k +/-SD -> 262k +/-SD (Delta) | Decode 8k +/-SD -> 262k +/-SD (Delta) | Wall 8k -> 262k s']
    for r in [x for x in comparison if x['arm']==arm]:
        parts=[]
        for m in ('prompt_tok_s','decode_tok_s'):
            parts.append(f"{r[m+'_8k_mean']:.2f} +/-{r[m+'_8k_sd']:.2f} -> {r[m+'_262k_mean']:.2f} +/-{r[m+'_262k_sd']:.2f} ({r[m+'_delta_pct']:+.2f}%)")
        lines.append(f"{r['prefill']:4} {r['output']:3} | "+' | '.join(parts)+f" | {r['wall_seconds_including_request_8k_mean']:.3f} -> {r['wall_seconds_including_request_262k_mean']:.3f}")
    lines.append('')
    for m in ('prompt_tok_s','decode_tok_s','wall_seconds_including_request'):
        values=[st.mean(r[m] for r in d['arms'][arm]['results']) for d in (old,new)]
        lines.append(f"Arithmetisches Mittel ueber 18 Anfragen, {m}: {values[0]:.4f} -> {values[1]:.4f} ({100*(values[1]/values[0]-1):+.2f}%)")
    lines.append('')

resources=list(csv.DictReader((HERE/'resources_ab.csv').open()))
peak={arm:max(int(card.split(',')[2]) for r in resources if r['arm']==arm for card in r['gpu_snapshot'].split('|') if '2B:00.0' in card) for arm in ('mtp','nomtp')}
lines += ['Einordnung:',
          '- Prefill im kurzen Grid praktisch unveraendert. Decode-Mittel etwas niedriger;',
          '  dies ist ein deskriptiver Vorher/Nachher-Vergleich, kein Signifikanznachweis.',
          '- Das Kontextlimit reserviert nicht den gesamten freien VRAM als KV. Im Log:',
          '  nach dem Dense-Load 16.24 GiB frei bei 8k, 15.49 GiB bei 262k.',
          '- Expert-Cache/DSpark werden lazy initialisiert. 8 GiB direkt nach Start ist',
          '  daher kein repraesentativer Verbrauch waehrend Generierung.',
          f'- Gemessene GPU1-Spitzen aus {len(resources)} Ressourcen-Samples: MTP an {peak["mtp"]} MiB, aus {peak["nomtp"]} MiB.',
          f'- Host MemAvailable Minimum {min(float(r["mem_available_gib"]) for r in resources):.3f} GiB.',
          '- Arena blieb laut beiden Engine-Logs bei 60 GiB / 6708 Experten.',
          '- Alle Benchmark-Engine-Prozesse endeten mit Exit 0; GPU1 kehrte zwischen/nach Armen',
          '  auf 333 MiB / 23785 MiB frei zurueck. Dashboard DS41 danach wiederhergestellt.',
          '- Keine Aussage ueber Geschwindigkeit/Korrektheit bei 262k tatsaechlicher Prompt-Tiefe:',
          '  hier wurden wie beauftragt nur 1024/2048/4096 Prompt-Tokens gemessen.',
          '- Beide Laeufe verwenden dieselbe Armfolge: MTP an, dann aus.',
          '  Der zweite Arm erbt Linux-Pagecache. Laeufe erfolgten zu unterschiedlichen Tageszeiten;',
          '  Hintergrundlast und adaptive Expert-Residency koennen kleine Deltas mitverursachen.',
          '', 'Artefakte:', str(HERE/'results_ab.json'),str(HERE/'comparison.csv'),str(HERE/'campaign.json')]
(HERE/'COMPARISON.txt').write_text('\n'.join(lines)+'\n')
(HERE/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
with (HERE/'comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
print('\n'.join(lines))
print('MTP_DECODE_GAIN_262K_PCT',100*(st.mean(r['decode_tok_s'] for r in new['arms']['mtp']['results'])/st.mean(r['decode_tok_s'] for r in new['arms']['nomtp']['results'])-1))
