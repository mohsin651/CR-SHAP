"""Document a technical stop without reporting partial results as replication."""
import json
from pathlib import Path

from experiment2.io import save_json, sha256, verify_snapshot


def main():
    root = Path('results/experiment_3')
    screening = Path('data/sugarcrepe_screening_exp3')
    preflight = json.loads((screening/'segmentation_preflight.json').read_text())
    rows = json.loads((root/'rows.json').read_text())
    integrity = json.loads((root/'protected_experiments.json').read_text())
    verify_snapshot(integrity)
    save_json(root/'segmentation_preflight.json',preflight)
    save_json(root/'run_status.json',{'status':'stopped_protocol_incompatibility','completed_instances':len(rows),
        'selected_instances':700,'failure':preflight['failures'],'partial_results_not_confirmatory':True,
        'reason':'Six SLIC regions permit only 64 distinct binary coalitions; frozen sampler requires 128 unique coalitions.',
        'parameters_changed':False,'sample_replacements':0,'earlier_experiments_unchanged':True})
    save_json(root/'integrity_check.json',{'all_unchanged':True,'n_files':len(integrity)})
    save_json(root/'implementation_hashes.json',{str(p):sha256(p) for p in sorted(Path('experiment3').glob('*')) if p.is_file()})
    (root/'requested_protocol.md').write_bytes((screening/'requested_protocol.md').read_bytes())
    text = f'''# Experiment 3: implementation and technical stop

**The confirmatory experiment is incomplete. No confirmatory conclusion is available.**

The frozen segmentation of selected instance `swap_att:39` yields six regions.
There are only 2^6 = 64 distinct binary coalitions. The unchanged validated
sampler requires 128 unique coalitions and explicitly rejects this case.
An outcome-independent check of all 700 selected instances identified this
single incompatibility. The run was stopped when the check finished, before
attempting that instance. No segmentation, coalition setting, sample ID,
caption, or method was changed, and no replacement was made.

## Completed work

- Screening recomputed CLIP embeddings and reproduced 7,511 total, 5,745
  correctly ranked, 1,766 incorrectly ranked, zero ties (76.49% accuracy).
- Seed 2026 froze 700 instances, 100/category. All 100 Experiment 2 IDs were
  excluded; every category had enough new examples and 100 unique images.
- The sample has 597 unique images, 94 repeated images across categories,
  and a maximum of three instances per image.
- The unchanged frozen model/processor, coalition and constrained SHAP
  functions, global-mean masking, and positive-only Telea evaluation are reused.
- The added signed-ranking evaluation records first whole-region crossings at
  10–50%, full cumulative masks, margins, preferences, rank flips, and normalized
  observed MatchedArea-PDAUC-50 without an extrapolated tail.
- Statistical code implements paired two-sided Wilcoxon, 10,000-resample
  instance and image-cluster bootstraps with seed 42, budget comparisons,
  category summaries, paired outcome counts, and exact McNemar tests.
- Three new unit tests passed. Seven pilot instances, one per category, passed
  all checks and independent saved-artifact validation.
- {len(rows)} full-run instances were completed and retained in `results.csv`
  and their example directories. A directory for the interrupted next example
  may be incomplete. These ordered partial results are diagnostic only and
  must not be presented as the confirmatory sample.
- All {len(integrity):,} protected Experiment 1/2 files remain unchanged.

## Required decision before continuing

The current requirements cannot all be satisfied for this frozen sample.
Changing SLIC, allowing repeated coalitions, or omitting/replacing the affected
instance would amend the specified protocol. No amendment has been applied.
One possible amendment is a documented, outcome-independent technical exclusion
of this one instance without replacement, yielding 699 instances and 99 in
`swap_att`; this needs your explicit authorization because the request prohibits
removing selected examples. An alternative is to keep this run stopped.

## Saved evidence and reproducibility

`segmentation_preflight.json` contains all 700 region counts and the single
failure. `run_status.json` records the stop. `selection_report.json` and
`selected_instances.json` retain the original sample. `requested_protocol.md`
preserves the supplied instructions, which end at “exact McNem” in section 30;
the implemented test interprets this as exact McNemar. `verification.json`
documents independent checks of completed raw artifacts. Source and commands
are in `experiment3/README.md`; future runs reject the failed preflight.
'''
    (root/'report.md').write_text(text,encoding='utf-8')
    print(f'Recorded technical stop; {len(rows)} completed instances preserved')


if __name__=='__main__':
    main()
