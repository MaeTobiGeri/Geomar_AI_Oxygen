# Open Data Questions

Things that would materially help the extreme-event framing (`CONCEPT.md`, `SPEC.md`) beyond what's
already available in `Documentation/data/`. None of these block starting `BUILD_PLAN.md` — they're
upgrades to fold in if they're realistically obtainable.

1. **Higher-frequency oxygen data.** The pipeline resamples everything to weekly (`SPEC.md` §4).
   Hypoxic events can develop and resolve on faster timescales driven by wind-driven advection
   (`reference/Research.txt`). If there's sub-weekly (daily, or the lander/buoy high-frequency data
   mentioned in `Research.txt` — the BIGO/FLUX/VIATOR 2020 deployment) data available for Boknis Eck
   or Eckernförde Bight, it could substantially sharpen both labeling (`SPEC.md` §6) and features
   (`SPEC.md` §5).
2. **Wind direction / a proper mixing or advection index.** `Research.txt` is explicit that
   wind-driven advection of low-oxygen Kiel Bight water is the dominant physical driver of hypoxia
   at this station — more so than local biology. The pipeline currently only pulls wind speed/
   direction from one DWD land station (Schönhagen), resolved into U/V components. Better: local
   sea-level pressure, or an actual current/advection proxy if one exists for the Eckernförde Bight
   / Kiel Bight boundary.
3. **A documented list of historical hypoxic/fish-kill events** at or near Boknis Eck, with dates —
   even an informal one. This would seed the held-out event-study set (`SPEC.md` §8, `BUILD_PLAN.md`
   Phase 4) far better than inferring events purely from threshold crossings in the raw series.
4. **Additional depths or a second station**, if available. `Research.txt` flags that models trained
   at one station generalize poorly to nearby sites — not asking to solve that here, but even a
   second depth level or nearby station for out-of-sample sanity-checking (does the model's notion
   of "extreme" hold up somewhere it wasn't trained?) would be valuable context.
5. **Chlorophyll-a and nutrient data past 2021/2023** (current chlorophyll file ends 2021, main
   nutrient/O2 series ends 2023) — if there's a more recent extract available from PANGAEA or GEOMAR
   internally, extending the record closer to the present would help both training data volume and
   give something current to actually run the rebuilt app against.
