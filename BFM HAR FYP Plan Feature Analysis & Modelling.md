# BFM Data Collection Plan

Oct 11, 2026 · @Jin Wei

This doc covers how the dataset is collected: the checklist, the protocol, the spot-swap pilot and what the existing standing and walking sessions show. The research side of my FYP (gap, research questions, features, models, evaluation and literature) is in FYP Research Plan: Feature Analysis & Modelling.

## Roadmap to the main collection

&#91;embedded content: Roadmap · 5 stages, 1 decision\]

Each stage links to the section that explains it:

1. **Re-collect Student 3's 5 standing sessions.** Her sessions sit far from everyone else's: [Evidence from the existing standing data](#mr58v43ry63.20630). Also on the [Data collection checklist](#mr58v43ry63.23991).
2. **Run the spot-swap pilot.** *Done 13 Oct: "in between" → shared spots P1–P3; see [Pilot results (13 Oct)](#pilot-results-13-oct).* Design, analysis and decision rule: [Spot-swap pilot](#mr58v43ry63.16444). Why it is needed: [Why shared spots, and a pilot to check it first](#mr58v43ry63.14068).
   - *Lying pilot done 14 Oct: lying and sitting can't be told apart by motion; see [Lying pilot results (14 Oct)](#lying-pilot-results-14-oct).*
   - *Agreed with the lecturer 16 Oct: lying is replaced by bending (bowing), and all activities except walking use shared spots P1–P3; see [Restart test and decisions (16 Oct)](#restart-test-and-decisions-16-oct).*
   - *Bending vs waving check done 20 Oct: with continuous bending and shoulder-level waving, the current features separate them (98–100% of windows, across subjects); see [Bending and waving check (20 Oct)](#bending-and-waving-check-20-oct).*
   - Flipped or in between → shared spots P1–P3, as laid out in [Data collection protocol](#mr58v43ry63.3610).
   - Mostly correct → keep fixed spots and report the pilot as evidence.
   - About 50/50 → fix stability first (daily empty-room baseline, normalisation), then repeat. Background: [Consistency across collection days](#mr58v43ry63.25229).
3. **Agree with the co-senior and lecturer**, since the dataset is shared: [Open decisions for senior and lecturer](#mr58v43ry63.12153).
4. **Main data collection**, following the [Data collection checklist](#mr58v43ry63.23991) every day and the [Data collection protocol](#mr58v43ry63.3610).
5. **Feature analysis and modelling** in FYP Research Plan: Feature Analysis & Modelling.

## Data collection checklist

**Goal:** collect data that can honestly answer "Can we tell these 5 activities apart for a person the model has never seen?" The rules below stop that answer from being faked by recording time, location or setup changes.

**Every collection session**

- [ ] Router and Pi untouched; same position every day (one reference photo).
- [ ] Rotate activity order within each subject (e.g. stand, sit, walk, bend, wave, then a different order next round), never all rounds of one activity in a row.
- [ ] Router fixed to channel 149, 80 MHz (`uci show wireless.radio0`); never left on `auto`.
- [ ] Router power on a socket nobody unplugs. Every router restart starts a new **run**: log the time and record a 2-minute empty room before the next session.
- [ ] 2-minute empty-room recording at the start of each collection day (the first run of the day).
- [ ] Every session has 234 subcarriers (a drop to 40 MHz gives 108; re-record those). Before each session, check the router still sends at 80 MHz (`iw dev phy0-ap0 station dump | grep 'tx bitrate'`); see [Bandwidth drops to 40 MHz (20 Oct)](#bandwidth-drops-to-40-mhz-20-oct).
- [ ] Everything around the subject the same each session: team members in the same places, door in the same state.

**Once, before the main collection**

- [ ] Re-collect Student 3's 5 standing sessions.
- [x] Run the 1-hour spot-swap pilot. Flipped answers → use shared spots P1–P3; correct answers → keep fixed spots, with the pilot as evidence. *13 Oct: in between (6/8) → P1–P3.*
- [x] Lying pilot (14 Oct): can lying, sitting and an empty room be told apart? *Motion features: no. Static levels: weakly, within one day. See [Lying pilot results (14 Oct)](#lying-pilot-results-14-oct).*
- [x] ~~Repeat a few lying and sitting sessions on another day to check whether the static lying/sitting signal holds across days.~~ *Dropped 16 Oct: lying replaced by bending.*
- [x] Restart test (16 Oct): does a router restart alone change the static signal? *Yes, even on a fixed channel; see [Restart test and decisions (16 Oct)](#restart-test-and-decisions-16-oct).*
- [ ] Fix the Pi, which didn't reconnect after the 16 Oct restart, and check why the signal keeps weakening (−23 dBm on 6 Oct, −42 dBm on 16 Oct).
- [ ] Door test (optional): does the open door weaken standing at B? *20 Oct sitting check: no door effect visible, but sitting can't show one; see [Door check with sitting (20 Oct)](#door-check-with-sitting-20-oct).*
- [x] Stop the 80 → 40 MHz drops (router rate-control setting). *Applied 20 Oct, persisted in `/etc/rc.local`; confirm after the next reboot;* see [Bandwidth drops to 40 MHz (20 Oct)](#bandwidth-drops-to-40-mhz-20-oct).
- [x] Agree the bending and waving instructions and record them in the protocol. *20 Oct: continuous bending, waving at shoulder level; see [Bending and waving check (20 Oct)](#bending-and-waving-check-20-oct).*
- [ ] Re-record Student 3's bending and waving with the new instructions (20 Oct sessions used paused bows and two-handed overhead waving).
- [ ] Checked: the router sounds with 2 transmit antennas (Nr = 2 in all 63 repo pcaps), so only `phi11` and `psi21` exist and no extra angles are available. Confirmed on an October Pi pcap (1,217 frames: 1 stream, 2 antennas). Keep all raw pcaps (`bfm_pcap/`) so data can be re-extracted later without re-collecting.
- [ ] Decide spots with the lecturer, then collect the remaining sessions.

**Not now:** open-space set and the quantisation limit (a finding to report, not something to fix before collecting).

## Data collection protocol

All activities except walking are done at shared spots (P1, P2, P3), so location stops predicting the label. If each activity has its own fixed spot (e.g. chair = sitting), a model can score well by recognising where the person is rather than what they are doing. *Confirmed with the lecturer on 16 Oct, based on the pilot results.*

&#91;embedded content: Room layout · router + Pi pair and shared spots P1–P3\]

Standing, sitting, bending and waving are each recorded at P1, P2 and P3, so no spot belongs to one activity. Walking keeps its existing path.

**Activities:** standing, walking, sitting, bending, waving. Standing and sitting are held still for the whole 2-minute session. Target: 10 subjects × 5 activities × 5 sessions = 250 sessions.

**Bending (bowing)** replaced lying on 16 Oct. Lying could not be told apart from sitting or an empty room by motion, and its static signal resets with every router restart. Bending is a movement, so the motion features can see it the way they see walking and waving.

- Continuous bows for the whole 2 minutes: down and straight back up, no pause between bows. A held bow would be a still posture again.
- Steady, comfortable pace and the same depth each time.
- Facing the same direction as standing at that spot.

**Waving:** wave continuously at shoulder level for the whole 2 minutes, facing the same direction as standing.

*Instructions changed 20 Oct, after the first trial (bows with a 2 s pause, two-handed waving overhead) gave sessions that were hard to tell apart, with pauses that look like standing; see [Bending and waving check (20 Oct)](#bending-and-waving-check-20-oct).*

**Spots**

- **P1** = the current standing spot, so existing standing data fits the layout.
- **P2** = farther from the device pair, on roughly the same line.
- **P3** = off to one side.
- Exact distances: set by a short pilot. Pick the farthest spot where waving is still clearly visible in the raw BFM signal.
- Mark spots with floor tape. Bring a portable chair to each spot.

**Session allocation per subject (5 sessions per activity)**

| Activity | Spots | Sessions per subject |
| --- | --- | --- |
| Standing | P1, P2, P3 | 5 (subjects 1–5: existing 5 at P1, plus 2 at P2 and 2 at P3) |
| Sitting | P1, P2, P3 (portable chair) | 5 |
| Bending | P1, P2, P3 | 5 |
| Waving | P1, P2, P3 | 5 |
| Walking | Existing path | 5 (unchanged) |

**Rotation of the 5 sessions across spots** (keeps spot counts balanced over 10 subjects: about 17 / 17 / 16 per activity)

| Subject group | P1 | P2 | P3 |
| --- | --- | --- | --- |
| S1, S4, S7, S10 | 2 | 2 | 1 |
| S2, S5, S8 | 2 | 1 | 2 |
| S3, S6, S9 | 1 | 2 | 2 |

**Existing data (5 subjects, standing and walking only)**

- Standing: kept as P1 sessions. Add 2 sessions at P2 and 2 at P3 for each of subjects 1–5. If they can't return, take the extra standing coverage from subjects 6–10 and say so in the report.
- Walking: unchanged. Note in the report that the walking path was fixed.
- Sitting, bending and waving: not yet collected, so they follow the new protocol from the start for all 10 subjects.
- Lying sessions from the 14 and 16 Oct pilots are kept for the report but are not part of the dataset.

**Extra conditions**

- **Empty-room baseline per run:** a 2-minute session with nobody in the room at the start of each day and after every router restart. Each restart changes the static signal, so each run needs its own reference.
- **Logging:** record router–Pi spacing, height and orientation once, with a photo. Note the date, spot and run for every session (run = day plus restart count, e.g. `20261020-R1`). Note every router or Pi restart and why it happened.

**Every class in every run:** a run must not hold only one activity, or the model can learn the run instead of the activity. If a restart happens mid-subject, carry on rotating activities in the new run.

**Activity order:** interleave or rotate activities within each subject's collection (e.g. standing R1, sitting R1, bending R1, waving R1, walking R1, then R2 in a different order), instead of recording all rounds of one activity in a row. The existing standing data shows time drift is picked up by both feature sets (see the evidence below).

**Open-space set (if time allows)**

- 3 subjects who are also in the room dataset.
- All 5 activities, fewer sessions each (e.g. 2).
- Same router–Pi geometry as in the room, plus an empty-space baseline session.

## Why shared spots, and a pilot to check it first

With fixed spots, posture and location always change together, so a model may learn a map of the room instead of postures. A pilot of about one hour per person shows whether this happens before the plan changes.

**Why it can happen**

- For a still posture, the signal is mostly the static reflection of the body, plus breathing.
- That reflection depends on **where the body is** (distance and angle to the router–Pi pair) and **its shape** (standing, sitting, bent over).
- Moving 1 m changes the reflected path by about 2 m (\~38 wavelengths at 5 GHz), so location may change the signal more than posture does.
- With fixed spots, training data never separates the two, and a model uses whichever difference is easiest. This is the same failure as an image classifier that "recognises cows" by detecting grass.

**What goes wrong if the model learns location**

- Standing beside the chair is likely predicted as sitting.
- Sitting at a spot normally used for standing is likely predicted as standing; sitting at a new spot gives unpredictable answers.
- Leave-one-subject-out testing does not catch it: the held-out subject uses the same chair and spots, so the shortcut still works and accuracy looks high.
- In a real home, with different chairs and furniture moved around, recognition of still postures fails.

**Props rule:** keep the chair in place for every session at a spot, including standing, bending and walking. If a prop is only present for one posture, its reflection becomes a shortcut of its own (an empty chair could be predicted as sitting).

**What shared spots do and do not fix**

| Test situation | Trained with fixed spots | Trained with P1–P3 shared |
| --- | --- | --- |
| Sit on the usual chair, new subject | Correct, possibly for the wrong reason | Correct |
| Stand beside the chair | Likely "sitting" | More likely "standing" |
| Sit at a spot used for standing | Likely "standing" | More likely "sitting" |
| Sit at a new spot (leave-one-position-out) | Cannot be tested | Testable |

Shared spots remove the shortcut but do not guarantee generalisation: with 3 spots a model could still memorise 9 spot–posture patterns, which leave-one-position-out testing exposes. Accuracy is expected to be lower than with fixed spots, because fixed-spot accuracy is partly inflated by the shortcut. Walking and waving are less exposed, since their motion is a strong signal of its own.

### Spot-swap pilot

Record standing and sitting at their usual spots, then swap them. Train only on the usual sessions and test on the swapped ones.

| Session type | Spot | Posture | Sessions (2 min each) | Used for |
| --- | --- | --- | --- | --- |
| Usual | A (current standing spot) | Standing | 3 | Training |
| Usual | B (chair spot) | Sitting | 3 | Training |
| Swapped | B | Standing, beside the chair | 2 | Testing |
| Swapped | A | Sitting, on a second chair | 2 | Testing |

- 1–2 team members, about 20 minutes of recording each.
- Both chairs stay in place for all 10 sessions (props rule).
- Same router–Pi geometry and capture settings as the main dataset.

**Analysis**

The current real-time features (`rtbfm.window_features`, 43 per 5-s window) remove each subcarrier's mean and linear drift before computing anything. They describe motion only, so they are largely blind to location, and may barely separate two still postures. The pilot therefore compares two feature sets:

| Feature set | What it contains | Question it answers |
| --- | --- | --- |
| A. Current `rtbfm` features | 15 fluctuation-size (magnitude std, magnitude and phase absolute difference), 10 fluctuation-shape (roughness), 15 band-power fractions (0–5 Hz), 3 subcarrier-correlation | Can the real-time features separate still postures at all? Do the fluctuation-size features leak distance? |
| B. Static-level features | Per window: mean log Ratio\_Mag and circular-mean Ratio\_Phase per subcarrier, compressed by PCA (5–10 components, fitted on training sessions only) | Is there a location shortcut in the part of the signal that `rtbfm` discards? |
| A + B (optional) | Both | Does adding static information help or hurt on swapped sessions? |

1. **Sanity check:** leave-one-session-out among the 6 usual sessions. A feature set that cannot separate standing at A from sitting at B here gives a meaningless swap result.
2. **Swap test:** train on the 6 usual sessions, predict the 4 swapped sessions. Score **per session** (fraction of windows predicted as each class), because 5-s windows with a 0.5-s hop overlap 90%.
3. **Window length:** run 5 s and about 10–20 s, since 5 s holds only 1–1.5 breaths.
4. **No-model check (R):** session-level means of the static features; compare the spot difference with the posture difference.

**Changes needed in `train_rt_bfm.py`** (done 13 Oct: `--pilot` mode)

- Add `"sitting"` to `CLASSES` and `parse_name`.
- Put the spot in the session description when collecting (e.g. `spotA`, `spotB`) so it appears in the filename, and parse it.
- Replace leave-one-person-out with the fixed usual → swapped split; keep XGBoost (or Logistic Regression) as the classifier.
- Add a static-feature function next to `rtbfm.window_features`, using the same 10 Hz grid and gap rule.

**Decision rule (agree before looking at results)**

| Swapped-session result | Meaning | Action |
| --- | --- | --- |
| Mostly flipped (standing at B → "sitting", sitting at A → "standing") | Model learned location | Adopt P1–P3 for all still postures |
| Mostly correct (e.g. ≥ 80%, threshold set in advance) | Features capture posture | Keep fixed spots; report the pilot as evidence |
| About 50/50 | Features mainly identify individual recordings (session drift) | Fix stability first (empty-room baseline, normalisation), then repeat |
| In between | Both matter | Adopt P1–P3, at least for standing and sitting |

With 1–2 people the pilot is indicative, not conclusive, but a clear flip is strong evidence. The same session can also try waving at a far spot, to help set the P2/P3 distances.

### Pilot results (13 Oct)

**In short**

- We recorded 20 two-minute sessions (Students 1 and 2, 10 each) plus two empty-room recordings. Sessions were interleaved, so a slow drift over time could not fake the result.
- The model got **6 of the 8 swapped sessions right (75%)**. That is below the 80% pass mark but far from a full flip, so the result is **"in between" → use shared spots P1–P3** for still postures.
- What the model actually detects is **how much the body sways the WiFi signal**. Standing at A sways it a lot. Standing at B sways it much less, so 2 of the 4 standing-at-B sessions were mistaken for sitting.
- **Sitting barely moves the signal.** It looks almost the same as an empty room.
- **The signal's static level doesn't change when a person enters the room**, so it can't tell where someone is or what posture they hold.
- **Next:** the lying sessions (14 Oct) need to check whether lying, sitting and an empty room can be told apart at all.

**Setup**

- **Spot A** is beside the wall; **spot B** is beside the door, which stayed open for every session. Both spots are the same perpendicular distance from the router–Pi pair and at least 1 m apart sideways. Both chairs stayed in place for every session.
- **Order:** every test session was recorded between one standing and one sitting training session. Student 2 went forward (10:17–10:44) and Student 1 in reverse (10:50–11:18).
- **Empty-room recordings** at 09:28 and 09:58.
- **Data quality:** every session is 120 s at 9.7–9.8 Hz on a single link, with no bursts of motion, and every raw pcap is archived. A duplicate `Student 2 … Spot A R1`, recorded without the chairs, was deleted and not used.

**Swap test** (train on standing @A + sitting @B, test on standing @B + sitting @A; sessions correct by majority of windows)

| Feature set | 5 s, XGBoost | 5 s, LR | 15 s, XGBoost | 15 s, LR | Sanity check (usual sessions) |
| --- | --- | --- | --- | --- | --- |
| A. Current `rtbfm` features | 6/8 | 6/8 | 6/8 | 6/8 | 12/12 in every setting |
| B. Static-level features | 4/8 | 5/8 | 3/8 | 5/8 | 8–10/12: unreliable, so its swap result doesn't count |
| A + B | 4/8 | 8/8 | 6/8 | 6/8 | 10–12/12 |

- **Feature set A** gives the same answer in every setting. The same two standing-at-B sessions (Student 1 R1, Student 2 R2) are always called sitting. One sitting-at-A session (Student 2 R2) is borderline, with 57–60% of windows correct.
- **A + B's 8/8 is one outlier** among four settings and shouldn't be quoted on its own.

**What the model is picking up**

| Session average | Standing @A | Standing @B | Sitting @A | Sitting @B | Empty room |
| --- | --- | --- | --- | --- | --- |
| Subcarriers that never changed in 2 min (of 234) | 44 | 94 | 120 | 123 | 116–122 |
| Fluctuation size (`sd_mean`) | 0.0167 | 0.0132 | 0.0124 | 0.0120 | 0.0141 |
| Share of power in 0–0.5 Hz (sway, breathing) | 0.27 | 0.21 | 0.15 | 0.15 | 0.13–0.14 |

- Standing makes the signal fluctuate. Standing at B does so much less than at A. Sitting looks the same at both spots, and the same as the empty room.
- **Can feature set A tell spots apart? (leave-one-session-out, logistic regression)**
  - Posture with the spot fixed: 9/10 sessions at A, 9/10 at B.
  - Spot among standing sessions: 7/10, weakly.
  - Spot among sitting sessions: chance level (44% of windows).
- So the model hasn't memorised a map of the room. Instead, **the spot changes how strongly standing shows up**.
- **The static level** of every occupied session differs from the empty room by 0.024–0.027 RMS. The two empty-room recordings differ from each other by 0.024. Only about 8% of subcarriers move by more than 0.05. A body's static reflection is smaller than one BFM quantisation step on most subcarriers, so static features carry neither location nor posture on this setup. *(Updated 14 Oct: lying vs sitting shows a weak static posture signal within a day; see [Lying pilot results (14 Oct)](#lying-pilot-results-14-oct).)*

**Why standing at B is weaker (not yet tested)**

A and B are the same distance from the router–Pi pair, so distance doesn't explain it. The surroundings probably do. At A, the wall reflects energy back towards the link. At B, part of the reflection escapes through the open door. The [Door test (planned)](#door-test-planned) checks this. Either way, the result shows that **what surrounds a spot (wall, door, furniture) changes the signal**. That is why postures must share spots, and why the door state must stay fixed and be logged.

**Decision and next steps**

- **Spots:** adopt P1–P3 for standing and sitting (decision rule: "in between"), and use them for lying and waving too.
- **Lying pilot (14 Oct), before the main collection:** record lying at A and B, plus sitting at each spot and an empty-room session, interleaved. Check whether lying, sitting and empty can be told apart. If they can't, agree with the lecturer whether to merge the still postures or change the setup before collecting 250 sessions.
- **One "stand still" instruction for everyone:** today's standing was calmer than on 7/9 Oct (`absdiff_mean` about 0.010 vs 0.017), and the model relies on sway.
- **Caveats:** 2 people, one day, 8 test sessions. All numbers use the 13 Oct `rtbfm` features (with the flat-subcarrier fix).
- **Reproduce:** `python train_rt_bfm.py --pilot --glob "bfm_real_imag_csv/bfm_ri_data_Student *_Spot *_20261013_*.csv" --window 5 --clf xgb`. Change `--window 15` or `--clf lr` for the other settings.

### Lying pilot results (14 Oct)

**In short**

- We recorded 30 two-minute sessions (Students 1–3, 10 each) plus one empty-room recording. The layout was the same as on 13 Oct, with lying in place of standing: lying at A and sitting at B for training; lying at B and sitting at A for testing.
- **The motion features (set A) can't tell lying from sitting, even at the same spot** (12/18 sessions, about chance). Lying and sitting both look like the empty room, so the spot-swap question can't be answered for these features.
- **The static levels (set B) separate lying from sitting, weakly.** They got 26/30 sessions right on subjects the model had never seen (logistic regression). But the difference is smaller than the gap between two repeats of the same session, and the static levels drift during the morning. So far this has only been shown within one day.
- **The room's signal was different from 13 Oct.** The empty room fluctuated about twice as much, mostly on 64 subcarriers in a deep fade. Device placement matched the reference photo. The likeliest cause is the router setting up the link differently each time it restarts (untested).
- **Next:** check whether the static lying/sitting difference still holds on another day before deciding whether lying and sitting stay separate classes.

**Setup**

- **Same as 13 Oct:** spots A and B, door open, both chairs in place. The subject lay on the floor beside the chairs.
- **Order:** lying and sitting alternate. Student 2 went from 10:26 to 10:53, Student 3 from 10:58 to 11:22, and Student 1 from 11:26 to 11:51. Empty room at 10:16.
- **Data quality:** all 31 sessions are 120 s at about 9.8 Hz with no gaps, and all four folders are complete. One raw CSV (`Student 2 … Sitting_Spot B R3`) has one incomplete packet row, which the analysis files drop.
- **Not part of this pilot:** Student 3's standing/sitting block after 11:58. Its sitting sessions reuse descriptions from the pilot (e.g. `Sitting_Spot A R1` at 11:08 and again at 12:10), so tell them apart by timestamp.

**Spot-swap test** (sessions correct by majority of windows; pass mark 80% = 15/18 for the sanity check and 10/12 for the swap)

| Setting | Sanity check (of 18): A / B / A+B | Swap (of 12): A / B / A+B |
| --- | --- | --- |
| 5 s, XGBoost | 12 / 14 / 14 | 7 / 8 / 8 |
| 5 s, LR | 14 / 14 / 13 | 6 / 11 / 10 |
| 15 s, XGBoost | 12 / 15 / 14 | 9 / 9 / 10 |
| 15 s, LR | 12 / 14 / 14 | 7 / 10 / 9 |

- **Feature set A fails the sanity check in every setting.** It can't separate lying from sitting even at the spot it was trained on, so its swap result means nothing.
- **Feature set B passes the sanity check only once** (15 s, XGBoost). Its good swap scores with logistic regression need the cross-subject check below.

**Leave one subject out** (5 s windows; train on two subjects, test all 10 sessions of the third)

| Classifier | A. `rtbfm` | B. Static levels | A + B |
| --- | --- | --- | --- |
| XGBoost | 19/30 | 22/30 | 21/30 |
| Logistic regression | 21/30 | **26/30** (lying 13/15, sitting 13/15) | 26/30 |
| *13 Oct standing vs sitting, for comparison (XGBoost / LR)* | *19/20 / 18/20* | *14/20 / 17/20* | *17/20 / 19/20* |

Each subject was recorded in their own time block, so drift over the morning can't help this test. Both postures score equally well, so the result isn't one class soaking up the drift.

**What the signal shows** (session averages)

| Session average | Lying @A | Lying @B | Sitting @A | Sitting @B | Empty room | Standing @A (11:59, 1 session) |
| --- | --- | --- | --- | --- | --- | --- |
| Subcarriers that never changed in 2 min (of 234) | 85 | 84 | 70 | 80 | 77 | 0 |
| Fluctuation size (`sd_mean`) | 0.028 | 0.029 | 0.029 | 0.029 | 0.031 | 0.052 |
| Share of power in 0–0.5 Hz (sway, breathing) | 0.16 | 0.16 | 0.17 | 0.16 | 0.16 | 0.34 |

- **Lying, sitting and the empty room look the same** within normal session-to-session spread. A standing session recorded the same morning is clearly different, so the link could still see body sway. Lying and sitting just don't move enough to reach it.
- **Static levels:**
  - **Consistent direction:** the lying-minus-sitting difference points the same way for all three subjects and both spots (median correlation +0.32; 93% of pairs positive). The A-minus-B spot difference is less consistent (+0.22; 73%).
  - **Small size:** RMS 0.05, against 0.06 between two repeats of the same session.
  - **Drift:** the static levels also drift over time. Their correlation with the time between sessions is 0.47, and the distance from the 10:16 empty room grew from 0.09 to 0.17 by 11:51. This drift is why the same-spot sanity check fails.
- **This updates the 13 Oct finding.** Static levels do carry a weak posture signal within a day, which a simple classifier can pick up across subjects. Whether it holds across days is untested. This matters because the empty-room signal changed between 13 and 14 Oct.

**Why 14 Oct was noisier than 13 Oct**

| Empty room | 13 Oct | 14 Oct |
| --- | --- | --- |
| Fluctuation size (`sd_mean`) | 0.014 | 0.031 |
| Subcarriers in a deep fade (psi 0–4) | 0 | 64 (noise 0.085, vs ≤ 0.025 elsewhere) |
| Signal strength at the router | −29 dBm | −32 dBm |
| Router channel | 153 | 149 (same 80 MHz block) |

- **The extra noise was steady, not bursty.** Every 10-second block of the empty room measured 0.026–0.035, and all 30 sessions were at the same level. So people walking in the big room are at most a small part of it.
- **Most of it is quantisation inside the fade.** When psi is only 1–3, one quantisation step changes the log magnitude by up to 0.69.
- **The channel change doesn't explain it.** Channels 149 and 153 are in the same 80 MHz block, so the subcarriers sit at the same frequencies. The router picks its channel automatically each day (36 on 6 Oct, 157 on 7 Oct), so it could jump to a different band. Fixing the channel prevents that, but wouldn't have prevented this fade.
- **Possible causes:** device placement matched the reference photo, and the only room change found was the new extension lead. But the check below points to the router's link setup instead.
- **Extension check (14 Oct, 12:45; empty room with the old lead put back):** the room went most of the way back to its 13 Oct state.

  | Empty room | 13 Oct | 14 Oct 10:16 (new lead) | 14 Oct 12:45 (old lead) |
  | --- | --- | --- | --- |
  | Fluctuation size (`sd_mean`) | 0.014 | 0.031 | 0.018 |
  | Subcarriers in a deep fade (psi 0–4) | 0 | 64 | 14 |
  | Median phi | 31 | 35 | 31 |
  | Static-level distance from 13 Oct (RMS) | – | 0.94 | 0.32 |
  | Signal strength at the router | −29 dBm | −32 dBm | −33 dBm |
  | Router channel (all 80 MHz, same block) | 153 | 149 | 157 |

  - **The extension can't be singled out.** The new lead was in use from the first session of 14 Oct. The change seen at 12:45 coincides with swapping it back, but the router channel also changed then (149 → 157), so the link had been set up again.
  - **The pattern across all days points to the link setup, not the room.** Each run of sessions keeps one static state, through people, chairs, lying and sitting. The state jumps only when the router channel changes. The same channel on two days (157) gave two different states, so it isn't the channel number either.

    | Run | Channel | Fade subcarriers | Median phi | Distance to the 13 Oct state (raw psi) |
    | --- | --- | --- | --- | --- |
    | 7 Oct, 11 sessions | 157 | 51–54 | 33 | 4.6 |
    | 13 Oct, 21 sessions | 153 | 0 | 31 | 0.2 within the run |
    | 14 Oct 10:16–12:15, 40 sessions | 149 | 63–69 | 35–37 | 5.3 |
    | 14 Oct 12:45, empty room | 157 | 14 | 31 | 2.4 |

  - **Likely cause (untested):** something fixed when the link is set up, e.g. which router antennas sound or their calibration, changes the static channel each time. The signal strength fell steadily across days (−23 → −33 dBm), independent of these states.
  - **Test (about 15 min):** with nothing else touched, record an empty room, restart the router, then record another. Repeat once with the channel fixed. A jump in fade count or median phi after a restart confirms the cause.
- **Bandwidth drop (12:19):** in the middle of Student 3's `Standing_Spot B R2` session (after 69 s), the link fell from 80 MHz to 40 MHz and stayed there for the rest of that block. `Standing_Spot A R3`, `Sitting_Spot A R2` and `Sitting_Spot B R3` have only 108 subcarriers, and `Standing_Spot B R2` mixes the two widths. These four sessions can't be combined with 234-subcarrier data and need re-recording if they are wanted. By 12:45 the link was back at 80 MHz.

**Decision and next steps**

- **Spots: keep P1–P3.** On 13 Oct, the spot changed how strongly standing showed up. On 14 Oct, lying vs sitting relied on static levels, which depend on where the body is. In the main analysis, report a leave-one-spot-out test alongside leave-one-subject-out.
- **Lying and sitting as separate classes:** *resolved 16 Oct: lying replaced by bending; see [Restart test and decisions (16 Oct)](#restart-test-and-decisions-16-oct).*
- **Fix the router channel and width in OpenWrt** (e.g. channel 149, 80 MHz) so neither can change between or during sessions, and state it in the report.
- **Check every session has 234 subcarriers** before analysis; a 40 MHz drop gives 108.
- **Keep objects near the link unchanged,** including extension leads and bags, and note who is in the big room.
- **Feature analysis:** try features that cope with the fade (e.g. `sd_med` instead of `sd_mean`), because a fade inflates averages.
- **Caveats:** 3 people, one day, one empty room, 12 test sessions.
- **Reproduce:** `python train_rt_bfm.py --pilot --pilot-posture lying --glob "bfm_real_imag_csv/bfm_ri_data_Student *_Spot *_20261014_1[01]*.csv"`. Change `--window 15` or `--clf lr` for the other settings. The `1[01]` keeps only sessions started 10:00–11:59, which leaves out Student 3's later block. The leave-one-subject-out check was a one-off script and isn't in `train_rt_bfm.py` yet.

### Restart test and decisions (16 Oct)

**In short**

- **The router channel is now fixed** to 149 at 80 MHz (`uci set wireless.radio0.channel='149'`, `htmode='VHT80'`, committed). Left on `auto`, it had picked a different channel every day: 36, 157, 153, 149, 157, and 48 on the morning of 16 Oct.
- **A router restart alone changes the static signal, even with the channel fixed.** Nothing else was touched, yet the empty room moved by more than any change seen between days.
- **Within a run, the static signal stays put** while people sit and lie.
- **Decisions with the lecturer:** lying is replaced by bending (bowing), and all activities except walking use the shared spots P1–P3.
- **The cross-restart lying/sitting test was dropped** after the decision; the Pi also failed to reconnect after the second restart.

**Setup:** spot A, chair in place, door as before. Run 1: empty room at 13:29, then Student 3 (2 lying, 2 sitting). Router power-cycled for about 10 s, nothing else changed. Run 2: empty room at 13:51, then Student 3 (same 4) and Student 2 (2 lying, 2 sitting). All sessions have 234 subcarriers and are 120 s at about 10.3 Hz on channel 149 at 80 MHz.

**Result**

| Empty room | Run 1 (13:29) | Run 2 (13:51, after restart) |
| --- | --- | --- |
| Median phi | 33 | 30 |
| Subcarriers in a fade (psi < 4) | 54 | 53 |
| Fluctuation size (`sd_mean`) | 0.025 | 0.031 |
| Signal strength at the router | −38 dBm | −42 dBm |
| Static distance between the two (raw psi RMS) | 7.6 | |

- **The restart changed the static state** by 7.6, more than any change seen between days (2.4–6.6). So it's the link setup, not the room, the channel number or the extension lead.
- **Within each run, Student 3's sessions stayed close** to that run's empty room (0.17–0.56).
- **Noise and fade count barely changed this time.** So the noisy fade on 14 Oct was that particular state, not something every restart causes.
- **The signal keeps weakening:** −23 dBm (6 Oct), −27 (7 Oct), −29 (13 Oct), −33 (14 Oct), −38/−42 (16 Oct). Not yet explained; check the antennas and the Pi.

**What this means for the main collection**

- **Each router restart starts a new run.** Record an empty room at the start of each run, record every activity in every run, and log restarts.
- **Static-level features can't be trusted across runs.** Rely on motion features. This is why bending, a movement, replaced lying.
- **Evaluate with leave-one-run-out (or leave-one-day-out)** alongside leave-one-subject-out and leave-one-spot-out.
- **Data:** `bfm_*/*_20261016_*`, descriptions `Run1 …` and `Run2 …`. These sessions are kept for the report but are not part of the dataset.

### Bending and waving check (20 Oct)

**In short**

- **First trial (Student 3, 09:31–09:38):** bows with a 2 s pause, waving with both hands overhead. The two were hard to separate: waving was no bigger than bending, and the classifier mostly followed session intensity (worst case 47% of windows).
- **Instructions changed:** bending is continuous, with no pause; waving is at shoulder level.
- **Second trial (Students 1 and 2, 10:17–10:44, 2 rounds each):** the current 43 features separate the two activities in 98–100% of 5 s windows, across rounds and across subjects. No change to `rtbfm` is needed for these two classes.

**Setup:** spot A, one run (empty room at 09:17), channel 149 at 80 MHz. All sessions have 234 subcarriers and are 120 s with no gaps; router signal about −31.6 dBm.

**Session averages (Students 1 and 2)**

| Session | Motion size (`sd_mean`) | Frame-to-frame change (`absdiff_mean`) | 0–0.5 Hz share | Burstiness (p90/p10 of motion level) |
| --- | --- | --- | --- | --- |
| Empty room | 0.024 | – | 0.15 | 1.54 |
| S1 Bending R1 / R2 | 0.043 / 0.048 | 0.032 / 0.035 | 0.15 / 0.14 | 1.83 / 2.14 |
| S1 Waving R1 / R2 | 0.035 / 0.036 | 0.019 / 0.020 | 0.31 / 0.31 | 1.60 / 1.59 |
| S2 Bending R1 / R2 | 0.048 / 0.053 | 0.034 / 0.038 | 0.15 / 0.14 | 2.09 / 2.05 |
| S2 Waving R1 / R2 | 0.033 / 0.053 | 0.020 / 0.024 | 0.25 / 0.37 | 1.61 / 1.62 |

**Classifier (share of windows correct, bending / waving)**

| Train → test | 43 features, 5 s, LR | 43 features, 5 s, XGBoost | 4 rhythm features, 10 s, LR |
| --- | --- | --- | --- |
| S1 R1→R2 / R2→R1 | 100/100, 100/100 | 100/100, 77/100 | 98/90, 69/100 |
| S2 R1→R2 / R2→R1 | 100/100, 100/100 | 100/100, 98/100 | 91/88, 98/100 |
| S1 → S2 | 100/97 | 100/100 | 96/92 |
| S2 → S1 | 100/100 | 100/100 | 86/99 |

**What separates them**

- **Not motion size:** Student 2's Waving R2 is as large as bending (0.053) and is still classified correctly.
- **Frame-to-frame change:** `phdiff_*`, `absdiff_*` and `phrough_*` are about 1.6× higher for bending, with no overlap between sessions.
- **Rhythm:** bending is bursty, with its motion level rising and falling at about 0.6 Hz. Waving looks slow and smooth to the channel (high 0–0.5 Hz share, burstiness close to the empty room), so the model may see "arms raised, moving gently" rather than the wave rhythm itself.

**For real-time windows:** the rhythm features stay stable from 5 to 15 s windows, but the current features already do better, so rhythm features can wait until feature analysis.

**Caveats:** 2 subjects, one run, one day, spot A only. Still to test across restarts, days and spots P1–P3 (leave-one-run-out). Student 3's sessions use the old instructions and are left out of this check.

**Data:** `bfm_*/*_20261020_*` (Student 3: 09:31–09:38, old instructions; Students 1 and 2: 10:17–10:44, new instructions).

### Door test (planned)

**Question:** does the open door explain why standing at B sways the signal less than standing at A?

**Idea:** change only the door and keep everything else as on 13 Oct. Spot A is the control. If the open door weakens the signal at B, closing it should make standing at B look more like standing at A, while standing at A stays about the same.

**Predictions (agree before recording)**

| Result | Meaning |
| --- | --- |
| Closing the door makes standing @B much closer to standing @A; standing @A barely changes | Supports the door explanation |
| Standing @B doesn't change | The door isn't the cause; the likelier cause is where B sits along the router–Pi line |
| Standing @A and @B change by a similar amount | The door changes the whole room, not spot B specifically; door state must still be fixed for every session |

"Closer to standing @A" means: fewer subcarriers that never change (13 Oct: about 94 at B vs 44 at A), a larger `sd_mean`, and a larger share of power at 0–0.5 Hz (0.21 at B vs 0.27 at A).

**Sessions** (one subject, about 25–30 min; a second subject makes it stronger)

| # | Session | Door |
| --- | --- | --- |
| 1 | Empty room | Open |
| 2 | Empty room | Closed |
| 3 | Standing @B | Open |
| 4 | Standing @B | Closed |
| 5 | Standing @A | Closed |
| 6 | Standing @A | Open |
| 7 | Standing @B | Closed |
| 8 | Standing @B | Open |
| 9 | Standing @A | Open |
| 10 | Standing @A | Closed |

- **Order:** each spot and door combination appears once in the first half and once in the second, so drift over time can't create a door effect.
- **Two empty-room sessions,** because closing the door changes the room's own signal. Each standing session is compared with the empty room that has the same door state.
- **Keep everything else as on 13 Oct:** both chairs in place, the same "stand still" instruction, nobody else near the door, door fully closed.
- **When:** after the lying pilot, as its own block, so the lying sessions keep the door open as on 13 Oct.
- **Naming:** put the door state in the description, e.g. activity `Standing`, description `Spot B Door Closed R1`. For the empty room use subject `Nobody`, activity `Empty`, description `Empty Door Closed`.
- **Also note** where A and B fall along the router–Pi line (e.g. "A near the middle, B near the Pi end"). This is needed if the door turns out not to be the cause.

**Analysis**

- **Main measure:** for each standing session, take the three numbers above minus the matching empty room. Compare how much they change at B (closed vs open) with the change at A.
- **Optional:** run the 13 Oct pilot model on the door-closed standing-at-B sessions. If the door explains the errors, more of them should be called standing than the 2 of 4 on 13 Oct.
- With one subject and 2 sessions per condition the result is indicative. A clear change at B and none at A would be good evidence.

### Door check with sitting (20 Oct)

**In short:** closing the door made no consistent difference. Sitting looks like the empty room in every condition, so this block can't show a door effect either way. Spot B again loses more of the signal than A.

**Setup:** sitting at A and B with the door open and closed, Students 1 and 2, one session each (8 sessions, 10:58–11:19). Order was reversed between subjects (Student 2: closed B, closed A, open B, open A; Student 1: open A, open B, closed A, closed B). Same run as the 09:17 empty room; no door-closed empty room was recorded. Student 1's closed-B session dropped to 40 MHz at 97 s, so only its first 96 s are used.

| Session (time) | Never-changing subcarriers | `sd_mean` | `sd_med` | 0–0.5 Hz share | Static distance from empty room |
| --- | --- | --- | --- | --- | --- |
| Empty room (09:17) | 89 | 0.024 | 0.000 | 0.15 | – |
| S2 Closed B (10:58) | 69 | 0.023 | 0.000 | 0.16 | 0.131 |
| S2 Closed A (11:01) | 61 | 0.022 | 0.000 | 0.15 | 0.093 |
| S2 Open B (11:04) | 100 | 0.024 | 0.000 | 0.15 | 0.119 |
| S2 Open A (11:06) | 76 | 0.022 | 0.000 | 0.15 | 0.132 |
| S1 Open A (11:09) | 77 | 0.023 | 0.000 | 0.15 | 0.093 |
| S1 Open B (11:12) | 92 | 0.023 | 0.000 | 0.16 | 0.119 |
| S1 Closed A (11:15) | 73 | 0.022 | 0.000 | 0.17 | 0.111 |
| S1 Closed B (11:17, 96 s) | 96 | 0.024 | 0.000 | 0.16 | 0.122 |

- **Sitting matches the empty room on every motion measure,** with either door state and at either spot, as in the 13 and 14 Oct pilots.
- **The door effect is inconsistent:** at B, closing it gave 31 fewer never-changing subcarriers for Student 2 but 4 more for Student 1.
- **The door barely changes the static signal at B** (open vs closed about 0.03, less than the difference between subjects); at A it changes by 0.05–0.11.
- **Spot B has more never-changing subcarriers than A** (92–100 vs 61–77, except Student 2's first session), as on 13 Oct (94 vs 44).

**Decision:** no reason to close the door for collection; keep it the same for every session. To settle the door question, repeat with standing or waving at B and include a door-closed empty room, as planned above.

**Data:** `bfm_*/*_Sitting_Door *_20261020_*`.

### Bandwidth drops to 40 MHz (20 Oct)

**In short:** twice so far, the link has dropped from 80 to 40 MHz mid-collection and stayed there until the router restarted. Each session after the drop has 108 subcarriers instead of 234. The cause is the router's own rate control; a firmware setting can stop it using 40 MHz.

**What happened**

| Day | Drop started | Lasted until | Sessions affected |
| --- | --- | --- | --- |
| 14 Oct | 12:19 (69 s into Student 3, Standing @B R2) | Router restart (extension swap, before 12:45) | 4 whole sessions + part of one |
| 20 Oct | 11:19 (97 s into Student 1, Sitting Door Closed @B R2) | Still at 40 MHz at 11:26 | Last 23 s of that session |

All other October pcaps are 80 MHz throughout (a few have one stray frame).

**Findings**

- **The router is a D-Link DIR-842 C2** (QCA9888, `ath10k` driver with Candela `10.4b-ct` firmware, OpenWrt 24.10.0). The firmware runs rate control itself.
- **The router chose 40 MHz, not the Pi.** At 11:26 the router sent to the Pi at 40 MHz (VHT MCS 6–7), while the Pi sent to the router at 80 MHz (MCS 9). The radio was still configured at 80 MHz. The Pi stayed associated throughout (connected since boot), so it wasn't a reconnection.
- **The sounding follows the data width.** The BFM report width matched the router's transmit width, so a rate-control drop shrinks every BFM report.
- **No neighbouring network overlaps our channel.** A laptop scan found nothing on 149–161; the campus networks use 36–132.
- **The channel is busy with our own traffic.** It was busy 72% of the time, 68% from the router transmitting, and 27% of the router's packets needed a retry. A short burst of failures at 80 MHz can push rate control down to 40 MHz, and it doesn't climb back.
- **Both drops started while a subject was at spot B.** Spot B seems to block more of the link, but other spot-B sessions didn't drop, so this is only a hint.

**Fix (option 1 applied 20 Oct, about 11:40)**

1. **Stop rate control from using 20 and 40 MHz** (`ct_special` id 0xE, mask 0x3). It takes effect the next time rates are set, i.e. when the Pi reconnects:

   ```sh
   echo 0x0000000E00000003 > /sys/kernel/debug/ieee80211/phy0/ath10k/ct_special
   ```

   Then run `iw dev phy0-ap0 set bitrates` (re-applies rates without disconnecting the Pi) and check `iw dev phy0-ap0 station dump | grep 'tx bitrate'` shows 80MHz. To undo, write `0x0000000E00000000`.

   *Applied 20 Oct: the router went from 40 MHz (VHT MCS 7) to 80 MHz (MCS 1–4) with the Pi still connected. A 30 s test capture under 20 Mbit/s traffic gave 300 BFM reports, all 80 MHz (about 10 Hz). Both lines are now in the router's `/etc/rc.local` (backup `/etc/rc.local.bak`), so they rerun at boot; a reboot hasn't been tested yet. Watch whether the drop comes back over the next collection days.*
2. **Fallback if 1 doesn't hold:** fix the data rate with `set_rate_override` (e.g. `phy0-ap0 tpc=255 sgi=1 mcs=7 nss=1 pream=3 retries=4 dynbw=0 bw=2 active=1`). This forces 80 MHz and also removes rate changes, but it's a stronger change; test it first.
3. **Check every session before recording:** confirm the router sends at 80 MHz, and check the subcarrier count after each session. If it drops: stop, fix, treat it as a new run (empty room), and re-record the affected session.
4. **Optional:** reduce the `iperf3` traffic load so the channel is less saturated, if the capture rate stays at 10 Hz without it.

**Still to check:** whether reconnecting the Pi alone (without a router restart) also shifts the static state; if it does, a reconnect also counts as a new run.

### Evidence from the existing standing data

The 25 standing sessions already collected (5 subjects × 5 rounds, 7 and 9 Oct, same spot) show the shortcut mechanism directly: models score high when a nuisance factor lines up with the label and collapse when it does not. Here the nuisance factor is recording time; in the planned dataset it would be the spot.

Data quality: all 25 sessions are 120 s at about 9.7 Hz, with no gaps and a single router–Pi link.

**How different are two standing sessions?** RMS difference of the session's static log-magnitude profile across 234 subcarriers:

| Comparison | Median | Range | vs same subject |
| --- | --- | --- | --- |
| Two 10-s blocks inside one session | 0.058 | — | 0.8× |
| Same subject, same day | 0.072 | 0.022–0.111 | 1× |
| Different subject, same day | 0.078 | 0.033–0.110 | 1.1× |
| Different day (7 Oct vs 9 Oct) | 0.389 | 0.364–0.417 | 5.4× |
| Student 3 (7 Oct, 10:21–10:40) vs all others | 1.189 | 1.042–1.322 | 16.5× |

Phase follows the same pattern (median 0.004 rad same subject, 0.005 different subject, 0.08 different day, 0.24 Student 3). Student 3's own sessions also drift steadily: the mean level rises from 2.577 to 2.603 across the five rounds.

**Can the features tell two subjects apart when both stand at the same spot?** Leave-one-session-out, logistic regression, 5-s windows:

| Test | Is recording time lined up with the subject? | A. Current `rtbfm` features | B. Static-level features |
| --- | --- | --- | --- |
| 7 Oct: Student 1 vs 2 (all of Student 1's sessions, then all of Student 2's) | Yes | 78% of windows, 9/10 sessions | 96%, 10/10 |
| 9 Oct: Student 4 vs 5 (sessions interleaved in time) | No | 28%, 0/10 | 19%, 1/10 |

On 7 Oct both feature sets appear to recognise the person. On 9 Oct, where the two students alternated, both fall below chance: each held-out session is assigned to whoever stood in the neighbouring time slots. The apparent "subject recognition" on 7 Oct is mostly recognition of the time block. This is one pair of subjects per day, so it is strong but not conclusive evidence.

**What this changes**

- **The shortcut risk is real on this setup**, even for the motion-only real-time features. It supports crossing spots with postures.
- **Interleave activity order within each subject's collection.** On 7 Oct, Students 1 and 2 recorded all standing rounds, then all walking rounds. For still postures, a blocked order lets time drift stand in for the label.
- **Static levels need a per-block reference.** Day-to-day shifts (5×) and one-off steps (16×) dwarf differences between people. Use the empty-room baseline recorded at the start of each block, or normalise per block; never compare raw levels across days.
- **Run the pilot in one block, interleaved, with the devices untouched** (e.g. A-stand, B-sit, B-stand, A-sit, …). Within-day drift is about 0.07, so a spot effect must clearly exceed that to show up.
- **Open question:** something changed on 7 Oct between Student 3's last session (walking, 10:49) and Student 1's first (10:53). If the router or Pi was moved or restarted, Student 3 should be treated as a separate setup or excluded.

### Consistency across collection days (standing and walking)

Standing looks the same on 7 Oct (Students 1, 2) and 9 Oct (Students 4, 5), and walking is clearly separable from standing on both days. But a model trained on one day only can fail on the other. Student 3 excluded; 50 sessions, all 120 s at about 9.7 Hz with no gaps.

| Session-median motion feature | Standing 7 Oct | Standing 9 Oct | Walking 7 Oct | Walking 9 Oct |
| --- | --- | --- | --- | --- |
| Magnitude change between samples (`absdiff_mean`) | 0.0174 | 0.0167 | 0.0224 | 0.0259 |
| Phase change between samples (`phdiff_mean`) | 0.0036 | 0.0036 | 0.0046 | 0.0052 |
| Share of signal in 0–0.5 Hz | 0.64 | 0.60 | 0.56 | 0.38 |
| Share of signal in 1–2 Hz | 0.08 | 0.09 | 0.10 | 0.14 |

- **Standing is consistent across days.** Walking moved the signal more on 9 Oct; pace, path or the subjects may differ.
- **Static level:** the whole baseline moved by about 3 quantisation steps between days (RMS 0.39). The shift is the same for standing and walking (correlation 0.99 across subcarriers), so the room or channel moved as a whole. The daily empty-room recording can correct this.

| Standing vs walking test (XGBoost, settings of `train_rt_bfm.py`) | Balanced accuracy |
| --- | --- |
| Within 7 Oct, held-out student | 96% |
| Within 9 Oct, held-out student | 99.7% |
| Train 9 Oct → test 7 Oct | 92% (all 20 sessions correct by majority) |
| Train 7 Oct → test 9 Oct | 57% (87% of walking windows called standing) |

The 7 Oct model relied on how unevenly motion spreads across subcarriers (`absdiff_std`, `phdiff_std`). That spread depends on the room's response on the day, and 9 Oct walking looked like 7 Oct standing on it. The model still ranks walking above standing (AUC 0.99), but its cut-off lands in the wrong place. Removing the `corr_*` features does not change this (52% / 93%).

*Note (13 Oct): the band-share rows above and the XGBoost results below were computed before the flat-subcarrier fix in `rtbfm` (those values were inflated by subcarriers that never changed). Recompute them before they go in the report.*

**Actions:** train on more than one day (the current script already does); add a leave-one-day-out test; give identical walking instructions (path, pace) every day; treat across-subcarrier spread features (`*_std`) with caution.

## Open decisions for senior and lecturer

- [ ] Agree the pilot's pass mark (e.g. ≥ 80% of swapped sessions correct) before running it.
- [x] If the pilot shows a location effect, agree the crossed-spot protocol (P1–P3) with the co-senior, since it is a shared dataset. *Agreed 16 Oct.*
- [x] Lying vs sitting: keep both, merge, or replace? *16 Oct: lying replaced by bending (bowing).*
- [ ] If P1–P3 is adopted: can subjects 1–5 return for 4 extra standing sessions (2 at P2, 2 at P3)?
- [ ] Exact P2/P3 distances, after a short pilot checking waving is visible at the farthest spot.
- [ ] Rotated activity order within each subject, and a 2-minute empty-room baseline per run (each day and after every restart): confirm both with the co-senior.
- [ ] Open-space set (3 subjects): confirm venue and whether time allows.
