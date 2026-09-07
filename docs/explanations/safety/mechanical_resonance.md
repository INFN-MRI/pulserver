# Mechanical resonance

```{admonition} TL;DR
:class: tip

**The criterion.** A gradient coil rings when it is driven inside one of its
forbidden bands *and* the drive is held there long enough for the mode to
build up. So, per physical gradient axis and at every frequency $f$ inside a
band, the check reads the amplitude of the sinusoid the scan sustains over the
coil's memory $W$,

$$A_W(f) = \max_{t_0}\;\frac{2}{W}\Bigl|\int_{t_0}^{t_0+W} g_\text{ax}(t)\,
e^{-2\pi i f t}\,\mathrm{d}t\Bigr|,$$

in mT/m, and compares it with what the band tolerates. A waveform whose
spectrum misses the band passes whatever its amplitude and however long it
runs. Inside the band, both matter and neither alone decides: how strongly the
waveform drives that frequency, and how much of the coil's memory it fills
before it stops.

**Two empirical constants, calibrated together.** $W$ is a property of the
coil, not of the band or of the sequence: `pulseg_opts.mech_memory_us`, **20
ms** unless the scanner configuration says otherwise. A band that states no
amplitude of its own is held to `SA_ZERO_BAND_SINUSOID_MT_PER_M`, **10 mT/m**.
Neither number is a vendor datum: both were fixed at once, by reading a corpus
of product prescriptions and asking which window and which threshold separate
the ones a lockout table refuses from the ones it covers and allows.

**Nothing is told what it is looking at.** What is read is the composite
gradient on each physical axis — every waveform the scan plays, the sequence's
own `ROTATIONS` applied, the prescription rotation composed in. There is no
echo-train detector and no equivalent echo spacing for a spiral: an EPI
readout, a bSSFP comb and a spiral arm all go through the same integral, and
are separated only by what they sustain.

**It needs a band table, and it is not the gate.** Forbidden bands are site
data that no sequence file carries, so the check runs when they are supplied
and is skipped rather than guessed when they are not. The verdict is an
estimate that runs ahead of the scanner's own preflight gate.
```

## Background

A gradient coil is a set of conductors clamped inside the bore of a magnet, so
every ampere of gradient current is also a Lorentz force on the former that
carries it. That former is stiff and only lightly damped, and like any such
structure it has mechanical modes — shapes it prefers to deform into, each
with its own natural frequency. Driving one of them is not the same as driving
the coil anywhere else: on resonance the deformation is amplified by the
mode's quality factor rather than merely transmitted. What comes out is
acoustic noise, vibration of the bore and, where that vibration moves
conductors in the static field, a perturbation of $B_0$ that the acquisition
sees as ghosting and signal loss.

```{figure} ../assets/mechanical_resonance/mode_response.png
One mechanical mode, in its own units. **(a)** frequency response: the mode
answers over a band of width $\Delta f$ about its natural frequency $f_0$
(shaded), not at a single frequency. **(b)** build-up in time under a drive at
$f_0$: full deformation takes about one memory, so a drive that stops earlier
(marker) never reaches it.
```

Two numbers describe a mode, one per panel, and the check needs both of them.

- **A frequency range — panel (a).** A mode does not answer at a single
  frequency but over a band of width $\Delta f$ around $f_0$, narrow compared
  with $f_0$ itself and set by how heavily the structure is damped. Vendors
  publish these as **forbidden bands**, stated per gradient axis, because a
  mode that deforms the coil along one axis is driven by the gradient on that
  axis and not by the others.
- **A memory — panel (b).** The same damping fixes how long the mode takes to
  reach full amplitude once a drive at $f_0$ starts, and how long it rings
  after that drive stops; it is the reciprocal of $\Delta f$, to within a
  factor of order one. At a fraction of a memory the deformation is a fraction
  of the steady one, so a burst that ends early is not the same stimulus as a
  train that keeps going.

Bands differ from coil to coil, and in principle each carries its own memory.
The check uses one window for all of them, chosen empirically — which one, and
why, is {ref}`the calibration section <mechres-window>`.

### Violation criteria

The two panels are two independent tests, and a waveform is a hazard only when
it fails both at once:

- it has content at a frequency inside the band;
- it holds that content long enough for the mode to build up.

Miss either and the mode stays quiet, and real sequences miss them for
opposite reasons. A spoiler with a steep edge has content at every frequency,
every band included, but it is over in a few hundred microseconds and leaves
the mode barely moved. A long flat readout runs for tens of milliseconds but
puts its content near DC, far below any band. Neither is a hazard.

That is the rule a product check encodes, and it is qualitative: it answers
*whether*, not *how much*. Three quantities decide how much, and all three are
properties of the gradient waveform alone:

1. **Where its content sits** — whether the waveform has any Fourier content
   at a frequency the band covers.
2. **How hard it drives there** — the amplitude of that content, in mT/m,
   since the deformation a mode reaches is proportional to the force driving
   it.
3. **How much of the memory it fills** — a burst lasting a tenth of the memory
   leaves the mode at a tenth of the deformation a sustained drive would
   reach.

The last two trade against each other. A short intense burst and a long weak
train can leave the coil in the same state, so no test on duration alone and
no test on amplitude alone separates the hazards from the rest — and a test on
frequency alone, which is what a zero-tolerance parameter lockout is, does not
either. The three have to collapse into a single number at each frequency.
That number is what the rest of this page computes.

### Sequence mechanical drive - equivalent sinusoid

A mode is a linear oscillator, so it does not respond to a gradient waveform
as a shape. It responds to however much of that waveform looks, at its own
frequency, like a sinusoid it can be pushed by — and only over as long as it
remembers. That is enough to fix the quantity. At a frequency $f$, over a
window of the coil's memory $W$ starting at $t_0$, the drive a gradient $g(t)$
delivers is the amplitude of the single sinusoid at $f$ carrying the same
Fourier content over that window:

$$A_W(f) = \frac{2}{W}\Bigl|\int_{t_0}^{t_0+W} g(t)\,e^{-2\pi i f t}\,
\mathrm{d}t\Bigr|.$$

It is a gradient amplitude and it is reported in mT/m. The factor $2/W$ is
fixed by the only calibration that makes the name honest: feed in a pure
sinusoid of amplitude $A$ at $f$ and $A_W(f)$ returns $A$.

```{figure} ../assets/mechanical_resonance/equivalent_sinusoid.png
**(a)** gradient temporal envelope: the same readout lobe played 2, 4, 8 and
16 times at a fixed plateau, against one window $W$. **(b)** mechanical
response at the train frequency against train length: it rises in proportion
to the length until the train fills the window, and stops rising after that.
```

Written that way, the three quantities of the previous section are all inside
one number, and the figure shows the two a sequence designer controls.

- **Duration enters as the fraction of the window it fills.** Up to one
  window, the reading rises in proportion to the train's length — a train a
  quarter of a window long deposits a quarter of what it eventually will. Past
  one window it stops rising: the integral has run out of window, and playing
  more echoes adds nothing at that frequency. That is why an echo train and a
  phase-encode blip built from the same lobe are not the same stimulus.
- **Amplitude enters linearly.** Double the plateau and the reading doubles,
  at every frequency and every duration, because the integral is linear in
  $g$.

The saturated value is a property of the lobe pattern, not of the amplifier: a
train that outlasts the window reads *above* its own plateau, at $4/\pi$ of it
for lobes that alternate squarely and a little less once the ramps are finite.
A readout train is a more effective drive than its peak gradient suggests,
which is why a plateau is not the quantity to compare against a band.

Two things this definition deliberately does not do. It does not ask what the
waveform is for — an echo train, a spiral arm and a spoiler enter the same
integral. And it does not depend on where the window is placed, until it does:
$A_W$ is written at one $t_0$ here, and a scan is judged at the worst $t_0$
anywhere in it, which is the subject of {ref}`the calculation section
<mechres-drive>`.

Two pairs taken from the shipped sequences show what that does in practice.
Each pair sets an element that drives the coil beside one that does not, and
both go through the same integral with nothing told about which is which.

```{figure} ../assets/mechanical_resonance/elements_epi.png
An echo-planar train on the axis that plays it. **(a)** the readout gradient,
alternating lobes for tens of milliseconds, and **(b)** the phase-encode blips
riding between them, on the same vertical scale. **(c)** and **(d)** their
mechanical responses, on the same scale as each other.
```

The readout is the case the definition was built for. It outlasts the window,
so it reads at the amplitude it sustains rather than a fraction of it; its
lobes alternate, so its period is twice the echo spacing and its line sits at
$1/(2\,\mathrm{ESP})$; and the reading comes out above the plateau, as a train
of alternating lobes must.

The blips are the counter-example, and they are not quiet for being small —
they reach nearly half the readout's peak gradient. They are quiet for two
reasons the definition already contains. Each one lasts a fraction of a
millisecond against a window of twenty, so it fills almost none of it. And
consecutive blips carry the *same* sign, so their period is one echo spacing
rather than two and their line sits an octave above the readout's, where the
lobe's own transform has already fallen away. Peak gradient predicts neither
of those, and the two panels differ by more than an order of magnitude.

```{figure} ../assets/mechanical_resonance/elements_spiral.png
One spiral trajectory acquired two ways: **(a)** a single long arm and
**(b)** the same coverage in sixteen short ones, on one vertical scale.
**(c)** and **(d)** their mechanical responses, also on one scale.
```

A spiral has no echo spacing to look up, and nothing in the check goes looking
for one. Its gradient sweeps in frequency as the arm winds outward, so it
crosses a band once and leaves behind whatever it sustained while it was
inside — which is set by how slowly it crosses. Both arms here reach the same
peak gradient and cover the same trajectory; the long one dwells in the region
the figure marks for several milliseconds and the short ones pass through it
in a fraction of that, and the readings differ by a factor of nearly three.

Neither pair needed a family label, an echo-train detector or an equivalent
echo spacing. Four waveforms, one integral, and the differences that come out
are the ones the physics puts there.

### Sequence-informed mechanical resonance check

A scanner's own check knows which sequence it is about to play, and it is
worth being exact about what that buys — because it is precisely what a `.seq`
file takes away.

Knowing the family, three decisions can be made before a scan is built:

1. **Which element could ring the coil.** In an echo train it is the readout:
   the one gradient that repeats, at one amplitude, for tens of milliseconds.
2. **Which parameter moves it.** The echo spacing sets the readout's period,
   so the drive frequency is a known function of a number already on the
   prescription card — $1/(2\,\mathrm{ESP})$ for lobes that alternate.
3. **Which values to forbid.** A table of parameter ranges, per gradient coil.
   A prescription that falls inside one is refused before any waveform exists.

Because the element is known, the rule can be sharper still: it can name the
axis the readout sits on, and it can state the plateau amplitude below which
that spacing is allowed after all.

Nothing else in those sequences is examined — and the echo-planar pair above
is why that is defensible rather than lazy. The blips ride on the same
gradient system, reach nearly half the readout's peak, and drive an order of
magnitude less. An element-by-element check would spend its effort on them for
nothing.

The scheme works exactly as long as the family cooperates. A `.seq` file
carries no family label, nominates no element and offers no knob to look up;
and the sequences most likely to arrive in one — a spiral, a radial
acquisition, an arm optimised per shot — are outside such a scheme even on a
scanner that has it. The check has to read the waveform.

## Problem statement

The interpreter is handed a finished `.seq` file and a table of forbidden
bands, and has to return a verdict. Every one of the three decisions the
previous section rests on is missing: there is no family, no nominated
element, no parameter to look up. What is left is the waveform and the
quantity the section before it defined.

That fixes the shape of the problem. Three requirements, and they pull against
each other:

1. **It must be sequence-agnostic.** Find the frequencies an arbitrary
   sequence drives and the amplitude at each, without being told what kind of
   sequence it is — no echo-train detector, no equivalent echo spacing
   invented for a spiral, no clause that fires on acquisition blocks and not
   on spoilers. One integral, applied to every gradient the scan plays.
2. **It must reproduce the decisions a scanner already makes.** On the
   families a product check does cover, the verdict has to agree. This is the
   only external evidence there is: no published dataset states how much
   sustained gradient a coil tolerates, so a vendor's own refusals and
   allowances are the calibration, and any divergence has to be understood and
   defensible rather than discovered later.
3. **It must not be over-conservative.** Every gradient puts *some* drive into
   every band, so a criterion without a threshold refuses everything; a
   threshold set too low refuses the protocols a site runs daily. A gate that
   fires on working prescriptions is not a safety feature, it is a reason to
   switch the gate off.

Requirement 1 is a matter of construction and is settled in the next section.
Requirements 2 and 3 are the same requirement seen from two sides, and between
them they leave exactly two numbers undetermined:

- **the window $W$** — how long a drive must be sustained before it counts;
- **the threshold** — how much sustained drive a band tolerates when it states
  nothing itself.

Neither is published anywhere, and neither can be derived from first
principles: the first would need the damping of each mode, the second the
force at which a coil is harmed. Both are read out of the same corpus of
product prescriptions, and they are read *together*, because a window that
changes moves every reading and so moves the threshold with it. That
calibration is the last section of this page.

One constraint that does not change the criterion but does constrain the
implementation: the verdict runs on a finished scan, which may be a million
blocks, and is one of several checks the interpreter performs before a scan
starts. How the same number is computed at that scale is the
{doc}`performance page <../performance/mechanical_resonance>`.

(mechres-drive)=
## Sequence mechanical drive calculation

A scan is not one window long. The quantity defined above is written at a
single start time $t_0$, and a scan offers as many start times as it has
raster points, so the drive a sequence delivers is the worst of them:

$$A_W(f) = \max_{t_0}\;\frac{2}{W}\Bigl|\int_{t_0}^{t_0+W} g_\text{ax}(t)\,
e^{-2\pi i f t}\,\mathrm{d}t\Bigr|.$$

Taking the maximum is not a formality. A coil does not average over a scan; it
is rung by the loudest stretch in it, and the rest of the scan cannot undo
that.

```{figure} ../assets/mechanical_resonance/window_placement.png
**(a)** one long spiral arm with the window laid on it at three placements
(shaded boxes), and **(b)** the mechanical response at each placement. Early
in the arm the sweep is nowhere near the frequency being read; late in the arm
it dwells there, and the same waveform reads forty times more. The verdict is
the peak.
```

What goes into the integral is the composite gradient on the axis — every
event the scan plays there, added as it is played, with no distinction between
a readout and a spoiler, and none between a gradient played under an RF pulse,
one played under an ADC, and one played under neither. A slice-select gradient
rings the coil exactly as a readout does, and is read exactly as one.

Two regimes follow, and they are the same arithmetic seen from opposite ends:

**A repetition shorter than the window.** Several repetitions fall inside one
window and add coherently, so what the window reads at a harmonic of the
repetition rate is what a steady comb of that amplitude would deliver — the
drive builds across repetitions rather than being counted one at a time. Where
a repetition is much shorter than the window, sliding changes almost nothing:
every placement contains the same number of repetitions, which is why the
maximum over $t_0$ costs nothing in that case and matters enormously in the
one above.

**A waveform longer than the window.** Only part of it is ever inside, so it
cannot be read as one object. The window is slid within it and the loudest
stretch decides — the figure is exactly that, and averaging over the arm
instead would report a fortieth of the drive the coil actually receives.

One boundary falls out of the definition rather than being imposed. Below
$f < 1/W$ the exponential completes less than one turn across the window, the
integral degenerates towards the gradient moment over that window, and the
reading goes to zero. No mechanical mode answers there, and no forbidden band
this check is given sits there either.

### Prescription orientation handling

Forbidden bands belong to physical gradient channels: a mode that deforms the
coil along one winding is driven by the current in that winding, and the three
channels of a coil differ both in where their bands sit and in whether they
have any at all.

A sequence is written in the logical frame, so the drive has to be carried
into the physical one before it can be judged. Two rotations apply, in order:

- the sequence's own `ROTATIONS`, which turn a spoke or an arm shot by shot;
- the prescription matrix the operator sets by choosing a slice orientation,
  composed to the left of them.

The composite on each physical channel is then read against the bands that
guard *that* channel. A sequence whose loudest drive sits on a lightly guarded
channel when the slice is axial arrives on a guarded one, scaled by the
direction cosine, as the prescription tilts.

## Resonance threshold calibration

Two numbers are still undetermined, and neither can be looked up. What is
available instead is a corpus of designs, sorted by what the product's own
check did with each — read from its lockout tables, not assigned by hand:

- **checked, refused** — the product checks this family, and this design's
  parameter falls inside a locked range;
- **checked, accepted** — the product checks this family, and this design's
  parameter is outside every locked range, so the product runs it;
- **not checked** — no lockout covers the family at all. This is not
  permission: it means the product has no opinion. These designs are evidence
  about whether the check refuses ordinary imaging, and about nothing else.

Each design is read on the physical axis that drives it and compared with what
its own band tolerates — a stated plateau where a band gives one, the
threshold where it does not. That ratio is the unit of everything below: one
means a design sits exactly at what its band allows.

A threshold is not optional. The harmonics of any repetition are $1/T_R$
apart, so a band of width $\Delta f$ contains one whenever $T_R$ exceeds
$1/\Delta f$ — which, for bands as narrow as a coil's, is true of essentially
every sequence ever prescribed. "A harmonic falls in the band" is therefore a
test that refuses everything, and the only content a zero-tolerance column can
carry is *how much* that harmonic delivers.

(mechres-window)=
### Window width selection

The window and the threshold cannot be chosen one at a time: change $W$ and
every reading moves. So fix the threshold at what the bands themselves state,
and ask what each candidate window then does to the corpus: how much of what
the product refuses it refuses too, and how much of what the product accepts
it refuses by mistake.

```{figure} ../assets/mechanical_resonance/window.png
**(a)** every checked design re-read at ten windows — refused by the product
in orange, accepted in blue — against the band tolerance (dashed). **(b)** what
the shipped threshold then does at each window: the share of the product's
refusals it reproduces, against the share of its acceptances it refuses in
error.
```

Two physical constraints squeeze the window, and panel (b) shows both:

- **Too short and the window cannot resolve a band.** A reading over $W$
  cannot separate content closer than $1/W$, so at a few milliseconds an echo
  train whose fundamental sits *outside* a band leaks into it. At the shortest
  window every design the product accepts is refused — the check reproduces
  everything and means nothing.
- **Too long and the window dilutes the trains the product refuses.** The
  shortest refused echo train is a burst of little more than ten milliseconds;
  past that its reading falls as $1/W$ while a steady comb's does not.
  Reproduction collapses from three fifths of them to a tenth.

**The window chosen is 20 ms** — `pulseg_opts.mech_memory_us`, settable from
Python as `memory=`. It is the shortest window at which nothing the product
accepts is refused, and the last at which reproduction is still high: three
fifths of the product's refusals, and none of its acceptances. One step longer
and reproduction falls by two thirds.

### Threshold selection

With the window fixed, the whole corpus can be read at once. A band that
states a plateau supplies its own tolerance; a band that states zero is held
to `SA_ZERO_BAND_SINUSOID_MT_PER_M`, 10 mT/m, and that constant is the last
free number in the check.

```{figure} ../assets/mechanical_resonance/scenario_table.png
Every design at the chosen window, against what its own band tolerates. A bar
past the dashed line is refused here. The three groups are what the product
does with each: checks and refuses, checks and accepts, or does not check.
```

Three things are worth reading off it.

**Nothing the product accepts is refused.** The loudest of them reaches
0.95 of what its band tolerates and stops there — an echo-planar train whose
fundamental sits just outside a band and leaks in through the window's own
resolution. The margin is thin by design: a threshold lower by five percent
would start refusing prescriptions a scanner runs today.

**A band that says nothing is held to more than a band that speaks.** The
constant works out at about three-quarters of the tolerance the tables state
where they state one, so a zero column is read as the stricter statement
rather than as an absent one.

**What the product never checks is now checked.** Most of that group sits far
below the threshold — spin echo, PROPELLER, radial, ZTE, the multi-arm spirals
— which is the evidence that ordinary imaging is not disturbed. The exception
is the single 73 ms spiral arm, which reads twice the threshold and is refused
on the same arithmetic that refuses an echo train. No lockout mentions it, and
on a product scanner nothing would have looked.

### Automatic vs sequence-informed mechanical resonance check

**Where they agree.** An echo train whose fundamental lands in a band is
refused, as it is on the scanner. A flyback train at a locked echo spacing is
refused once it runs long enough to fill the window. A balanced steady state
at a locked repetition time is refused on whichever channel carries the drive
— the slice select in a 2D acquisition, the readout in a 3D one — which the
check finds without being told that the family, or the difference, exists.

**Where they part.** Every disagreement found on the corpus runs the same way:
the product refuses a prescription this check allows. That is not an accident.
A lockout table has one number per row to work with, so where a distinction
would cost a new column it is cheaper to refuse the whole range — and each
divergence is one of those distinctions:

- **the readout is too weak.** A low-bandwidth echo-planar train has a shallow
  readout gradient and sustains a few mT/m at its own frequency. The lockout
  refuses it on the spacing alone; here it is read and allowed.
- **the train is too short.** A handful of echoes at a locked spacing fills a
  fraction of the window and never builds the mode up.
- **the lobes alternate.** A bipolar multi-echo gradient echo is an
  echo-planar readout without the blips: it repeats every two echo spacings,
  not every one. The table that locks its spacing is written for flyback
  trains, which repeat every one — so at that spacing the bipolar train drives
  half the guarded frequency and passes. At a spacing the echo-planar table
  locks it is refused, like the echo train it is. The check disagrees with one
  table by agreeing with the other.
- **the drive is on an unguarded channel.** A repetition-time lockout names no
  axis, so it refuses whatever the orientation. This check reads each physical
  channel against the bands that guard it, so a near-axial prescription whose
  loudest drive lands on a channel with no bands is allowed — and refused
  again as soon as the prescription tilts.

In each case the product is being conservative about something a table cannot
express, and none of them can be refused on amplitude, because in each there
is no in-band amplitude to refuse. Nothing on the corpus runs the other way:
no prescription is refused here that a product check allows.

**What the automatic check adds.** It covers what no table lists: a spiral, a
radial acquisition, an arm optimised per shot. It reads every gradient rather
than one nominated in advance, so a spoiler or an encode train that lands in a
band is caught even in a family whose readout never would. It judges per
physical channel and per prescription. And because the rule is the drive
itself rather than a parameter standing in for it, a prescription that cannot
physically reach the threshold is not refused for the sake of its number.
