#!/usr/bin/env node
// Size one training job with the product's own plan arithmetic.
//
// `/api/v1/plan` answers at its default run (batch 32, priced on an A10G), and the
// question a pull request has to answer is a different one: does THIS job, at the
// batch and sequence length it actually trains at, still fit the card it reserves?
// The arithmetic to answer it is the vendored plan bundle (`vendor/modelPlan.bundle.mjs`,
// built from neurarch `lib/modelPlan.bundle.mjs` by `npm run build:plan`), run here
// on the traced graph. Nothing in this file computes a memory number of its own:
// the peak, the activation term, the smallest card and the cost come from
// `buildPlan`, and the largest batch from `maxBatchFor`, the same two functions the
// hosted plan, the CLI and neurarch-mcp call. What is local is the card table the
// bundle does not export, and that is pinned by a test against the bundle's own
// `smallestFitGpu` answers.
//
// Usage: node size_job.mjs request.json  ->  JSON on stdout.
//   request: { head: path, base?: path, job: Job, base_job?: Job, history?: rows }
//   Job:     { gpu?, batch?, seq_len?, attention?, epochs?, samples? }

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const bundlePath = process.env.NEURARCH_PLAN_BUNDLE || path.join(here, 'vendor', 'modelPlan.bundle.mjs');
const P = await import(pathToFileURL(bundlePath).href);

// The cards the plan prices and sizes, in the order `smallestFitGpu` walks them.
// Mirrors PREFLIGHT_GPUS in neurarch src/utils/preflightReport.ts; the bundle does
// not export it, and test_job.py fails if a name here stops matching the bundle.
export const CARDS = [
  { name: 'T4 (16GB)', gb: 16 },
  { name: 'L4 (24GB)', gb: 24 },
  { name: 'A10G (24GB)', gb: 24 },
  { name: 'L40S (48GB)', gb: 48 },
  { name: 'A100 (40GB)', gb: 40 },
  { name: 'A100 (80GB)', gb: 80 },
  { name: 'H100 (80GB)', gb: 80 },
];
export const DEFAULT_BATCH = 32;

const norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, '');

/** "A10G", "a10g-24gb", "A100 80GB", "A10G (24GB)" -> the card; null when nothing matches. */
export function matchCard(input) {
  if (input == null || input === '') return null;
  const key = norm(input);
  for (const c of CARDS) {
    const model = norm(c.name.split(' ')[0]);
    const gb = String(c.gb);
    if (key === norm(c.name) || key === model + gb + 'gb' || key === model + gb) return c;
  }
  // The bare model name: the first card of that name, which for A100 is the 40GB one.
  return CARDS.find((c) => norm(c.name.split(' ')[0]) === key) ?? null;
}

function cardOf(name) {
  return CARDS.find((c) => c.name === name) ?? null;
}

function seqOption(job) {
  const n = Number(job?.seq_len);
  return Number.isInteger(n) && n > 0 ? n : 'default';
}

/** One graph at one job: the peak, what it is made of, and the answers that read it backwards. */
export function size(graph, job, card) {
  const batch = Number.isInteger(Number(job?.batch)) && Number(job.batch) > 0 ? Number(job.batch) : DEFAULT_BATCH;
  const attention = P.readAttention(job?.attention) ?? P.DEFAULT_ATTENTION;
  const seqLen = seqOption(job);
  const samples = Number(job?.samples) > 0 ? Number(job.samples) : null;
  const epochs = Number(job?.epochs) > 0 ? Math.round(Number(job.epochs)) : undefined;
  const opts = (gpu) => ({
    attention,
    seqLen,
    run: { batchSize: batch, ...(gpu ? { gpu } : {}), ...(epochs ? { epochs } : {}) },
    ...(samples ? { workload: { samples, source: '.neurarch.yml job.samples' } } : {}),
  });
  const plan = P.buildPlan(graph, opts(card?.name));
  const c = plan.cost;
  const params = plan.model.params;
  const act = c.activationGb;
  const footprintGb = act != null ? c.peakGb - act : c.peakGb;
  const perSample = act != null ? act / batch : null;
  const fits = card && params > 0 ? c.peakGb * P.FIT_HEADROOM <= card.gb : null;
  const maxBatch = card && params > 0 ? P.maxBatchFor({ footprintGb, activationPerSampleGb: perSample }, card.gb) : null;
  const smallest = c.smallestFitGpu ? cardOf(c.smallestFitGpu) : null;
  // Priced on the reserved card when the job fits it; otherwise on the card that would
  // hold it, because a price for a run that cannot start on its card is not a price.
  let cost = { usd: c.estCostUsd, sec: c.estTrainSec, card: c.fitsGpu };
  if (card && fits === false && smallest) {
    const alt = P.buildPlan(graph, opts(smallest.name)).cost;
    cost = { usd: alt.estCostUsd, sec: alt.estTrainSec, card: alt.fitsGpu };
  }
  return {
    params,
    legal: plan.run.legal,
    batch,
    seqLen: c.seqLen,
    attention,
    peakGb: c.peakGb,
    footprintGb,
    activationGb: act,
    needGb: c.peakGb * P.FIT_HEADROOM,
    fits,
    maxBatch,
    smallestFitGpu: smallest ? smallest.name : null,
    smallestFitGb: smallest ? smallest.gb : null,
    twoCardSplit: c.twoCardSplit ? { gpu: c.twoCardSplit.gpu, fits: c.twoCardSplit.fits } : null,
    costUsd: cost.usd,
    trainSec: cost.sec,
    costCard: cost.card,
    epochs: epochs ?? 10,
    samples: samples ?? 50000,
    samplesAssumed: samples == null,
  };
}

export function sizeRequest(req) {
  const read = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));
  const job = req.job ?? {};
  const card = matchCard(job.gpu);
  const cardError = job.gpu != null && job.gpu !== '' && !card
    ? `\`${job.gpu}\` is not a card the plan knows; use one of ${CARDS.map((c) => c.name.split(' ')[0] + ' ' + c.gb + 'GB').join(', ')}`
    : null;
  const head = size(read(req.head), job, card);
  let base = null;
  if (req.base) {
    const baseJob = req.base_job ?? job;
    // The base is held to the card the HEAD reserves: the question is whether this change
    // moved the job off its card, so both sides are read against the same line.
    base = size(read(req.base), baseJob, card);
  }
  const rows = Array.isArray(req.history) ? req.history : [];
  let historyLine = null;
  if (rows.length) {
    try {
      const more = rows.length - 1;
      historyLine = P.historySentence(rows[0]) + (more > 0 ? ` (+${more} earlier run${more === 1 ? '' : 's'} on this structure)` : '');
    } catch { historyLine = null; }
  }
  return { ok: true, headroom: P.FIT_HEADROOM, card, cardError, head, base, baseJob: req.base ? (req.base_job ?? job) : null, historyLine };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const req = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
    process.stdout.write(JSON.stringify(sizeRequest(req)));
  } catch (e) {
    process.stdout.write(JSON.stringify({ ok: false, error: e instanceof Error ? e.message : String(e) }));
    process.exitCode = 0;
  }
}
