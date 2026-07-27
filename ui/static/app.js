"use strict";
const $ = (id) => document.getElementById(id);
const api = (p, o) => fetch(p, o).then(async (r) => {
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || j.reason || r.statusText);
  return j;
});
function toast(msg, isErr) {
  const t = $("toast"); t.textContent = msg; t.className = "toast" + (isErr ? " err" : "");
  t.hidden = false; clearTimeout(t._t); t._t = setTimeout(() => (t.hidden = true), 3200);
}

/* ── Tabs ─────────────────────────────────────────────── */
document.querySelectorAll(".tab").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $(b.dataset.tab).classList.add("active");
    if (b.dataset.tab === "data") loadViz();
  })
);

/* ── Status polling (uptime + model + pending) ───────────── */
async function poll() {
  try {
    const s = await api("/api/status");
    $("svc").textContent = "up"; $("pill-service").classList.add("up");
    $("uptime").textContent = s.uptime.human;
    $("modelver").textContent = s.model_ready ? (s.model.version || "ready") : "not trained";
    $("pend-benign").textContent = s.pending_uploads.benign || 0;
    $("pend-malignant").textContent = s.pending_uploads.malignant || 0;
    $("pend-total").textContent = s.pending_total;
    $("thresh").textContent = s.retrain_threshold;
    reflectJob(s.retrain_job);
  } catch (e) {
    $("svc").textContent = "down"; $("pill-service").classList.remove("up");
  }
}

/* ── Overview metrics ────────────────────────────────────── */
async function loadMetrics() {
  try {
    const m = await api("/api/metrics");
    const f = (x) => (x * 100).toFixed(1) + "%";
    $("m-acc").textContent = f(m.accuracy); $("m-prec").textContent = f(m.precision);
    $("m-rec").textContent = f(m.recall); $("m-f1").textContent = f(m.f1);
    $("m-auc").textContent = m.roc_auc.toFixed(3); $("m-n").textContent = m.n_test;
    if (m.trained_at) $("trained-at").textContent = "Last trained: " + m.trained_at + " · version " + (m.version || "—");
    const bust = "?t=" + Date.now();
    $("fig-cm").src = "/reports/confusion_matrix.png" + bust;
    $("fig-roc").src = "/reports/roc_curve.png" + bust;
  } catch (e) { /* not trained yet */ }
}

/* ── Visualizations (Chart.js) ───────────────────────────── */
let charts = {};
async function loadViz() {
  const v = await api("/api/visualizations");
  const mk = (id, cfg) => { if (charts[id]) charts[id].destroy(); charts[id] = new Chart($(id), cfg); };
  const CN = ["benign", "malignant"];

  // Feature 1: class balance
  mk("chart-dist", {
    type: "bar",
    data: { labels: CN, datasets: [
      { label: "train", data: CN.map((c) => v.class_distribution.train[c] || 0), backgroundColor: "#0f9d8f" },
      { label: "test", data: CN.map((c) => v.class_distribution.test[c] || 0), backgroundColor: "#9dd8d1" }]},
    options: { plugins: { legend: { position: "bottom" } }, scales: { y: { beginAtZero: true } } },
  });
  const tr = v.class_distribution.train;
  $("interp-dist").textContent =
    `The dataset is imbalanced — benign (${tr.benign}) outnumbers malignant (${tr.malignant}) roughly ${(tr.benign / Math.max(tr.malignant,1)).toFixed(1)}:1. ` +
    `That is why training weights the malignant class more heavily, so the model does not simply learn to say "benign".`;

  // Feature 2: mean intensity by class
  const inten = v.intensity_by_class || {};
  mk("chart-intensity", {
    type: "bar",
    data: { labels: CN, datasets: [{ label: "mean grayscale intensity", data: CN.map((c) => inten[c] || 0),
      backgroundColor: ["#2f9e6f", "#e0654b"] }]},
    options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, max: 255 } } },
  });
  const diff = (inten.benign || 0) - (inten.malignant || 0);
  $("interp-intensity").textContent =
    `Average brightness differs by class (benign ${(inten.benign||0).toFixed(0)} vs malignant ${(inten.malignant||0).toFixed(0)}). ` +
    `Malignant lesions tend to appear ${diff > 0 ? "darker (more hypoechoic)" : "different in echogenicity"}, ` +
    `a genuine ultrasound cue the CNN can exploit beyond simple shape.`;

  // Feature 3: image size
  const s = v.image_size_stats || {};
  mk("chart-size", {
    type: "bar",
    data: { labels: ["min", "mean", "max"], datasets: [
      { label: "width", data: [s.min_w, s.mean_w, s.max_w], backgroundColor: "#0f9d8f" },
      { label: "height", data: [s.min_h, s.mean_h, s.max_h], backgroundColor: "#e0a24b" }]},
    options: { plugins: { legend: { position: "bottom" } }, scales: { y: { beginAtZero: true } } },
  });
  $("interp-size").textContent =
    `Raw images range from ${s.min_w}×${s.min_h} to ${s.max_w}×${s.max_h} px (mean ${Math.round(s.mean_w||0)}×${Math.round(s.mean_h||0)}). ` +
    `Because sizes vary, every image is resized to 224×224 before entering MobileNetV2 — the story: preprocessing must standardise geometry.`;
}

/* ── Single prediction ───────────────────────────────────── */
const dz = $("drop-predict"), pf = $("predict-file");
dz.addEventListener("click", (e) => { if (e.target.id !== "predict-btn") pf.click(); });
["dragover", "dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.toggle("drag", ev === "dragover"); }));
dz.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) setPredictFile(e.dataTransfer.files[0]); });
pf.addEventListener("change", () => pf.files[0] && setPredictFile(pf.files[0]));
let predictBlob = null;
function setPredictFile(file) {
  predictBlob = file;
  const url = URL.createObjectURL(file);
  const img = $("predict-preview"); img.src = url; img.hidden = false;
  $("predict-hint").textContent = file.name;
  $("predict-btn").disabled = false;
}
$("predict-btn").addEventListener("click", async () => {
  if (!predictBlob) return;
  $("predict-btn").disabled = true; $("predict-btn").textContent = "Classifying…";
  try {
    const fd = new FormData(); fd.append("file", predictBlob);
    const r = await api("/api/predict", { method: "POST", body: fd });
    showPrediction(r);
  } catch (e) { toast(e.message, true); }
  $("predict-btn").disabled = false; $("predict-btn").textContent = "Classify image";
});
function showPrediction(r) {
  $("predict-result").hidden = false;
  const v = $("verdict"); v.textContent = r.prediction.toUpperCase();
  v.className = "verdict " + r.prediction;
  const pb = r.probabilities.benign * 100, pm = r.probabilities.malignant * 100;
  $("bar-benign").style.width = pb + "%"; $("pct-benign").textContent = pb.toFixed(1) + "%";
  $("bar-malignant").style.width = pm + "%"; $("pct-malignant").textContent = pm.toFixed(1) + "%";
  $("confidence").textContent = (r.confidence * 100).toFixed(1) + "%";
  $("pred-ver").textContent = r.model_version;
}

/* ── Bulk upload ─────────────────────────────────────────── */
$("upload-btn").addEventListener("click", async () => {
  const files = $("upload-files").files;
  if (!files.length) return toast("Pick one or more images first.", true);
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  fd.append("label", $("upload-label").value);
  $("upload-btn").disabled = true;
  try {
    const r = await api("/api/upload", { method: "POST", body: fd });
    $("upload-msg").textContent = `Uploaded ${r.saved} image(s) as ${r.label}. Pending total: ${r.pending_total}.`;
    if (r.retrain_recommended) toast("Enough new data — retraining is recommended.");
    poll();
  } catch (e) { toast(e.message, true); }
  $("upload-btn").disabled = false;
});

/* ── Retrain trigger + polling ───────────────────────────── */
let jobTimer = null;
$("retrain-btn").addEventListener("click", async () => {
  try {
    const fd = new FormData();
    await api("/api/retrain", { method: "POST", body: fd });
    toast("Retraining started…");
    $("job-box").hidden = false;
    if (!jobTimer) jobTimer = setInterval(pollJob, 2500);
  } catch (e) { toast(e.message, true); }
});
async function pollJob() {
  try { reflectJob(await api("/api/retrain/status")); } catch (e) {}
}
function reflectJob(j) {
  if (!j || j.status === "idle") return;
  $("job-box").hidden = false;
  $("job-status").textContent = j.status;
  $("job-msg").textContent = j.message || "";
  $("job-spin").style.visibility = j.status === "running" ? "visible" : "hidden";
  if (j.status === "completed" || j.status === "failed") {
    if (jobTimer) { clearInterval(jobTimer); jobTimer = null; }
    if (j.metrics) {
      $("job-metrics").hidden = false;
      $("job-metrics").textContent = JSON.stringify(j.metrics, null, 2);
    }
    if (j.status === "completed") { toast("Retrain complete — model updated."); loadMetrics(); }
    else toast("Retrain failed. See status.", true);
  }
}

/* ── Boot ────────────────────────────────────────────────── */
poll(); loadMetrics();
setInterval(poll, 5000);
