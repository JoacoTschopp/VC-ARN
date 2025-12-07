import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm";

const metricSvg = d3.select("#metric-chart");
const lossSvg = d3.select("#loss-chart");
const metricLegendEl = document.getElementById("metric-legend");
const lossLegendEl = document.getElementById("loss-legend");
const slider = document.getElementById("epoch");
const output = document.getElementById("epochValue");
const playBtn = document.getElementById("play");

let history = [];
let lossHistory = [];
let maxEpoch = 1;
let timer = null;
let experimentNames = [];
let colorScale = null;
let bestPoints = [];
let bestByExperiment = new Map();

async function loadData() {
  const res = await fetch("experiments_log.jsonl");
  const text = await res.text();

  const experiments = text
    .trim()
    .split(/\n+/)
    .map(line =>
      JSON.parse(line.replace(/\bInfinity\b/g, "null"))
    );

  history = experiments.flatMap(exp =>
    (exp.results?.val_metrics ?? []).map((gain, idx) => ({
      experiment_name: exp.experiment_name,
      model_class: exp.model_class,
      epoch: idx + 1,
      val_metric: gain
    }))
  );

  lossHistory = experiments.flatMap(exp =>
    (exp.results?.val_losses ?? []).map((loss, idx) => ({
      experiment_name: exp.experiment_name,
      model_class: exp.model_class,
      epoch: idx + 1,
      val_loss: loss
    }))
  );

  experimentNames = Array.from(new Set(history.map(d => d.experiment_name)));
  colorScale = d3.scaleOrdinal(d3.schemeTableau10).domain(experimentNames);
  bestPoints = experimentNames.map(name => {
    const entries = history.filter(d => d.experiment_name === name);
    return entries.reduce((best, current) =>
      !best || current.val_metric > best.val_metric ? current : best,
    null);
  }).filter(Boolean);
  bestByExperiment = new Map(bestPoints.map(point => [point.experiment_name, point]));

  maxEpoch = Math.max(
    d3.max(history, d => d.epoch) ?? 1,
    d3.max(lossHistory, d => d.epoch) ?? 1
  );
  slider.max = maxEpoch;
  renderCharts(1);
}

function renderCharts(currentEpoch) {
  output.value = currentEpoch;

  const partialHistory = history.filter(d => d.epoch <= currentEpoch);
  const partialLoss = lossHistory.filter(d => d.epoch <= currentEpoch);
  const currentSlice = partialHistory.filter(d => d.epoch === currentEpoch);
  const currentLossSlice = partialLoss.filter(d => d.epoch === currentEpoch);
  const visibleBest = bestPoints.filter(d => d.epoch <= currentEpoch);

  renderMetricChart(partialHistory, currentSlice, visibleBest);
  renderLossChart(partialLoss, currentLossSlice);
}

function renderMetricChart(partialHistory, currentSlice, visibleBest) {
  metricSvg.selectAll("*").remove();
  metricLegendEl.innerHTML = "";

  const yMax = d3.max(history, d => d.val_metric) ?? 1;

  const plot = Plot.plot({
    width: metricSvg.node().clientWidth || 960,
    height: 480,
    marginRight: 40,
    marginBottom: 60,
    marginLeft: 70,
    x: {
      label: "Época",
      labelAnchor: "center",
      labelArrow: null,
      domain: [0, maxEpoch]
    },
    y: {
      label: "Ganancia (val_metric)",
      labelAnchor: "center",
      labelArrow: null,
      domain: [0, yMax * 1.05]
    },
    color: {
      label: "Experimento",
      domain: experimentNames,
      range: experimentNames.map(name => colorScale(name))
    },
    marks: [
      Plot.ruleY([0]),
      Plot.ruleX([0], { stroke: "#444", strokeWidth: 1.2 }),
      Plot.line(partialHistory, { x: "epoch", y: "val_metric", stroke: "experiment_name", z: "experiment_name" }),
      Plot.dot(currentSlice, { x: "epoch", y: "val_metric", fill: "experiment_name", r: 5, z: null }),
      Plot.dot(visibleBest, { x: "epoch", y: "val_metric", fill: "experiment_name", stroke: "experiment_name", r: 6, symbol: "triangle", opacity: 0.9, z: null }),
      Plot.text(visibleBest, {
        x: "epoch",
        y: d => d.val_metric + 0.015,
        text: d => d.val_metric.toFixed(4),
        fill: "experiment_name",
        fontSize: 10,
        dy: -4,
        z: null
      }),
      Plot.text(currentSlice, {
        x: "epoch",
        y: d => d.val_metric + 0.01,
        text: d => d.val_metric.toFixed(4),
        fill: "experiment_name",
        dy: -6
      })
    ]
  });

  metricSvg.node().appendChild(plot);
  renderLegend(metricLegendEl, "Ganancia (val_metric)");
}

function renderLossChart(partialLoss, currentLossSlice) {
  lossSvg.selectAll("*").remove();
  lossLegendEl.innerHTML = "";

  const yMax = d3.max(lossHistory, d => d.val_loss) ?? 1;
  const latestLossPoints = experimentNames
    .map(name => {
      const entries = partialLoss.filter(d => d.experiment_name === name);
      return entries.length ? entries[entries.length - 1] : null;
    })
    .filter(Boolean);

  const plot = Plot.plot({
    width: lossSvg.node().clientWidth || 960,
    height: 480,
    marginRight: 40,
    marginBottom: 60,
    marginLeft: 70,
    x: {
      label: "Época",
      labelAnchor: "center",
      labelArrow: null,
      domain: [0, maxEpoch]
    },
    y: {
      label: "Pérdida (val_loss)",
      labelAnchor: "center",
      labelArrow: null,
      domain: [0, yMax * 1.05]
    },
    color: {
      label: "Experimento",
      domain: experimentNames,
      range: experimentNames.map(name => colorScale(name))
    },
    marks: [
      Plot.ruleY([0]),
      Plot.ruleX([0], { stroke: "#444", strokeWidth: 1.2 }),
      Plot.line(partialLoss, {
        x: "epoch",
        y: "val_loss",
        stroke: "experiment_name",
        z: d => `${d.experiment_name}-loss`,
        strokeDasharray: "4 3"
      }),
      Plot.dot(currentLossSlice, { x: "epoch", y: "val_loss", fill: "experiment_name", r: 4, z: null }),
      Plot.text(latestLossPoints, {
        x: "epoch",
        y: d => d.val_loss + 0.02,
        text: d => d.val_loss.toFixed(4),
        fill: "experiment_name",
        fontSize: 10,
        dy: -8,
        z: null
      }),
      Plot.dot(latestLossPoints, { x: "epoch", y: "val_loss", fill: "experiment_name", stroke: "experiment_name", r: 6, symbol: "triangle", rotation: 180, opacity: 0.9, z: null })
    ]
  });

  lossSvg.node().appendChild(plot);
  renderLegend(lossLegendEl, "Pérdida (val_loss)", true);
}

function renderLegend(container, titleText, dashed = false) {
  if (!colorScale) return;
  container.innerHTML = "";

  const sortedExperiments = [...experimentNames].sort((a, b) => {
    const bestB = bestByExperiment.get(b)?.val_metric ?? -Infinity;
    const bestA = bestByExperiment.get(a)?.val_metric ?? -Infinity;
    return bestB - bestA;
  });

  const section = document.createElement("div");
  section.className = "legend-section";
  const title = document.createElement("span");
  title.className = "legend-title";
  title.textContent = titleText;
  const items = document.createElement("div");
  items.className = "legend-items";

  sortedExperiments.forEach(exp => {
    const item = document.createElement("div");
    item.className = "legend-item";

    if (dashed) {
      const line = document.createElement("span");
      line.className = "legend-line";
      line.style.color = colorScale(exp);
      item.appendChild(line);
    } else {
      const swatch = document.createElement("span");
      swatch.className = "legend-swatch";
      swatch.style.backgroundColor = colorScale(exp);
      item.appendChild(swatch);
    }

    const label = document.createElement("span");
    label.textContent = exp;
    item.appendChild(label);
    items.appendChild(item);
  });

  section.appendChild(title);
  section.appendChild(items);
  container.appendChild(section);
}

slider.addEventListener("input", e => {
  renderCharts(Number(e.target.value));
});

playBtn.addEventListener("click", () => {
  if (timer) {
    clearInterval(timer);
    timer = null;
    playBtn.textContent = "▶︎ Play";
    return;
  }
  playBtn.textContent = "❚❚ Pause";
  timer = setInterval(() => {
    let next = Number(slider.value) + 1;
    if (next > maxEpoch) {
      clearInterval(timer);
      timer = null;
      playBtn.textContent = "▶︎ Play";
      slider.value = maxEpoch;
      renderCharts(maxEpoch);
      return;
    }
    slider.value = next;
    renderCharts(next);
  }, 600);
});

loadData();