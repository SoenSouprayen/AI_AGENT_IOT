"use strict";

const byId = (name) => document.getElementById(name);
const number = (value, digits = 1) => new Intl.NumberFormat("fr-FR", {maximumFractionDigits: digits}).format(value);
let configuration;
let generation = 0;
let predictionTimer;
let predictionController;
let explanationInFlight = false;
let activeResult = null;

function payload() {
  return {position_deg: Number(byId("position").value), N: Number(byId("load").value), seed: Number(byId("scenario").value)};
}

function updateScenarios() {
  const previous = byId("scenario").value;
  const load = Number(byId("load").value);
  byId("scenario").replaceChildren();
  const available = configuration.scenarios.filter((scenario) => scenario.loads.includes(load));
  for (const [index, scenario] of available.entries()) byId("scenario").add(new Option(`Scénario ${index + 1} · rép. ${scenario.replication}`, String(scenario.seed)));
  if (available.some((scenario) => String(scenario.seed) === previous)) byId("scenario").value = previous;
}

function updateQuestionControls() {
  const enabled = Boolean(activeResult) && !explanationInFlight;
  byId("question-input").disabled = !activeResult;
  byId("question-send").disabled = !enabled;
  for (const button of document.querySelectorAll(".question-suggestion")) button.disabled = !enabled;
}

function appendMessage(role, text) {
  const message = document.createElement("div");
  message.className = `question-message ${role === "user" ? "question-user" : "question-agent"}`;
  const author = document.createElement("strong");
  author.textContent = role === "user" ? "Vous" : "Agent · Ollama";
  const content = document.createElement("p");
  content.textContent = text;
  message.append(author, content);
  const history = byId("question-history");
  const welcome = history.querySelector(".question-welcome");
  if (welcome) welcome.remove();
  history.append(message);
  while (history.children.length > 12) history.firstElementChild.remove();
  history.scrollTop = history.scrollHeight;
}

function renderOrbit() {
  const position = Number(byId("position").value);
  const radians = position * Math.PI / 180;
  const horizontal = 380 - 300 * Math.cos(radians);
  const vertical = 337 - 280 * Math.sin(radians);
  const elevation = Math.min(position, 180 - position);
  byId("satellite").setAttribute("transform", `translate(${horizontal} ${vertical})`);
  byId("satellite").setAttribute("aria-valuenow", String(position));
  byId("satellite").setAttribute("aria-valuetext", `${elevation} degrés d’élévation, côté ${position <= 90 ? "gauche" : "droit"}`);
  byId("beam").setAttribute("x2", String(horizontal));
  byId("beam").setAttribute("y2", String(vertical));
  byId("beam-area").setAttribute("d", `M370 337 L${horizontal} ${vertical} L390 337Z`);
  const direction = position <= 90 ? -1 : 1;
  const angle = Math.atan2(337 - vertical, Math.abs(horizontal - 380));
  const arcEndX = 380 + direction * 48 * Math.cos(angle);
  const arcEndY = 337 - 48 * Math.sin(angle);
  byId("angle-arc").setAttribute("d", `M${380 + direction * 48} 337 A48 48 0 0 ${direction < 0 ? 1 : 0} ${arcEndX} ${arcEndY}`);
  byId("angle-label").setAttribute("x", String(380 + direction * 66));
  byId("angle-label").setAttribute("text-anchor", "middle");
  byId("angle-label").textContent = `${elevation}°`;
  byId("requested-angle").textContent = `${elevation}° d’élévation`;
}

function clearResult() {
  activeResult = null;
  document.querySelector(".result-card").classList.add("pending");
  byId("prediction-status").textContent = "Calcul…";
  byId("best-mode").textContent = "—";
  byId("best-per").textContent = "—";
  byId("family").textContent = "Nouveau scénario";
  for (const name of ["used-angle", "distance", "velocity", "doppler"]) byId(name).textContent = "—";
  for (const row of document.querySelectorAll(".mode-row")) {
    row.querySelector(".mode-value").textContent = "—";
    row.querySelector(".bar-fill").style.width = "0%";
    row.classList.remove("best");
  }
  byId("explain").disabled = true;
  byId("explanation").textContent = "Une fois le scénario choisi, demandez au LLM d’expliquer la recommandation.";
  byId("llm-status").textContent = explanationInFlight ? "L’explication précédente est encore en cours ; elle ne sera pas affichée pour ce scénario." : "";
  byId("error").hidden = true;
  byId("question-history").replaceChildren();
  byId("question-context").textContent = "Sélection d’un nouveau scénario…";
  byId("question-status").textContent = explanationInFlight ? "L’ancienne réponse en cours sera ignorée." : "";
  updateQuestionControls();
}

async function request(path, data, signal) {
  const response = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data), signal});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Erreur du serveur local.");
  return result;
}

function displayResult(result) {
  const decision = result.decision;
  const geometry = result.scenario.geometry;
  activeResult = result;
  document.querySelector(".result-card").classList.remove("pending");
  byId("prediction-status").textContent = "Prédiction prête";
  byId("best-mode").textContent = decision.best_mode;
  byId("best-per").textContent = `${number(decision.predicted_per * 100, 2)} %`;
  byId("family").textContent = decision.best_mode.startsWith("DR") ? "LR-FHSS" : "LoRa CSS";
  for (const mode of configuration.modes) {
    const row = byId(`mode-${mode}`);
    row.classList.toggle("best", decision.tied_modes.includes(mode));
    const percent = decision.per_by_mode[mode] * 100;
    row.querySelector(".bar-fill").style.width = `${percent}%`;
    row.querySelector(".mode-value").textContent = `${number(percent, 1)} %`;
  }
  byId("used-angle").textContent = `${number(geometry.elevation_mean_deg, 2)}°`;
  byId("distance").textContent = `${number(geometry.distance_mean_km, 0)} km`;
  byId("velocity").textContent = `${number(geometry.relative_velocity_mean_ms / 1000, 2)} km/s`;
  byId("doppler").textContent = `${number(geometry.doppler_shift_mean_hz, 0)} Hz`;
  byId("source-note").textContent = `Source : ${result.input_source.dataset}, seed ${result.input_source.seed}, réplication ${result.input_source.replication}. ${result.input_source.note}`;
  byId("scientific-limitation").textContent = result.limitation;
  byId("explain").disabled = explanationInFlight;
  byId("question-context").textContent = `N = ${number(result.scenario.N, 0)} · élévation utilisée : ${number(geometry.elevation_mean_deg, 2)}° · mode recommandé : ${decision.best_mode}`;
  updateQuestionControls();
}

function schedulePrediction() {
  renderOrbit();
  generation += 1;
  const version = generation;
  clearResult();
  clearTimeout(predictionTimer);
  if (predictionController) predictionController.abort();
  predictionTimer = setTimeout(async () => {
    predictionController = new AbortController();
    try {
      const result = await request("/api/predict", payload(), predictionController.signal);
      if (version === generation) displayResult(result);
    } catch (error) {
      if (error.name === "AbortError" || version !== generation) return;
      byId("prediction-status").textContent = "Indisponible";
      byId("error").textContent = error.message;
      byId("error").hidden = false;
    }
  }, 140);
}

async function explain(question = null) {
  if (!activeResult || explanationInFlight) return;
  const version = generation;
  explanationInFlight = true;
  byId("explain").disabled = true;
  updateQuestionControls();
  if (question !== null) {
    appendMessage("user", question);
    byId("question-input").value = "";
    byId("question-status").textContent = "Ollama prépare sa réponse…";
  }
  byId("llm-status").textContent = "Ollama prépare son explication… Cela peut prendre quelques secondes.";
  try {
    const result = await request(question === null ? "/api/explain" : "/api/ask", {
      ...payload(), ...(question === null ? {} : {question}),
    });
    if (version !== generation) return;
    if (result.llm_status === "generated") {
      if (question === null) byId("explanation").textContent = result.explanation;
      else appendMessage("assistant", result.explanation);
      byId("question-status").textContent = question === null ? "" : "Réponse reçue · texte non vérifié automatiquement.";
      byId("llm-status").textContent = "Texte généré par le LLM, non vérifié automatiquement. La décision reste celle de Python.";
    } else {
      byId("explanation").textContent = "La décision Python est disponible, mais Ollama n’a pas pu produire l’explication.";
      byId("llm-status").textContent = result.llm_error || "Démarrez Ollama puis réessayez.";
      if (question !== null) {
        appendMessage("assistant", "Ollama est indisponible. La décision Python reste affichée. Démarrez Ollama puis renvoyez votre question.");
        byId("question-status").textContent = "Aucune réponse du LLM. Vous pouvez réessayer.";
        if (!byId("question-input").value) byId("question-input").value = question;
      }
    }
  } catch (error) {
    if (version === generation) {
      byId("llm-status").textContent = error.message;
      if (question !== null) {
        byId("question-status").textContent = error.message;
        if (!byId("question-input").value) byId("question-input").value = question;
      }
    }
  } finally {
    explanationInFlight = false;
    byId("explain").disabled = !activeResult;
    updateQuestionControls();
    if (version !== generation) byId("llm-status").textContent = "Scénario modifié : vous pouvez demander une nouvelle explication.";
    if (version !== generation) byId("question-status").textContent = "Scénario modifié : l’ancienne réponse a été ignorée.";
  }
}

function moveFromPointer(event) {
  const svg = byId("orbit");
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  const local = point.matrixTransform(svg.getScreenCTM().inverse());
  const angle = Math.atan2(Math.max(0, (337 - local.y) / 280), (380 - local.x) / 300) * 180 / Math.PI;
  byId("position").value = String(Math.round(Math.max(5, Math.min(175, angle))));
  schedulePrediction();
}

async function start() {
  const response = await fetch("/api/config");
  if (!response.ok) throw new Error("Impossible de charger les scénarios.");
  configuration = await response.json();
  for (const load of configuration.loads) byId("load").add(new Option(number(load, 0), String(load)));
  updateScenarios();
  byId("load").disabled = false;
  byId("scenario").disabled = false;
  byId("dataset-name").textContent = `${configuration.dataset} · ${number(configuration.training_rows, 0)} lignes d’entraînement`;
  byId("llm-model").textContent = `Ollama · ${configuration.llm_model}`;
  for (const mode of configuration.modes) {
    const row = document.createElement("div");
    row.className = "mode-row";
    row.id = `mode-${mode}`;
    const name = document.createElement("span");
    name.className = "mode-name";
    name.textContent = mode;
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    track.append(fill);
    const value = document.createElement("span");
    value.className = "mode-value";
    value.textContent = "—";
    row.append(name, track, value);
    byId("mode-bars").append(row);
  }
  for (const name of ["position", "scenario"]) byId(name).addEventListener("input", schedulePrediction);
  byId("load").addEventListener("input", () => { updateScenarios(); schedulePrediction(); });
  byId("reset").addEventListener("click", () => {
    byId("position").value = "45";
    byId("load").selectedIndex = 0;
    updateScenarios();
    byId("scenario").selectedIndex = 0;
    schedulePrediction();
  });
  byId("explain").addEventListener("click", () => explain());
  byId("question-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const question = byId("question-input").value.trim();
    if (!question || !activeResult || explanationInFlight) return;
    explain(question);
  });
  byId("question-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      byId("question-form").requestSubmit();
    }
  });
  for (const button of document.querySelectorAll(".question-suggestion")) button.addEventListener("click", () => {
    if (activeResult && !explanationInFlight) explain(button.textContent);
  });
  const satellite = byId("satellite");
  let dragging = false;
  satellite.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    dragging = true;
    satellite.classList.add("dragging");
    satellite.setPointerCapture(event.pointerId);
    satellite.focus();
  });
  satellite.addEventListener("pointermove", (event) => { if (dragging) moveFromPointer(event); });
  function release() { dragging = false; satellite.classList.remove("dragging"); }
  satellite.addEventListener("pointerup", release);
  satellite.addEventListener("pointercancel", release);
  satellite.addEventListener("lostpointercapture", release);
  satellite.addEventListener("keydown", (event) => {
    const increments = {ArrowLeft: -2, ArrowRight: 2, ArrowUp: 2, ArrowDown: -2};
    if (!(event.key in increments)) return;
    event.preventDefault();
    byId("position").value = String(Math.max(5, Math.min(175, Number(byId("position").value) + increments[event.key])));
    schedulePrediction();
  });
  schedulePrediction();
}

start().catch((error) => {
  byId("error").textContent = error.message;
  byId("error").hidden = false;
  byId("prediction-status").textContent = "Erreur de chargement";
});
