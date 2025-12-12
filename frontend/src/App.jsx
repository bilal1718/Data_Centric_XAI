import { useEffect, useMemo, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from "chart.js";
import { Line, Bar } from "react-chartjs-2";
import ReactFlow, { Background, Controls, MiniMap } from "reactflow";
import "reactflow/dist/style.css";
import "./App.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const DATASETS = [
  { label: "MNIST", value: "mnist" },
  { label: "Fashion-MNIST", value: "fashion" },
  { label: "CIFAR-10", value: "cifar10" },
];

const MODELS = [
  { label: "MobileNetV2", value: "mobilenet_v2" },
  { label: "Efficient CNN", value: "efficient_cnn" },
  { label: "ResNet18", value: "resnet18" },
];

const METHODS = [
  { label: "Grad-CAM", value: "gradcam" },
  { label: "SHAP", value: "shap" },
  { label: "LIME", value: "lime" },
];

function Sidebar({ active, onChange }) {
  const items = [
    { key: "data", label: "Data & Training" },
    { key: "evaluation", label: "Model Evaluation" },
    { key: "playground", label: "XAI Playground" },
  ];
  return (
    <aside className="sidebar">
      <div className="brand">XAI Studio</div>
      <nav>
        {items.map((item) => (
          <button
            key={item.key}
            className={`nav-item ${active === item.key ? "active" : ""}`}
            onClick={() => onChange(item.key)}
          >
            {item.label}
          </button>
        ))}
      </nav>
    </aside>
  );
}

function ConfusionMatrixHeatmap({ matrix, classes, metrics }) {
  if (!matrix || !classes) return <p className="muted">Loading confusion matrix...</p>;
  const maxVal = Math.max(...matrix.flat());
  const datasets = matrix.map((row, rowIdx) => ({
    label: `True: ${classes[rowIdx]}`,
    data: row,
    backgroundColor: row.map((val) => {
      const intensity = maxVal > 0 ? val / maxVal : 0;
      return `rgba(59, 130, 246, ${0.35 + intensity * 0.6})`;
    }),
  }));
  const chartData = { labels: classes, datasets };
  return (
    <div style={{ display: "grid", gap: "8px" }}>
      {metrics?.accuracy !== undefined && (
        <div className="muted">Accuracy: {(metrics.accuracy * 100).toFixed(2)}%</div>
      )}
      <div style={{ height: "320px", position: "relative" }}>
        <Bar
          data={chartData}
          options={{
            indexAxis: "y",
            scales: {
              x: { title: { display: true, text: "Predicted", color: "#94a3b8" }, ticks: { color: "#94a3b8" } },
              y: { title: { display: true, text: "True", color: "#94a3b8" }, ticks: { color: "#94a3b8" } },
            },
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  title: () => "",
                  label: (context) => {
                    const r = context.datasetIndex;
                    const c = context.dataIndex;
                    return `True: ${classes[r]} → Pred: ${classes[c]} = ${matrix[r][c]}`;
                  },
                },
              },
            },
          }}
        />
      </div>
    </div>
  );
}

function TrainingCurves({ dataset, model }) {
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!dataset || !model) return;
    setLoading(true);
    fetch(`/api/training_history?dataset=${dataset}&model_name=${model}`)
      .then((r) => r.json())
      .then((d) => setHistory(d.history))
      .catch(() => setHistory(null))
      .finally(() => setLoading(false));
  }, [dataset, model]);
  if (loading) return <div className="skeleton">Loading training curves...</div>;
  if (!history) return <p className="muted">Training history unavailable.</p>;
  const epochs = history.accuracy.map((_, i) => i + 1);
  const chartData = {
    labels: epochs,
    datasets: [
      { label: "Train Acc", data: history.accuracy, borderColor: "#22c55e", backgroundColor: "rgba(34,197,94,0.1)", tension: 0.4 },
      { label: "Val Acc", data: history.val_accuracy, borderColor: "#3b82f6", backgroundColor: "rgba(59,130,246,0.1)", tension: 0.4 },
      { label: "Train Loss", data: history.loss, borderColor: "#ef4444", backgroundColor: "rgba(239,68,68,0.1)", tension: 0.4, yAxisID: "y1" },
      { label: "Val Loss", data: history.val_loss, borderColor: "#f97316", backgroundColor: "rgba(249,115,22,0.1)", tension: 0.4, yAxisID: "y1" },
    ],
  };
  return (
    <div style={{ height: "280px" }}>
      <Line
        data={chartData}
        options={{
          maintainAspectRatio: false,
          scales: {
            y: { min: 0, max: 1, title: { display: true, text: "Accuracy" } },
            y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "Loss" } },
          },
          plugins: { legend: { position: "top" }, title: { display: true, text: "Training History" } },
        }}
      />
    </div>
  );
}

function ValidationCard({ dataset, model }) {
  const [val, setVal] = useState(null);
  useEffect(() => {
    if (!dataset || !model) return;
    fetch(`/api/validation?dataset=${dataset}&model_name=${model}`)
      .then((r) => r.json())
      .then((d) => setVal(d));
  }, [dataset, model]);
  if (!val) return <p className="muted">Validation pending.</p>;
  const labels = val.val_accuracy.map((_, i) => i + 1);
  const data = {
    labels,
    datasets: [
      { label: "Val Acc", data: val.val_accuracy, borderColor: "#10b981", backgroundColor: "rgba(16,185,129,0.2)", tension: 0.3 },
      { label: "Val Loss", data: val.val_loss, borderColor: "#eab308", backgroundColor: "rgba(234,179,8,0.2)", tension: 0.3, yAxisID: "y1" },
    ],
  };
  return (
    <div style={{ height: "220px" }}>
      <Line
        data={data}
        options={{
          maintainAspectRatio: false,
          scales: {
            y: { min: 0, max: 1, title: { display: true, text: "Acc" } },
            y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "Loss" } },
          },
          plugins: { legend: { position: "top" }, title: { display: true, text: "Validation" } },
        }}
      />
    </div>
  );
}

function EdgeSimCard({ dataset, model }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const runSim = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/edge_simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset, model_name: model, quantize: true }),
      });
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setResult(null);
    } finally {
      setLoading(false);
    }
  };
  return (
    <div className="card">
      <div className="eyebrow">Edge Simulation</div>
      <p className="muted">Simulated quantization size/latency/accuracy drop.</p>
      <button className="primary" onClick={runSim} disabled={loading}>
        {loading ? "Simulating..." : "Run Quantization"}
      </button>
      {result && (
        <div className="table-card compact" style={{ marginTop: 10 }}>
          <table>
            <tbody>
              <tr><td>Baseline Size (MB)</td><td>{result.size_mb?.baseline}</td></tr>
              <tr><td>Quantized Size (MB)</td><td>{result.size_mb?.edge}</td></tr>
              <tr><td>Latency (ms)</td><td>{result.latency_ms}</td></tr>
              <tr><td>Accuracy Drop</td><td>{result.accuracy_drop}</td></tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function BiasCard({ dataset }) {
  const [bias, setBias] = useState(null);
  useEffect(() => {
    if (!dataset) return;
    fetch(`/api/bias_summary?dataset=${dataset}`)
      .then((r) => r.json())
      .then((d) => setBias(d))
      .catch(() => setBias(null));
  }, [dataset]);
  if (!bias) return <p className="muted">Bias summary unavailable.</p>;
  return (
    <div className="card">
      <div className="eyebrow">Bias & Coverage</div>
      <p className="muted">Entropy: {bias.entropy?.toFixed(3)} | Total: {bias.total}</p>
      <div className="table-card compact" style={{ maxHeight: 200, overflow: "auto" }}>
        <table>
          <tbody>
            {Object.entries(bias.counts || {}).map(([cls, cnt]) => (
              <tr key={cls}>
                <td>{cls}</td>
                <td>{cnt}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function UploadCard() {
  const [label, setLabel] = useState("");
  const [b64, setB64] = useState("");
  const [message, setMessage] = useState("");
  const handleUpload = async () => {
    setMessage("");
    try {
      const res = await fetch("/api/upload_sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label, image_b64: b64, dataset: "custom" }),
      });
      const data = await res.json();
      setMessage(data.message || data.error || "done");
    } catch (e) {
      setMessage("upload failed");
    }
  };
  return (
    <div className="card">
      <div className="eyebrow">Data Upload</div>
      <p className="muted">Paste base64 image + label to add a sample.</p>
      <div className="form">
        <label className="field">
          <span>Label</span>
          <input value={label} onChange={(e) => setLabel(e.target.value)} />
        </label>
        <label className="field">
          <span>Image (base64 png)</span>
          <textarea value={b64} onChange={(e) => setB64(e.target.value)} rows={3} />
        </label>
        <button className="primary" onClick={handleUpload}>Upload</button>
        {message && <div className="muted">{message}</div>}
      </div>
    </div>
  );
}

function ReportDownload({ dataset }) {
  const download = async () => {
    const res = await fetch(`/api/report?dataset=${dataset}`);
    const data = await res.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `report_${dataset}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  return (
    <div className="card">
      <div className="eyebrow">Report</div>
      <button className="primary" onClick={download}>Download JSON Report</button>
    </div>
  );
}

function ModelEvaluation() {
  const [dataset, setDataset] = useState(DATASETS[0].value);
  const [model, setModel] = useState(MODELS[0].value);
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [matrix, setMatrix] = useState(null);
  useEffect(() => {
    setLoading(true);
    setError("");
    fetch(`/api/performance?dataset=${dataset}`)
      .then((res) => res.json())
      .then((data) => setRows(data.performance || []))
      .catch((err) => setError(err.message || "Failed"))
      .finally(() => setLoading(false));
  }, [dataset]);
  useEffect(() => {
    fetch(`/api/confusion?dataset=${dataset}&model=${model}`)
      .then((r) => r.json())
      .then((d) => setMatrix(d))
      .catch(() => setMatrix(null));
  }, [dataset, model]);
  return (
    <div className="panel">
      <header className="panel-header">
        <div>
          <div className="eyebrow">Model Evaluation</div>
          <h2>Performance Comparison</h2>
        </div>
        <div className="controls">
          <label className="field">
            <span>Dataset</span>
            <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
              {DATASETS.map((d) => (
                <option key={d.value} value={d.value}>{d.label}</option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Model</span>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {MODELS.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          </label>
        </div>
      </header>
      {error && <div className="alert error">{error}</div>}
      {loading ? (
        <div className="skeleton">Loading metrics...</div>
      ) : (
        <div className="table-card">
          <table>
            <thead>
              <tr><th>Model</th><th>Accuracy</th><th>F1-Score</th><th>Parameters</th><th>Training Time</th></tr>
            </thead>
            <tbody>
              {rows.length === 0 && (<tr><td colSpan={5} className="muted">No data</td></tr>)}
              {rows.map((row) => (
                <tr key={`${row.model}-${row.dataset}`}>
                  <td>{row.model}</td><td>{row.accuracy}</td><td>{row.f1_score}</td><td>{row.parameters}</td><td>{row.training_time}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="viz-placeholder">
        <div className="card">
          <div className="eyebrow">Confusion Matrix</div>
          <ConfusionMatrixHeatmap matrix={matrix?.matrix} classes={matrix?.classes} metrics={matrix?.metrics} />
        </div>
        <div className="card">
          <div className="eyebrow">Training & Validation</div>
          <TrainingCurves dataset={dataset} model={model} />
          <div style={{ marginTop: 10 }}>
            <ValidationCard dataset={dataset} model={model} />
          </div>
        </div>
      </div>
    </div>
  );
}

function XAIPlayground() {
  const [dataset, setDataset] = useState(DATASETS[0].value);
  const [model, setModel] = useState(MODELS[0].value);
  const [method, setMethod] = useState(METHODS[0].value);
  const [imageIndex, setImageIndex] = useState(0);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const canAnalyze = useMemo(
    () => !loading && dataset && model && Number.isInteger(Number(imageIndex)) && imageIndex >= 0,
    [loading, dataset, model, imageIndex],
  );

  const handleAnalyze = async () => {
    if (!canAnalyze) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const params = new URLSearchParams({
        dataset_name: dataset,
        model_name: model,
        image_index: imageIndex,
      });
      const endpoint =
        method === "shap" ? "/api/explain_shap" : method === "lime" ? "/api/explain_lime" : "/api/explain";
      const res = await fetch(`${endpoint}?${params.toString()}`);
      if (!res.ok) throw new Error(`Request failed (${res.status})`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to run analysis");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="panel">
      <header className="panel-header">
        <div>
          <div className="eyebrow">XAI Playground</div>
          <h2>Explainability Explorer</h2>
        </div>
        <div className="controls controls-inline">
          <label className="field">
            <span>Dataset</span>
            <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
              {DATASETS.map((d) => (<option key={d.value} value={d.value}>{d.label}</option>))}
            </select>
          </label>
          <label className="field">
            <span>Model</span>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {MODELS.map((m) => (<option key={m.value} value={m.value}>{m.label}</option>))}
            </select>
          </label>
          <label className="field">
            <span>Method</span>
            <select value={method} onChange={(e) => setMethod(e.target.value)}>
              {METHODS.map((m) => (<option key={m.value} value={m.value}>{m.label}</option>))}
            </select>
          </label>
          <label className="field">
            <span>Image Index</span>
            <input type="number" min="0" value={imageIndex} onChange={(e) => setImageIndex(Number(e.target.value))} />
          </label>
          <button className="primary" onClick={handleAnalyze} disabled={!canAnalyze}>
            {loading ? "Analyzing..." : "Analyze"}
          </button>
        </div>
      </header>
      {error && <div className="alert error">{error}</div>}
      {result ? (
        <div className="grid two">
          <div className="card">
            <div className="eyebrow">Prediction</div>
            <h3>{result.prediction}</h3>
            <p className="muted">Confidence: {(result.confidence * 100).toFixed(2)}%</p>
            {result.metrics && (
              <p className="muted">
                Fidelity {result.metrics.fidelity?.toFixed(3)} | Stability {result.metrics.stability} | Coverage {result.metrics.coverage ?? "n/a"} | Entropy {result.metrics.entropy ?? "n/a"}
              </p>
            )}
            <div className="image-pair">
              <div>
                <div className="eyebrow">Overlay</div>
                <img src={`data:image/png;base64,${result.image_b64 || ""}`} alt="Overlay" className="viz-img" />
              </div>
              <div>
                <div className="eyebrow">Heatmap</div>
                <img src={`data:image/png;base64,${result.heatmap_b64 || ""}`} alt="Heatmap" className="viz-img" />
              </div>
            </div>
          </div>
          <InterventionPanel dataset={dataset} model={model} />
        </div>
      ) : (
        <div className="placeholder-box tall">Run an analysis to see explanations.</div>
      )}
    </div>
  );
}

function InterventionPanel({ dataset, model }) {
  const [intervention, setIntervention] = useState("augment");
  const [dataId, setDataId] = useState(0);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch("/api/intervene", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          intervention_type: intervention,
          model_name: model,
          data_id: Number(dataId),
          dataset_name: dataset,
        }),
      });
      if (!res.ok) throw new Error(`Request failed (${res.status})`);
      const data = await res.json();
      setResult(data.metrics);
    } catch (err) {
      setError(err.message || "Intervention failed");
    } finally {
      setLoading(false);
    }
  };
  return (
    <div className="card">
      <div className="eyebrow">XAI-Guided Intervention</div>
      <h3>Relabel or Augment</h3>
      <form className="form" onSubmit={handleSubmit}>
        <label className="field">
          <span>Intervention</span>
          <select value={intervention} onChange={(e) => setIntervention(e.target.value)}>
            <option value="relabel">Relabel</option>
            <option value="augment">Augment</option>
          </select>
        </label>
        <label className="field">
          <span>Data ID</span>
          <input type="number" min="0" value={dataId} onChange={(e) => setDataId(Number(e.target.value))} />
        </label>
        <button className="primary" type="submit" disabled={loading}>{loading ? "Applying..." : "Apply"}</button>
      </form>
      {error && <div className="alert error">{error}</div>}
      {result && (
        <div className="table-card compact">
          <table>
            <thead><tr><th></th><th>Accuracy</th><th>F1</th></tr></thead>
            <tbody>
              <tr><td>Before</td><td>{result.before.accuracy}</td><td>{result.before.f1_score}</td></tr>
              <tr><td>After</td><td>{result.after.accuracy}</td><td>{result.after.f1_score}</td></tr>
              <tr><td>Δ</td><td>{result.delta.accuracy}</td><td>{result.delta.f1_score}</td></tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function NeuralNetworkVisualization({ layers }) {
  if (!layers || layers.length === 0) return <p className="muted">No layer data.</p>;
  const nodes = [];
  const edges = [];
  const nodeWidth = 150;
  const nodeHeight = 60;
  const layerSpacing = 220;
  let xPos = 50;
  let nodeId = 0;
  layers.forEach((layer) => {
    const nodeIdStr = `node-${nodeId}`;
    nodes.push({
      id: nodeIdStr,
      position: { x: xPos, y: 80 + nodeId * (nodeHeight + 10) },
      data: {
        label: (
          <div style={{ fontSize: "10px", textAlign: "center" }}>
            <div style={{ fontWeight: "bold" }}>{layer.name}</div>
            <div style={{ color: "#94a3b8" }}>{layer.type}</div>
            {layer.filters && <div style={{ fontSize: "9px" }}>Filters: {layer.filters}</div>}
            {layer.units && <div style={{ fontSize: "9px" }}>Units: {layer.units}</div>}
          </div>
        ),
      },
      style: { background: "#0f172a", color: "#e5e7eb", border: "1px solid #3b82f6", width: nodeWidth, height: nodeHeight },
    });
    if (nodeId > 0) {
      edges.push({ id: `edge-${nodeId}`, source: `node-${nodeId - 1}`, target: nodeIdStr, animated: true, style: { stroke: "#3b82f6" } });
    }
    xPos += layerSpacing;
    nodeId++;
  });
  return (
    <div style={{ width: "100%", height: "420px", border: "1px solid #334155", borderRadius: "8px" }}>
      <ReactFlow nodes={nodes} edges={edges} fitView>
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  );
}

function DataTraining() {
  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState(null);
  const [summary, setSummary] = useState("");
  const [layers, setLayers] = useState([]);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const datasetForBias = "mnist";

  useEffect(() => {
    fetch("/api/models")
      .then((r) => r.json())
      .then((d) => setModels(d.models || []))
      .catch(() => setModels([]));
  }, []);

  const handleSelect = async (model) => {
    setSelected(model);
    setSummary("");
    setLayers([]);
    setLoadingSummary(true);
    try {
      const params = new URLSearchParams({ dataset: model.dataset || "mnist", model_name: model.model });
      const res = await fetch(`/api/model_structure?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        setSummary(data.summary || "");
        setLayers(data.layers || []);
      } else {
        setSummary("Summary unavailable.");
      }
    } catch (e) {
      setSummary("Summary unavailable.");
    } finally {
      setLoadingSummary(false);
    }
  };

  return (
    <div className="panel">
      <header className="panel-header">
        <div>
          <div className="eyebrow">Data & Training</div>
          <h2>Pipeline Overview</h2>
        </div>
      </header>
      <div className="grid two">
        <div className="card">
          <div className="eyebrow">Saved Models</div>
          <div className="table-card compact" style={{ maxHeight: 260, overflow: "auto" }}>
            <table>
              <thead><tr><th>File</th><th>Model</th><th>Dataset</th><th>Aug</th></tr></thead>
              <tbody>
                {models.length === 0 && (<tr><td colSpan={4} className="muted">No saved models found.</td></tr>)}
                {models.map((m) => (
                  <tr key={m.file} className={selected?.file === m.file ? "row-active" : ""} onClick={() => handleSelect(m)} style={{ cursor: "pointer" }}>
                    <td>{m.file}</td><td>{m.model}</td><td>{m.dataset}</td><td>{m.augmented ? "Yes" : "No"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ marginTop: 12 }}>
            <EdgeSimCard dataset={selected?.dataset || "mnist"} model={selected?.model || "mobilenet_v2"} />
          </div>
        </div>

        <div className="card" style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          <div className="eyebrow">Neural Network Architecture</div>
          {selected ? (
            loadingSummary ? (
              <div className="skeleton">Loading architecture...</div>
            ) : (
              <>
                <NeuralNetworkVisualization layers={layers} />
                {summary && (
                  <details style={{ marginTop: "10px", width: "100%" }}>
                    <summary style={{ cursor: "pointer", fontWeight: "bold" }}>Text Summary</summary>
                    <pre className="summary" style={{ fontSize: "10px", maxHeight: "200px", overflow: "auto" }}>{summary}</pre>
                  </details>
                )}
              </>
            )
          ) : (
            <p className="muted">Select a saved model to view its architecture.</p>
          )}
        </div>
      </div>

      <div className="grid two" style={{ marginTop: 16 }}>
        <BiasCard dataset={datasetForBias} />
        <UploadCard />
      </div>
      <div style={{ marginTop: 12 }}>
        <ReportDownload dataset={datasetForBias} />
      </div>
    </div>
  );
}

function App() {
  const [active, setActive] = useState("evaluation");
  const renderView = () => {
    switch (active) {
      case "evaluation":
        return <ModelEvaluation />;
      case "playground":
        return <XAIPlayground />;
      default:
        return <DataTraining />;
    }
  };
  return (
    <div className="layout">
      <Sidebar active={active} onChange={setActive} />
      <main>{renderView()}</main>
    </div>
  );
}

export default App;
