"use client";
import { useEffect, useMemo, useState } from "react";

type Row = {
  segment: string; policies: number; exposure: number; claims: number;
  observed_frequency: number; glm_frequency: number; lgb_frequency: number;
  observed_pure_premium: number; glm_pure_premium: number; lgb_pure_premium: number; lgb_relativity: number;
};
type Note = { dimension: string; verified: boolean; note: string | null; model: string; problems: string[] };
type Metrics = any;

const BASE = process.env.NEXT_PUBLIC_BASE_PATH || "";
const API = process.env.NEXT_PUBLIC_API_URL || "";
const DIMS: Record<string, string> = {
  BonusMalusBand: "Bonus-malus", DrivAgeBand: "Driver age", VehAgeBand: "Vehicle age",
  Area: "Area (A rural - F urban)", VehGas: "Fuel", Region: "Region",
};

async function load<T>(apiPath: string, file: string): Promise<T> {
  // Use the live API when one is configured, otherwise the exported holdout snapshot.
  const url = API ? `${API}${apiPath}` : `${BASE}/data/${file}`;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

const pct = (x: number) => `${(100 * x).toFixed(2)}%`;
const eur = (x: number) => `€${Math.round(x).toLocaleString()}`;

export default function Page() {
  const [segs, setSegs] = useState<Record<string, Row[]> | null>(null);
  const [notes, setNotes] = useState<Note[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [dim, setDim] = useState("BonusMalusBand");

  useEffect(() => {
    load<Record<string, Row[]>>("/segments-all", "segments.json").then(setSegs).catch(console.error);
    load<Note[]>("/notes", "notes.json").then(setNotes).catch(console.error);
    load<Metrics>("/metrics", "metrics.json").then(setMetrics).catch(console.error);
  }, []);

  const rows = segs?.[dim] ?? [];
  const maxRel = useMemo(() => Math.max(1, ...rows.map((r) => r.lgb_relativity)), [rows]);
  const note = notes.find((n) => n.dimension === dim);

  return (
    <main>
      <h1>Claim Frequency &amp; Severity Pricing</h1>
      <p className="sub">
        678,013 French motor liability policies (freMTPL2). Poisson and gamma GLMs against LightGBM, scored on a
        25% policy holdout. <a href="https://github.com/AbhayJuloori/claim-frequency-severity-fremtpl2">Code</a>
      </p>

      {metrics && (
        <div className="grid">
          <div className="card"><b>{metrics.frequency.lgb.gini.toFixed(3)} vs {metrics.frequency.glm.gini.toFixed(3)}</b><span>frequency Gini, LightGBM vs GLM</span></div>
          <div className="card"><b>{metrics.frequency.lgb.poisson_deviance.toFixed(4)}</b><span>LightGBM Poisson deviance (GLM {metrics.frequency.glm.poisson_deviance.toFixed(4)}, mean-only {metrics.frequency.dummy.poisson_deviance.toFixed(4)})</span></div>
          <div className="card"><b>{metrics.pure_premium.lgb.gini.toFixed(3)} vs {metrics.pure_premium.glm.gini.toFixed(3)}</b><span>pure premium Gini, LightGBM vs GLM</span></div>
          <div className="card"><b>{metrics.pure_premium.lgb.ratio_pred_to_obs.toFixed(3)}</b><span>LightGBM predicted / observed losses on holdout</span></div>
        </div>
      )}

      <h2>Compare segments</h2>
      <div className="tabs" role="group" aria-label="Rating factor">
        {Object.entries(DIMS).map(([k, label]) => (
          <button key={k} aria-pressed={dim === k} onClick={() => setDim(k)}>{label}</button>
        ))}
      </div>
      <div className="scroll">
        <table>
          <thead>
            <tr><th>Segment</th><th>Exposure (yrs)</th><th>Claims</th><th>Observed freq</th><th>GLM freq</th><th>LightGBM freq</th>
              <th>Observed PP</th><th>LightGBM PP</th><th>Relativity</th></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.segment}>
                <td>{r.segment}</td><td>{Math.round(r.exposure).toLocaleString()}</td><td>{r.claims.toLocaleString()}</td>
                <td>{pct(r.observed_frequency)}</td><td>{pct(r.glm_frequency)}</td><td>{pct(r.lgb_frequency)}</td>
                <td>{eur(r.observed_pure_premium)}</td><td>{eur(r.lgb_pure_premium)}</td>
                <td><span className="bar" style={{ width: `${(60 * r.lgb_relativity) / maxRel}px` }} /> {r.lgb_relativity.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="legend">PP = pure premium, expected loss per policy-year. Relativity = segment PP / portfolio PP. Holdout policies only.</p>
      {note && (note.verified ? (
        <div className="note"><strong>Segment note</strong> (drafted by {note.model}, every number and claim checked against the table): {note.note}</div>
      ) : (
        <div className="note rejected"><strong>No note.</strong> The drafted note failed the check ({note.problems.join("; ")}), so only the table is shown.</div>
      ))}

      <Predict />
    </main>
  );
}

function Predict() {
  const [form, setForm] = useState({ Area: "C", VehBrand: "B12", VehGas: "Regular", Region: "R11", VehPower: 6, VehAge: 3, DrivAge: 40, BonusMalus: 50, Density: 1200 });
  const [out, setOut] = useState<any>(null);
  const [err, setErr] = useState("");
  const num = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"];
  if (!API) {
    return (
      <>
        <h2>Price a policy</h2>
        <p className="legend">The live pricing form needs the FastAPI backend. Run it locally (see README) and build the UI with NEXT_PUBLIC_API_URL set.</p>
      </>
    );
  }
  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setErr("");
    const r = await fetch(`${API}/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify([form]) });
    if (!r.ok) { setErr(`API error ${r.status}`); return; }
    setOut((await r.json())[0]);
  };
  return (
    <>
      <h2>Price a policy</h2>
      <form className="predict" onSubmit={submit}>
        {Object.entries(form).map(([k, v]) => (
          <label key={k}>{k}
            <input value={v} type={num.includes(k) ? "number" : "text"}
              onChange={(e) => setForm({ ...form, [k]: num.includes(k) ? Number(e.target.value) : e.target.value })} />
          </label>
        ))}
        <button className="go" type="submit">Predict</button>
      </form>
      {err && <p className="legend">{err}</p>}
      {out && (
        <div className="grid" style={{ marginTop: 12 }}>
          {(["glm", "lgb"] as const).map((k) => (
            <div className="card" key={k}><b>{eur(out[k].pure_premium)}</b>
              <span>{k === "glm" ? "GLM" : "LightGBM"}: {pct(out[k].frequency)} claims/yr × {eur(out[k].severity)} per claim</span></div>
          ))}
        </div>
      )}
    </>
  );
}
