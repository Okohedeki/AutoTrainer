import { useEffect, useState } from "react";
import {
  ApiClientError,
  getRefinementSettings,
  setRefinementSettings,
  type RefinementSettings,
} from "./api";

export default function RefinementResourcePanel({ disabled = false }: { disabled?: boolean }) {
  const [settings, setSettings] = useState<RefinementSettings | null>(null);
  const [maxVram, setMaxVram] = useState("20");
  const [enforcement, setEnforcement] = useState<"hard" | "soft">("hard");
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getRefinementSettings(controller.signal)
      .then((next) => {
        setSettings(next);
        setMaxVram(String(next.vram.max_gib));
        setEnforcement(next.vram.enforcement);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Refinement limits could not be loaded.");
      });
    return () => controller.abort();
  }, []);

  const save = async () => {
    const parsed = Number(maxVram);
    if (!Number.isFinite(parsed) || parsed < 4 || parsed > 192) {
      setError("VRAM limit must be between 4 and 192 GiB.");
      return;
    }
    setBusy(true);
    setSaved(false);
    setError(null);
    try {
      const next = await setRefinementSettings({ max_vram_gib: parsed, enforcement });
      setSettings(next);
      setMaxVram(String(next.vram.max_gib));
      setSaved(true);
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "Refinement limits could not be saved.");
    } finally {
      setBusy(false);
    }
  };

  const changed = Boolean(settings) && (
    Number(maxVram) !== settings?.vram.max_gib || enforcement !== settings?.vram.enforcement
  );

  return (
    <section className="panel refinement-resource-panel" aria-labelledby="refinement-resource-heading">
      <header className="panel-header">
        <div>
          <p className="eyebrow">Resource boundary</p>
          <h2 id="refinement-resource-heading">Adapter-only refinement</h2>
          <p>AutoTrainer never enables full-model training. Choose how much of your GPU this run may use.</p>
        </div>
        <span className="status-chip good">Base weights frozen</span>
      </header>

      <div className="refinement-resource-form">
        <label><span>Maximum VRAM</span><div className="vram-input"><input type="number" min="4" max="192" step="0.5" value={maxVram} onChange={(event) => { setMaxVram(event.target.value); setSaved(false); }} disabled={busy || disabled} /><b>GiB</b></div></label>
        <fieldset>
          <legend>Enforcement</legend>
          <label><input type="radio" name="vram-enforcement" checked={enforcement === "hard"} onChange={() => { setEnforcement("hard"); setSaved(false); }} disabled={busy || disabled} /><span><strong>Hard limit</strong><small>Install a CUDA allocator cap. The run stops instead of exceeding it.</small></span></label>
          <label><input type="radio" name="vram-enforcement" checked={enforcement === "soft"} onChange={() => { setEnforcement("soft"); setSaved(false); }} disabled={busy || disabled} /><span><strong>Soft target</strong><small>Pass the ceiling to model loading and report observed use, but allow runtime recovery.</small></span></label>
        </fieldset>
        <button className="primary-button" type="button" onClick={() => void save()} disabled={busy || disabled || !changed}>{busy ? "Saving…" : saved ? "Saved" : "Save GPU limit"}</button>
      </div>
      {error && <div className="source-error" role="alert">{error}</div>}
    </section>
  );
}
