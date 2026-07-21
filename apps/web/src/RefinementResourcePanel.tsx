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
    const validatedMinimum = settings?.minimum_vram_gib;
    const minimum = validatedMinimum ?? 4;
    if (!Number.isFinite(parsed) || parsed < minimum || parsed > 192) {
      setError(
        validatedMinimum == null
          ? "VRAM setting must be between 4 and 192 GiB."
          : `${settings?.model_id ?? "The selected model"} requires at least ${minimum} GiB for its validated local refinement profile. This applies to both hard limits and soft monitoring targets.`,
      );
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
  const minimumVram = settings?.minimum_vram_gib ?? 4;

  return (
    <section className="panel refinement-resource-panel" aria-labelledby="refinement-resource-heading">
      <header className="panel-header">
        <div>
          <p className="eyebrow">Resource boundary</p>
          <h2 id="refinement-resource-heading">Adapter-only refinement</h2>
          <p>AutoTrainer never enables full-model training. Set a hard allocator limit or a soft monitoring target.</p>
        </div>
        <span className="status-chip good">Base weights frozen</span>
      </header>

      <div className="refinement-resource-form">
        <label><span>VRAM limit or target</span><div className="vram-input"><input type="number" min={minimumVram} max="192" step="0.5" value={maxVram} onChange={(event) => { setMaxVram(event.target.value); setSaved(false); }} disabled={busy || disabled} /><b>GiB</b></div><small>{settings === null ? "Loading the model's VRAM requirements…" : settings.minimum_vram_gib == null ? "No validated minimum is published for this model." : `${settings.model_id ?? "This model"} has a validated minimum of ${minimumVram} GiB.`}</small></label>
        <fieldset>
          <legend>Enforcement</legend>
          <label><input type="radio" name="vram-enforcement" checked={enforcement === "hard"} onChange={() => { setEnforcement("hard"); setSaved(false); }} disabled={busy || disabled} /><span><strong>Hard limit</strong><small>Install a CUDA allocator cap. The run stops instead of exceeding it.</small></span></label>
          <label><input type="radio" name="vram-enforcement" checked={enforcement === "soft"} onChange={() => { setEnforcement("soft"); setSaved(false); }} disabled={busy || disabled} /><span><strong>Soft target</strong><small>Report the target in telemetry without installing an allocator cap. Training may use more.</small></span></label>
        </fieldset>
        <button className="primary-button" type="button" onClick={() => void save()} disabled={busy || disabled || !changed}>{busy ? "Saving…" : saved ? "Saved" : "Save GPU setting"}</button>
      </div>
      {error && <div className="source-error" role="alert">{error}</div>}
    </section>
  );
}
