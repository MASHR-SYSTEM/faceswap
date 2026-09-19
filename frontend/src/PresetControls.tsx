import { useState } from "react";
import type { EffectConfig } from "./types";
const key = "faceswap.effect-presets.v1";
export function PresetControls({ effect, onSelect }: { effect: EffectConfig; onSelect: (value: EffectConfig) => void }) {
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [presets, setPresets] = useState<Record<string, EffectConfig>>(() => {
    try { return JSON.parse(localStorage.getItem(key) ?? "{}"); } catch { return {}; }
  });
  function save() {
    if (!name.trim()) return;
    const next = { ...presets, [name.trim()]: effect };
    try { localStorage.setItem(key, JSON.stringify(next)); setPresets(next); setError(""); }
    catch { setError("This browser could not save the preset."); }
  }
  return <section aria-label="Effect presets">
    <label>Saved preset<select defaultValue="" onChange={event => {
      const chosen = presets[event.target.value]; if (chosen) onSelect({ ...effect, ...chosen });
    }}><option value="">Choose a preset</option>{Object.keys(presets).map(n => <option key={n}>{n}</option>)}</select></label>
    <label>Preset name<input value={name} onChange={event => setName(event.target.value)} maxLength={80} /></label>
    <button onClick={save} disabled={!name.trim()}>Save preset</button>
    {error && <p role="alert">{error}</p>}
  </section>;
}
